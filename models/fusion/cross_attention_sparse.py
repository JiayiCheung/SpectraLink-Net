# models/fusion/cross_attention_sparse.py
"""
稀疏采样交叉注意力模块 (v4.1)
通过只选择重要的空间位置来大幅降低计算复杂度

核心思想：
- 不是所有 N=D×H×W 个空间位置都需要与频域交互
- 只选择 Top-K 重要位置做交叉注意力
- 非采样位置通过插值或原始值保留

复杂度对比：
- 原版: O(N × M) = O(100万 × 216) = 2.16亿
- 稀疏: O(K × M) = O(4096 × 216) = 88万 (减少 250 倍!)

v4.1 更新：
- 支持自适应采样数量 (sample_ratio / auto)
- 支持基于内容的动态采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class AdaptiveSampleCalculator:
    """
    自适应采样数量计算器
    
    支持多种采样策略：
    1. 固定数量: num_samples=4096
    2. 固定比例: sample_ratio=0.5% (自动根据分辨率计算)
    3. 自动模式: 'auto' (根据分辨率智能选择)
    
    使用示例:
        # 固定数量
        calc = AdaptiveSampleCalculator(mode='fixed', num_samples=4096)
        
        # 固定比例 (0.5%)
        calc = AdaptiveSampleCalculator(mode='ratio', sample_ratio=0.005)
        
        # 自动模式 (推荐)
        calc = AdaptiveSampleCalculator(mode='auto')
        
        k = calc.compute(num_tokens=1048576)  # 输入 token 数量
    """
    
    def __init__(
        self,
        mode: str = 'auto',
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 512,
        max_samples: int = 16384,
        auto_scale: float = 4.0
    ):
        """
        参数:
            mode: 采样模式
                - 'fixed': 固定数量 (使用 num_samples)
                - 'ratio': 固定比例 (使用 sample_ratio)
                - 'auto': 自动模式 (基于 sqrt(N) 智能选择)
            num_samples: 固定采样数量 (mode='fixed' 时使用)
            sample_ratio: 采样比例 0-1 (mode='ratio' 时使用, 如 0.005=0.5%)
            min_samples: 最小采样数量
            max_samples: 最大采样数量
            auto_scale: 自动模式的缩放因子 (K = sqrt(N) * scale)
        """
        self.mode = mode
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.auto_scale = auto_scale
        
        # 验证参数
        if mode == 'fixed' and num_samples is None:
            raise ValueError("mode='fixed' 需要指定 num_samples")
        if mode == 'ratio' and sample_ratio is None:
            raise ValueError("mode='ratio' 需要指定 sample_ratio")
    
    def compute(self, num_tokens: int) -> int:
        """
        计算采样数量
        
        参数:
            num_tokens: 总 token 数量 (N = D × H × W)
            
        返回:
            采样数量 K
        """
        if self.mode == 'fixed':
            k = self.num_samples
            
        elif self.mode == 'ratio':
            k = int(num_tokens * self.sample_ratio)
            
        elif self.mode == 'auto':
            # 自动模式：根据分辨率智能选择
            # 基于经验公式：K = sqrt(N) * scale_factor
            # 这样在不同分辨率下保持相对合理的采样率
            #
            # 分辨率      N          sqrt(N)    K (scale=4)   采样率
            # 64×128×128  1,048,576  1024       4096          0.39%
            # 32×64×64    131,072    362        1448          1.10%
            # 16×32×32    16,384     128        512           3.13%
            k = int(math.sqrt(num_tokens) * self.auto_scale)
            
        else:
            raise ValueError(f"未知的采样模式: {self.mode}")
        
        # 应用上下界约束
        k = max(self.min_samples, min(self.max_samples, k))
        
        # 确保不超过总 token 数
        k = min(k, num_tokens)
        
        return k
    
    def get_info(self, num_tokens: int) -> dict:
        """获取采样信息"""
        k = self.compute(num_tokens)
        return {
            'mode': self.mode,
            'num_tokens': num_tokens,
            'num_samples': k,
            'sample_ratio': k / num_tokens * 100,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples,
        }
    
    def __repr__(self):
        if self.mode == 'fixed':
            return f"AdaptiveSampleCalculator(mode='fixed', num_samples={self.num_samples})"
        elif self.mode == 'ratio':
            return f"AdaptiveSampleCalculator(mode='ratio', sample_ratio={self.sample_ratio:.4f})"
        else:
            return f"AdaptiveSampleCalculator(mode='auto', scale={self.auto_scale}, range=[{self.min_samples}, {self.max_samples}])"


class SparseImportanceScorer(nn.Module):
    """
    空间位置重要性评分器
    
    学习预测每个空间位置的重要性分数，用于选择 Top-K
    
    评分策略:
    - 基于空域特征本身的重要性
    - 可选: 结合频域信息进行条件评分
    
    参数:
        spatial_channels: 空域特征通道数
        freq_channels: 频域特征通道数 (用于 freq_gate)
        reduction: 通道缩减比例
        use_freq_gate: 是否使用频域门控
    """
    
    def __init__(
        self, 
        spatial_channels: int, 
        freq_channels: int = None,
        reduction: int = 4, 
        use_freq_gate: bool = True
    ):
        super().__init__()
        
        self.use_freq_gate = use_freq_gate
        
        # 基础重要性评分器 - 处理空域特征
        self.scorer = nn.Sequential(
            nn.Linear(spatial_channels, spatial_channels // reduction),
            nn.GELU(),
            nn.Linear(spatial_channels // reduction, 1)
        )
        
        # 可选：结合频域信息来评分 - 处理频域特征
        if use_freq_gate:
            # 使用 freq_channels，若未指定则回退到 spatial_channels
            freq_ch = freq_channels if freq_channels is not None else spatial_channels
            self.freq_gate = nn.Sequential(
                nn.Linear(freq_ch, max(freq_ch // reduction, 1)),
                nn.GELU(),
                nn.Linear(max(freq_ch // reduction, 1), 1),
                nn.Sigmoid()
            )
    
    def forward(self, spatial_tokens: torch.Tensor, freq_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算每个空间位置的重要性分数
        
        参数:
            spatial_tokens: [B, N, C] 空域 tokens
            freq_tokens: [B, M, C] 频域 tokens (可选，用于条件评分)
            
        返回:
            importance: [B, N] 重要性分数
        """
        # 基础重要性（基于空域特征本身）
        importance = self.scorer(spatial_tokens).squeeze(-1)  # [B, N]
        
        # 如果提供频域信息，用频域调制重要性
        if self.use_freq_gate and freq_tokens is not None:
            # 频域全局特征
            freq_global = freq_tokens.mean(dim=1)  # [B, C]
            freq_weight = self.freq_gate(freq_global)  # [B, 1]
            
            # 调制（频域认为重要的区域权重更高）
            importance = importance * (1 + freq_weight)
        
        return importance


class SparseCrossAttention(nn.Module):
    """
    稀疏采样交叉注意力
    
    只在 Top-K 重要空间位置上做交叉注意力，大幅降低计算量
    
    参数:
        spatial_channels: 空域特征通道数
        freq_channels: 频域 token 通道数
        num_heads: 注意力头数
        sample_mode: 采样模式 ('fixed', 'ratio', 'auto')
        num_samples: 固定采样数量 (mode='fixed')
        sample_ratio: 采样比例 (mode='ratio')
        min_samples: 最小采样数量
        max_samples: 最大采样数量
        dropout: dropout 比例
        use_freq_guided_sampling: 是否用频域信息引导采样
    """
    
    def __init__(
        self,
        spatial_channels: int,
        freq_channels: int,
        num_heads: int = 4,
        sample_mode: str = 'auto',
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 512,
        max_samples: int = 16384,
        dropout: float = 0.0,
        use_freq_guided_sampling: bool = True
    ):
        super().__init__()
        
        self.spatial_channels = spatial_channels
        self.freq_channels = freq_channels
        self.num_heads = num_heads
        self.use_freq_guided_sampling = use_freq_guided_sampling
        
        self.d_model = spatial_channels
        self.head_dim = self.d_model // num_heads
        
        assert self.d_model % num_heads == 0, \
            f"d_model({self.d_model}) 必须能被 num_heads({num_heads}) 整除"
        
        # 自适应采样计算器
        self.sample_calculator = AdaptiveSampleCalculator(
            mode=sample_mode,
            num_samples=num_samples,
            sample_ratio=sample_ratio,
            min_samples=min_samples,
            max_samples=max_samples
        )
        
        # 重要性评分器
        self.importance_scorer = SparseImportanceScorer(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            reduction=4,
            use_freq_gate=use_freq_guided_sampling
        )
        
        # Q/K/V 投影
        self.q_proj = nn.Linear(spatial_channels, self.d_model)
        self.k_proj = nn.Linear(freq_channels, self.d_model)
        self.v_proj = nn.Linear(freq_channels, self.d_model)
        self.out_proj = nn.Linear(self.d_model, spatial_channels)
        
        # Dropout
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        
        # 非采样位置的处理：学习一个默认的增强向量
        self.default_enhancement = nn.Parameter(torch.zeros(1, 1, spatial_channels))
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        spatial_tokens: torch.Tensor,
        freq_tokens: torch.Tensor,
        return_indices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        稀疏交叉注意力前向传播
        
        参数:
            spatial_tokens: [B, N, C_sp] 空域 tokens
            freq_tokens: [B, M, C_fd] 频域 tokens
            return_indices: 是否返回采样的索引（用于可视化）
            
        返回:
            output: [B, N, C_sp] 增强后的空域 tokens
            (可选) indices: [B, K] 采样的索引
        """
        B, N, C = spatial_tokens.shape
        M = freq_tokens.shape[1]
        
        # 自适应计算采样数量
        K = self.sample_calculator.compute(N)
        
        # ========== 1. 计算重要性并选择 Top-K ==========
        if self.use_freq_guided_sampling:
            importance = self.importance_scorer(spatial_tokens, freq_tokens)
        else:
            importance = self.importance_scorer(spatial_tokens)
        
        # 选择 Top-K 重要位置
        topk_values, topk_indices = importance.topk(K, dim=-1)  # [B, K]
        
        # ========== 2. 提取 Top-K tokens ==========
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
        sampled_tokens = torch.gather(spatial_tokens, 1, topk_indices_expanded)  # [B, K, C]
        
        # ========== 3. 在采样的 tokens 上做交叉注意力 ==========
        Q = self.q_proj(sampled_tokens)  # [B, K, d_model]
        K_proj = self.k_proj(freq_tokens)  # [B, M, d_model]
        V = self.v_proj(freq_tokens)  # [B, M, d_model]
        
        # 多头 reshape
        Q = Q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, D]
        K_proj = K_proj.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D]
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D]
        
        # Flash Attention (如果可用)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K_proj, V,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, H, K, D]
        else:
            attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, self.d_model)
        attn_output = self.out_proj(attn_output)  # [B, K, C]
        
        # ========== 4. 将结果散射回原位置 ==========
        # 初始化输出为默认增强值
        # 注意：需要保证 dtype 一致（AMP 训练时 attn_output 可能是 float16）
        output = self.default_enhancement.expand(B, N, -1).to(attn_output.dtype).clone()  # [B, N, C]
        
        # 将注意力输出散射到采样位置
        output.scatter_(1, topk_indices_expanded, attn_output)
        
        if return_indices:
            return output, topk_indices
        return output
    
    def get_sample_info(self, num_tokens: int) -> dict:
        """获取采样信息"""
        return self.sample_calculator.get_info(num_tokens)


class SparseFreqGuidedAttentionPair(nn.Module):
    """
    稀疏采样版本的频域引导注意力对
    
    替换原有的 FreqGuidedAttentionPair，使用稀疏采样来加速
    
    参数:
        spatial_channels: 空域特征通道数
        freq_channels: 频域 token 通道数
        num_heads: 注意力头数
        sample_mode: 采样模式 ('fixed', 'ratio', 'auto')
        num_samples: 固定采样数量 (mode='fixed')
        sample_ratio: 采样比例 (mode='ratio')
        dropout: dropout 比例
        use_freq_guided_sampling: 是否用频域信息引导采样
    """
    
    def __init__(
        self,
        spatial_channels: int,
        freq_channels: int,
        num_heads: int = 4,
        sample_mode: str = 'auto',
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 512,
        max_samples: int = 16384,
        dropout: float = 0.0,
        use_freq_guided_sampling: bool = True
    ):
        super().__init__()
        
        self.spatial_channels = spatial_channels
        self.freq_channels = freq_channels
        self.sample_mode = sample_mode
        
        # 稀疏交叉注意力
        self.cross_attn = SparseCrossAttention(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            sample_mode=sample_mode,
            num_samples=num_samples,
            sample_ratio=sample_ratio,
            min_samples=min_samples,
            max_samples=max_samples,
            dropout=dropout,
            use_freq_guided_sampling=use_freq_guided_sampling
        )
        
        # Pre-norm
        self.norm_spatial = nn.LayerNorm(spatial_channels)
        self.norm_freq = nn.LayerNorm(freq_channels)
        
        # 可学习的残差缩放 (参考3D-TransUNet的LayerScale，初始值1e-5)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-5)
    
    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            spatial_feat: [B, C, D, H, W] 空域特征
            freq_tokens: [B, M, C] 频域 tokens
            
        返回:
            output: [B, C, D, H, W] 增强后的空域特征
        """
        B, C, D, H, W = spatial_feat.shape
        
        # 1. 展平空域特征
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # 2. 保存残差
        identity = spatial_tokens
        
        # 3. Pre-norm
        spatial_tokens_normed = self.norm_spatial(spatial_tokens)
        freq_tokens_normed = self.norm_freq(freq_tokens)
        
        # 4. 稀疏交叉注意力
        attn_out = self.cross_attn(spatial_tokens_normed, freq_tokens_normed)
        
        # 5. 残差连接
        output = identity + self.gamma * attn_out
        
        # 6. 恢复空间形状
        output = output.transpose(1, 2).view(B, C, D, H, W)
        
        return output
    
    def get_sample_info(self, spatial_shape: Tuple[int, int, int]) -> dict:
        """获取采样信息"""
        D, H, W = spatial_shape
        return self.cross_attn.get_sample_info(D * H * W)


class ContentAwareSparseAttention(nn.Module):
    """
    基于内容的自适应稀疏注意力
    
    根据输入内容动态调整采样数量：
    - 复杂样本（血管多、细节多）→ 多采样
    - 简单样本（背景多）→ 少采样
    
    复杂度评估基于:
    - 特征方差（高方差 = 复杂）
    - 梯度幅值（高梯度 = 边缘多）
    """
    
    def __init__(
        self,
        spatial_channels: int,
        freq_channels: int,
        num_heads: int = 4,
        base_samples: int = 2048,
        min_samples: int = 512,
        max_samples: int = 8192,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.base_samples = base_samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        
        # 复杂度评估器（轻量级）
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spatial_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 重要性评分器
        self.importance_scorer = SparseImportanceScorer(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels
        )
        
        # 注意力组件
        self.q_proj = nn.Linear(spatial_channels, spatial_channels)
        self.k_proj = nn.Linear(freq_channels, spatial_channels)
        self.v_proj = nn.Linear(freq_channels, spatial_channels)
        self.out_proj = nn.Linear(spatial_channels, spatial_channels)
        
        self.num_heads = num_heads
        self.head_dim = spatial_channels // num_heads
        self.dropout_p = dropout
        
        self.norm_spatial = nn.LayerNorm(spatial_channels)
        self.norm_freq = nn.LayerNorm(freq_channels)
        # 参考3D-TransUNet的LayerScale，初始值1e-5
        self.gamma = nn.Parameter(torch.ones(1) * 1e-5)
    
    def forward(self, spatial_feat: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = spatial_feat.shape
        N = D * H * W
        
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        identity = spatial_tokens
        
        # 1. 估计复杂度，决定采样数量
        complexity = self.complexity_estimator(spatial_tokens.transpose(1, 2))  # [B, 1]
        
        # 动态计算采样数量
        # complexity ∈ [0, 1]，0=简单，1=复杂
        num_samples_float = (
            self.min_samples + 
            (self.max_samples - self.min_samples) * complexity
        )  # [B, 1]
        
        # 为简化，使用 batch 内最大值
        K = int(num_samples_float.max().item())
        K = max(self.min_samples, min(self.max_samples, K, N))
        
        # 2. Pre-norm
        spatial_tokens_normed = self.norm_spatial(spatial_tokens)
        freq_tokens_normed = self.norm_freq(freq_tokens)
        
        # 3. 重要性采样
        importance = self.importance_scorer(spatial_tokens_normed, freq_tokens_normed)
        _, topk_indices = importance.topk(K, dim=-1)
        
        # 4. 稀疏注意力
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        sampled_tokens = torch.gather(spatial_tokens_normed, 1, topk_indices_expanded)
        
        Q = self.q_proj(sampled_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(freq_tokens_normed).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(freq_tokens_normed).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            Q, K_proj, V, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, K, C)
        attn_output = self.out_proj(attn_output)
        
        # 5. 散射回原位置
        output = torch.zeros_like(spatial_tokens)
        output.scatter_(1, topk_indices_expanded, attn_output)
        
        # 6. 残差 + reshape
        output = identity + self.gamma * output
        output = output.transpose(1, 2).view(B, C, D, H, W)
        
        return output


# ============================================================================
# 工具函数
# ============================================================================

def get_attention_module(
    attention_type: str,
    spatial_channels: int,
    freq_channels: int,
    num_heads: int = 4,
    sample_mode: str = 'auto',
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    工厂函数：根据类型返回对应的注意力模块
    
    参数:
        attention_type: 
            - 'standard': 标准注意力
            - 'sparse': 稀疏采样注意力
            - 'sparse_auto': 自动采样数量的稀疏注意力
            - 'content_aware': 基于内容的自适应稀疏注意力
        其他参数同各注意力模块
    """
    if attention_type == 'standard':
        from .cross_attention import FreqGuidedAttentionPair
        return FreqGuidedAttentionPair(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            dropout=dropout
        )
    elif attention_type == 'sparse':
        return SparseFreqGuidedAttentionPair(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            sample_mode='fixed' if num_samples else 'auto',
            num_samples=num_samples,
            sample_ratio=sample_ratio,
            dropout=dropout,
            **kwargs
        )
    elif attention_type == 'sparse_auto':
        return SparseFreqGuidedAttentionPair(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            sample_mode='auto',
            dropout=dropout,
            **kwargs
        )
    elif attention_type == 'content_aware':
        return ContentAwareSparseAttention(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"未知的注意力类型: {attention_type}")


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("自适应采样计算器测试")
    print("=" * 70)
    
    # 不同分辨率
    resolutions = [
        (64, 128, 128),   # Level 0
        (32, 64, 64),     # Level 1
        (16, 32, 32),     # Level 2
    ]
    
    # 测试不同模式
    modes = [
        ('fixed', {'num_samples': 4096}),
        ('ratio', {'sample_ratio': 0.005}),  # 0.5%
        ('auto', {}),
    ]
    
    for mode, kwargs in modes:
        print(f"\n模式: {mode}")
        print("-" * 50)
        calc = AdaptiveSampleCalculator(mode=mode, **kwargs)
        print(f"配置: {calc}")
        
        for d, h, w in resolutions:
            n = d * h * w
            info = calc.get_info(n)
            print(f"  [{d}×{h}×{w}] N={n:>10,} → K={info['num_samples']:>6,} ({info['sample_ratio']:.2f}%)")
