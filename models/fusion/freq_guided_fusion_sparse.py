# models/fusion/freq_guided_fusion_sparse.py
"""
支持稀疏采样的频域引导融合模块

可以在配置中选择使用标准注意力或稀疏注意力
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .cross_attention import FreqGuidedAttentionPair
from .cross_attention_sparse import SparseFreqGuidedAttentionPair


class FreqGuidedFusionSparse(nn.Module):
    """
    支持稀疏采样的频域引导融合模块
    
    可配置参数:
        use_sparse: 是否使用稀疏注意力
        num_samples: 稀疏采样数量 (仅当 use_sparse=True 时有效)
    
    其余参数与 FreqGuidedFusion 相同
    """
    
    def __init__(
        self,
        spatial_channels: list,
        freq_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_sparse: bool = False,
        num_samples: int = 4096
    ):
        super().__init__()
        
        assert len(spatial_channels) == 3, "需要3个空域path的通道数"
        
        self.spatial_channels = spatial_channels
        self.freq_channels = freq_channels
        self.out_channels = out_channels
        self.use_sparse = use_sparse
        self.num_samples = num_samples
        
        ch1, ch2, ch3 = spatial_channels
        
        # 选择注意力类型
        AttentionClass = SparseFreqGuidedAttentionPair if use_sparse else FreqGuidedAttentionPair
        
        # 构建注意力对的参数
        def get_attn_kwargs(ch):
            kwargs = {
                'spatial_channels': ch,
                'freq_channels': freq_channels,
                'num_heads': num_heads,
                'dropout': dropout
            }
            if use_sparse:
                kwargs['num_samples'] = num_samples
            return kwargs
        
        # 3对交叉注意力
        self.attn_pair_0 = AttentionClass(**get_attn_kwargs(ch3))  # 低频 → path3
        self.attn_pair_1 = AttentionClass(**get_attn_kwargs(ch2))  # 中频 → path2
        self.attn_pair_2 = AttentionClass(**get_attn_kwargs(ch1))  # 高频 → path1
        
        # 拼接后的总通道数
        concat_channels = ch1 + ch2 + ch3
        
        # 1×1 Conv 融合
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        spatial_paths: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        freq_tokens: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        """
        path1, path2, path3 = spatial_paths
        token0, token1, token2 = freq_tokens
        
        # 交叉注意力
        enhanced_path3 = self.attn_pair_0(path3, token0)  # 低频 → 大感受野
        enhanced_path2 = self.attn_pair_1(path2, token1)  # 中频 → 中感受野
        enhanced_path1 = self.attn_pair_2(path1, token2)  # 高频 → 小感受野
        
        # 拼接 + 融合
        concat = torch.cat([enhanced_path1, enhanced_path2, enhanced_path3], dim=1)
        fused = self.fusion_conv(concat)
        
        return fused


class FreqGuidedFusionHybrid(nn.Module):
    """
    混合注意力版本：不同层使用不同策略
    
    - Level 0 (最大分辨率): 稀疏注意力 (N 最大，必须稀疏)
    - Level 1 (中等分辨率): 稀疏注意力
    - Level 2 (最小分辨率): 标准注意力 (N 较小，可以用标准)
    """
    
    def __init__(
        self,
        spatial_channels: list,
        freq_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_samples: int = 4096,
        level: int = 0  # 当前 encoder level
    ):
        super().__init__()
        
        self.level = level
        ch1, ch2, ch3 = spatial_channels
        
        # 根据 level 决定是否使用稀疏
        # Level 0: 64×128×128 = 1M tokens → 必须稀疏
        # Level 1: 32×64×64 = 128K tokens → 建议稀疏
        # Level 2: 16×32×32 = 16K tokens → 可以标准
        use_sparse = (level <= 1)
        
        AttentionClass = SparseFreqGuidedAttentionPair if use_sparse else FreqGuidedAttentionPair
        
        def get_attn_kwargs(ch):
            kwargs = {
                'spatial_channels': ch,
                'freq_channels': freq_channels,
                'num_heads': num_heads,
                'dropout': dropout
            }
            if use_sparse:
                kwargs['num_samples'] = num_samples
            return kwargs
        
        self.attn_pair_0 = AttentionClass(**get_attn_kwargs(ch3))
        self.attn_pair_1 = AttentionClass(**get_attn_kwargs(ch2))
        self.attn_pair_2 = AttentionClass(**get_attn_kwargs(ch1))
        
        concat_channels = ch1 + ch2 + ch3
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        spatial_paths: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        freq_tokens: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        path1, path2, path3 = spatial_paths
        token0, token1, token2 = freq_tokens
        
        enhanced_path3 = self.attn_pair_0(path3, token0)
        enhanced_path2 = self.attn_pair_1(path2, token1)
        enhanced_path1 = self.attn_pair_2(path1, token2)
        
        concat = torch.cat([enhanced_path1, enhanced_path2, enhanced_path3], dim=1)
        fused = self.fusion_conv(concat)
        
        return fused
