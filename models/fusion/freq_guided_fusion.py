# models/fusion/freq_guided_fusion.py
"""
频域引导融合模块
整合3对交叉注意力，实现频域-空域特征融合
"""

import torch
import torch.nn as nn
from typing import Tuple

from .cross_attention import FreqGuidedAttentionPair


class FreqGuidedFusion(nn.Module):
	"""
	频域引导融合模块

	将3个频域token与3个空域path进行交叉注意力融合：
	- token0 (低频, 大血管) → guide → path3 (7×7×7, 大感受野)
	- token1 (中频, 中血管) → guide → path2 (5×5×5, 中感受野)
	- token2 (高频, 小血管) → guide → path1 (3×3×3, 小感受野)

	核心设计理念：
	- 低频信息对应大尺度结构，用于引导大感受野特征
	- 高频信息对应细节结构，用于引导小感受野特征
	- 这种显式对应关系增强了频域-空域的协同作用

	融合流程：
	1. 3对独立的 CrossAttention
	2. 拼接3个增强后的path特征
	3. 1×1 Conv 融合通道

	参数:
		spatial_channels: 空域各path的通道数列表 [ch1, ch2, ch3]
						 ch1: path1 (3×3×3) 通道数
						 ch2: path2 (5×5×5) 通道数
						 ch3: path3 (7×7×7) 通道数
		freq_channels: 频域token的通道数（3个token通道数相同）
		out_channels: 输出通道数
		dropout: dropout比例

	输入:
		spatial_paths: (path1, path2, path3) 空域特征元组
			- path1: [B, ch1, D, H, W] (3×3×3, 小血管)
			- path2: [B, ch2, D, H, W] (5×5×5, 中血管)
			- path3: [B, ch3, D, H, W] (7×7×7, 大血管)
		freq_tokens: (token0, token1, token2) 频域token元组
			- token0: [B, freq_ch] (低频, 大血管)
			- token1: [B, freq_ch] (中频, 中血管)
			- token2: [B, freq_ch] (高频, 小血管)

	输出:
		fused: [B, out_channels, D, H, W] 融合后的特征
	"""
	
	def __init__(
			self,
			spatial_channels: list,
			freq_channels: int,
			out_channels: int,
			dropout: float = 0.0
	):
		super().__init__()
		
		assert len(spatial_channels) == 3, "需要3个空域path的通道数"
		
		self.spatial_channels = spatial_channels  # [ch1, ch2, ch3]
		self.freq_channels = freq_channels
		self.out_channels = out_channels
		
		ch1, ch2, ch3 = spatial_channels
		
		# 3对交叉注意力
		# Pair 0: token0 (低频) → path3 (7×7×7)
		self.attn_pair_0 = FreqGuidedAttentionPair(
			spatial_channels=ch3,  # path3的通道数
			freq_channels=freq_channels,
			dropout=dropout
		)
		
		# Pair 1: token1 (中频) → path2 (5×5×5)
		self.attn_pair_1 = FreqGuidedAttentionPair(
			spatial_channels=ch2,  # path2的通道数
			freq_channels=freq_channels,
			dropout=dropout
		)
		
		# Pair 2: token2 (高频) → path1 (3×3×3)
		self.attn_pair_2 = FreqGuidedAttentionPair(
			spatial_channels=ch1,  # path1的通道数
			freq_channels=freq_channels,
			dropout=dropout
		)
		
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

		参数:
			spatial_paths: (path1, path2, path3)
				- path1: [B, ch1, D, H, W] (3×3×3, 小尺度)
				- path2: [B, ch2, D, H, W] (5×5×5, 中尺度)
				- path3: [B, ch3, D, H, W] (7×7×7, 大尺度)
			freq_tokens: (token0, token1, token2)
				- token0: [B, freq_ch] (低频)
				- token1: [B, freq_ch] (中频)
				- token2: [B, freq_ch] (高频)

		返回:
			fused: [B, out_channels, D, H, W]
		"""
		path1, path2, path3 = spatial_paths
		token0, token1, token2 = freq_tokens
		
		# 交叉注意力（注意对应关系）
		# token0 (低频, 大血管) → guide → path3 (7×7×7, 大感受野)
		enhanced_path3 = self.attn_pair_0(path3, token0)
		
		# token1 (中频, 中血管) → guide → path2 (5×5×5, 中感受野)
		enhanced_path2 = self.attn_pair_1(path2, token1)
		
		# token2 (高频, 小血管) → guide → path1 (3×3×3, 小感受野)
		enhanced_path1 = self.attn_pair_2(path1, token2)
		
		# 拼接 [B, ch1+ch2+ch3, D, H, W]
		concat = torch.cat([enhanced_path1, enhanced_path2, enhanced_path3], dim=1)
		
		# 1×1 Conv 融合
		fused = self.fusion_conv(concat)
		
		return fused
	
	def forward_with_details(
			self,
			spatial_paths: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
			freq_tokens: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
	) -> Tuple[torch.Tensor, dict]:
		"""
		前向传播，返回详细中间结果（用于可视化/分析）

		返回:
			(fused, details):
			- fused: 融合结果
			- details: 包含增强后各path的字典
		"""
		path1, path2, path3 = spatial_paths
		token0, token1, token2 = freq_tokens
		
		enhanced_path3 = self.attn_pair_0(path3, token0)
		enhanced_path2 = self.attn_pair_1(path2, token1)
		enhanced_path1 = self.attn_pair_2(path1, token2)
		
		concat = torch.cat([enhanced_path1, enhanced_path2, enhanced_path3], dim=1)
		fused = self.fusion_conv(concat)
		
		details = {
			'enhanced_path1': enhanced_path1,  # 高频引导
			'enhanced_path2': enhanced_path2,  # 中频引导
			'enhanced_path3': enhanced_path3,  # 低频引导
			'concat': concat
		}
		
		return fused, details