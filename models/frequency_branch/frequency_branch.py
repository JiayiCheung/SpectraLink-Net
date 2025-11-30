# models/frequency_branch/frequency_branch.py
"""
频域分支主模块
ImprovedFrequencyBranch: 基于3D FFT的多频带特征提取

修改说明 (v2.0):
- 原版：全局池化生成 [B, token_channels] 的token，导致K/V只有1个位置
- 新版：自适应池化生成 [B, p, token_channels] 的token序列，保留空间选择性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .band_partition import LearnableBandPartition
from .complex_block import ComplexBlock


class ImprovedFrequencyBranch(nn.Module):
	"""
	改进的频域分支 (v2.0)

	完整流程：
	1. 3D FFT 变换到频域
	2. 可学习频带划分（3个频带）
	3. 各频带独立 ComplexBlock 增强
	4. 3D IFFT 逆变换回空域
	5. **自适应池化保留空间结构**（替代全局池化）
	6. 投影生成token序列

	输出3个token序列，与空域分支的3个path形成对应：
	- token0 (低频, p_low tokens) → guide path3 (7×7×7, 大血管)
	- token1 (中频, p_mid tokens) → guide path2 (5×5×5, 中血管)
	- token2 (高频, p_high tokens) → guide path1 (3×3×3, 小血管)

	参数:
		in_channels: 输入通道数
		token_channels: 每个token的通道数
		expansion_ratio: ComplexBlock的通道扩展倍数
		init_cuts: 频带初始切割点 (r1, r2)
		transition_width: 频带过渡宽度
		pool_sizes: 各频带的池化输出尺寸，默认 [(2,2,2), (4,4,4), (4,4,4)]
				   对应 p_low=8, p_mid=64, p_high=64

	输入:
		x: [B, in_channels, D, H, W]

	输出:
		(token0, token1, token2): 三个频带token序列
		- token0: [B, p_low, token_channels] 低频tokens
		- token1: [B, p_mid, token_channels] 中频tokens
		- token2: [B, p_high, token_channels] 高频tokens
	"""
	
	def __init__(
			self,
			in_channels: int,
			token_channels: int,
			expansion_ratio: int = 2,
			init_cuts: Tuple[float, float] = (0.2, 0.5),
			transition_width: float = 0.05,
			pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.token_channels = token_channels
		
		# 默认池化尺寸：低频少，高频多（但不要太大以控制显存）
		if pool_sizes is None:
			pool_sizes = ((2, 2, 2), (4, 4, 4), (4, 4, 4))
		
		self.pool_sizes = pool_sizes
		
		# 计算各频带的token数量
		self.p_low = pool_sizes[0][0] * pool_sizes[0][1] * pool_sizes[0][2]  # 8
		self.p_mid = pool_sizes[1][0] * pool_sizes[1][1] * pool_sizes[1][2]  # 64
		self.p_high = pool_sizes[2][0] * pool_sizes[2][1] * pool_sizes[2][2]  # 64
		
		# 可学习频带划分
		self.band_partition = LearnableBandPartition(
			init_cuts=init_cuts,
			transition_width=transition_width
		)
		
		# 三个频带的独立增强模块
		self.enhance_low = ComplexBlock(
			channels=in_channels,
			expansion_ratio=expansion_ratio
		)
		self.enhance_mid = ComplexBlock(
			channels=in_channels,
			expansion_ratio=expansion_ratio
		)
		self.enhance_high = ComplexBlock(
			channels=in_channels,
			expansion_ratio=expansion_ratio
		)
		
		# 自适应池化（替代全局池化）
		# 使用可学习的卷积下采样而非固定池化，保留更多信息
		self.spatial_compress_low = self._make_spatial_compressor(
			in_channels, pool_sizes[0]
		)
		self.spatial_compress_mid = self._make_spatial_compressor(
			in_channels, pool_sizes[1]
		)
		self.spatial_compress_high = self._make_spatial_compressor(
			in_channels, pool_sizes[2]
		)
		
		# 通道投影：[B, p, in_channels] → [B, p, token_channels]
		self.proj_low = nn.Linear(in_channels, token_channels)
		self.proj_mid = nn.Linear(in_channels, token_channels)
		self.proj_high = nn.Linear(in_channels, token_channels)
	
	def _make_spatial_compressor(
			self,
			channels: int,
			output_size: Tuple[int, int, int]
	) -> nn.Module:
		"""
		创建空间压缩模块

		使用自适应平均池化 + 可学习的1x1卷积refinement
		这比纯池化保留更多信息，比大stride卷积更灵活
		"""
		return nn.Sequential(
			nn.AdaptiveAvgPool3d(output_size),
			nn.Conv3d(channels, channels, kernel_size=1, bias=False),
			nn.LayerNorm([channels, *output_size]),
			nn.GELU()
		)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		前向传播

		参数:
			x: 输入特征 [B, C, D, H, W]

		返回:
			(token0, token1, token2):
			- token0: [B, p_low, token_channels] 低频tokens → guide path3
			- token1: [B, p_mid, token_channels] 中频tokens → guide path2
			- token2: [B, p_high, token_channels] 高频tokens → guide path1
		"""
		B, C, D, H, W = x.shape
		
		# 1. 3D FFT 变换到频域
		spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1))
		
		# 2. 可学习频带划分
		spec_low, spec_mid, spec_high = self.band_partition(spectrum)
		
		# 3. 各频带独立增强（在复数域）
		spec_low = self.enhance_low(spec_low)
		spec_mid = self.enhance_mid(spec_mid)
		spec_high = self.enhance_high(spec_high)
		
		# 4. 3D IFFT 逆变换回空域
		spatial_low = torch.fft.irfftn(spec_low, s=(D, H, W), dim=(-3, -2, -1))
		spatial_mid = torch.fft.irfftn(spec_mid, s=(D, H, W), dim=(-3, -2, -1))
		spatial_high = torch.fft.irfftn(spec_high, s=(D, H, W), dim=(-3, -2, -1))
		
		# 5. 空间压缩（保留空间结构，替代全局池化）
		# [B, C, D, H, W] → [B, C, p_d, p_h, p_w]
		compressed_low = self.spatial_compress_low(spatial_low)
		compressed_mid = self.spatial_compress_mid(spatial_mid)
		compressed_high = self.spatial_compress_high(spatial_high)
		
		# 6. 展平空间维度为token序列
		# [B, C, p_d, p_h, p_w] → [B, C, p] → [B, p, C]
		tokens_low = compressed_low.flatten(2).transpose(1, 2)  # [B, p_low, C]
		tokens_mid = compressed_mid.flatten(2).transpose(1, 2)  # [B, p_mid, C]
		tokens_high = compressed_high.flatten(2).transpose(1, 2)  # [B, p_high, C]
		
		# 7. 通道投影 → [B, p, token_channels]
		token0 = self.proj_low(tokens_low)  # 低频 → guide path3
		token1 = self.proj_mid(tokens_mid)  # 中频 → guide path2
		token2 = self.proj_high(tokens_high)  # 高频 → guide path1
		
		return token0, token1, token2
	
	def forward_with_spatial(self, x: torch.Tensor) -> Tuple[
		Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
		Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
	]:
		"""
		前向传播，同时返回空域特征（用于可视化/分析）

		参数:
			x: 输入特征 [B, C, D, H, W]

		返回:
			(tokens, spatial_features):
			- tokens: (token0, token1, token2) 各为 [B, p, token_channels]
			- spatial_features: (spatial_low, spatial_mid, spatial_high) 各为 [B, C, D, H, W]
		"""
		B, C, D, H, W = x.shape
		
		# FFT
		spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1))
		
		# 频带划分
		spec_low, spec_mid, spec_high = self.band_partition(spectrum)
		
		# 频带增强
		spec_low = self.enhance_low(spec_low)
		spec_mid = self.enhance_mid(spec_mid)
		spec_high = self.enhance_high(spec_high)
		
		# IFFT
		spatial_low = torch.fft.irfftn(spec_low, s=(D, H, W), dim=(-3, -2, -1))
		spatial_mid = torch.fft.irfftn(spec_mid, s=(D, H, W), dim=(-3, -2, -1))
		spatial_high = torch.fft.irfftn(spec_high, s=(D, H, W), dim=(-3, -2, -1))
		
		# 空间压缩
		compressed_low = self.spatial_compress_low(spatial_low)
		compressed_mid = self.spatial_compress_mid(spatial_mid)
		compressed_high = self.spatial_compress_high(spatial_high)
		
		# 展平 + 投影
		tokens_low = compressed_low.flatten(2).transpose(1, 2)
		tokens_mid = compressed_mid.flatten(2).transpose(1, 2)
		tokens_high = compressed_high.flatten(2).transpose(1, 2)
		
		token0 = self.proj_low(tokens_low)
		token1 = self.proj_mid(tokens_mid)
		token2 = self.proj_high(tokens_high)
		
		return (token0, token1, token2), (spatial_low, spatial_mid, spatial_high)
	
	def get_band_info(self) -> dict:
		"""
		获取当前频带划分信息

		返回:
			频带划分的详细信息，包含token数量
		"""
		info = self.band_partition.get_band_info()
		info['token_counts'] = {
			'low': self.p_low,
			'mid': self.p_mid,
			'high': self.p_high
		}
		info['pool_sizes'] = self.pool_sizes
		return info