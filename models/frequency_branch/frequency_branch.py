# models/frequency_branch/frequency_branch.py
"""
频域分支主模块
ImprovedFrequencyBranch: 基于3D FFT的多频带特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .band_partition import LearnableBandPartition
from .complex_block import ComplexBlock


class ImprovedFrequencyBranch(nn.Module):
	"""
	改进的频域分支

	完整流程：
	1. 3D FFT 变换到频域
	2. 可学习频带划分（3个频带）
	3. 各频带独立 ComplexBlock 增强
	4. 3D IFFT 逆变换回空域
	5. 全局平均池化生成 token

	输出3个token，与空域分支的3个path形成对应：
	- token0 (低频) → guide path3 (7×7×7, 大血管)
	- token1 (中频) → guide path2 (5×5×5, 中血管)
	- token2 (高频) → guide path1 (3×3×3, 小血管)

	参数:
		in_channels: 输入通道数
		token_channels: 每个token的通道数（3个token通道数相同）
		expansion_ratio: ComplexBlock的通道扩展倍数
		init_cuts: 频带初始切割点 (r1, r2)
		transition_width: 频带过渡宽度

	输入:
		x: [B, in_channels, D, H, W]

	输出:
		(token0, token1, token2): 三个频带token
		- token0: [B, token_channels] 低频token
		- token1: [B, token_channels] 中频token
		- token2: [B, token_channels] 高频token
	"""
	
	def __init__(
			self,
			in_channels: int,
			token_channels: int,
			expansion_ratio: int = 2,
			init_cuts: Tuple[float, float] = (0.2, 0.5),
			transition_width: float = 0.05
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.token_channels = token_channels
		
		# 可学习频带划分
		self.band_partition = LearnableBandPartition(
			init_cuts=init_cuts,
			transition_width=transition_width
		)
		
		# 三个频带的独立增强模块
		# 每个频带在频域中处理，通道数等于输入通道数
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
		
		# 通道投影：将空域特征投影到token_channels
		# IFFT后每个频带是 [B, in_channels, D, H, W]
		# 池化后是 [B, in_channels]
		# 需要投影到 [B, token_channels]
		self.proj_low = nn.Linear(in_channels, token_channels)
		self.proj_mid = nn.Linear(in_channels, token_channels)
		self.proj_high = nn.Linear(in_channels, token_channels)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		前向传播

		参数:
			x: 输入特征 [B, C, D, H, W]

		返回:
			(token0, token1, token2):
			- token0: [B, token_channels] 低频token → guide path3
			- token1: [B, token_channels] 中频token → guide path2
			- token2: [B, token_channels] 高频token → guide path1
		"""
		B, C, D, H, W = x.shape
		
		# 1. 3D FFT 变换到频域
		# rfftn: 实数输入的FFT，输出形状 [B, C, D, H, W//2+1]
		spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1))
		
		# 2. 可学习频带划分
		spec_low, spec_mid, spec_high = self.band_partition(spectrum)
		
		# 3. 各频带独立增强（在复数域）
		spec_low = self.enhance_low(spec_low)
		spec_mid = self.enhance_mid(spec_mid)
		spec_high = self.enhance_high(spec_high)
		
		# 4. 3D IFFT 逆变换回空域
		# irfftn: 逆FFT，需要指定原始空间尺寸
		spatial_low = torch.fft.irfftn(spec_low, s=(D, H, W), dim=(-3, -2, -1))
		spatial_mid = torch.fft.irfftn(spec_mid, s=(D, H, W), dim=(-3, -2, -1))
		spatial_high = torch.fft.irfftn(spec_high, s=(D, H, W), dim=(-3, -2, -1))
		
		# 5. 全局平均池化 → [B, C]
		token_low = spatial_low.mean(dim=(-3, -2, -1))  # [B, C]
		token_mid = spatial_mid.mean(dim=(-3, -2, -1))  # [B, C]
		token_high = spatial_high.mean(dim=(-3, -2, -1))  # [B, C]
		
		# 6. 通道投影 → [B, token_channels]
		token0 = self.proj_low(token_low)  # 低频 → guide path3
		token1 = self.proj_mid(token_mid)  # 中频 → guide path2
		token2 = self.proj_high(token_high)  # 高频 → guide path1
		
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
			- tokens: (token0, token1, token2)
			- spatial_features: (spatial_low, spatial_mid, spatial_high)
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
		
		# 池化 + 投影
		token_low = spatial_low.mean(dim=(-3, -2, -1))
		token_mid = spatial_mid.mean(dim=(-3, -2, -1))
		token_high = spatial_high.mean(dim=(-3, -2, -1))
		
		token0 = self.proj_low(token_low)
		token1 = self.proj_mid(token_mid)
		token2 = self.proj_high(token_high)
		
		return (token0, token1, token2), (spatial_low, spatial_mid, spatial_high)
	
	def get_band_info(self) -> dict:
		"""
		获取当前频带划分信息

		返回:
			频带划分的详细信息
		"""
		return self.band_partition.get_band_info()