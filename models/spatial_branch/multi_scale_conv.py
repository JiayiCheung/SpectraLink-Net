# models/spatial_branch/multi_scale_conv.py
"""
多尺度卷积模块
包含单路径卷积PathConv，用于捕获不同尺度的血管结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathConv(nn.Module):
	"""
	单路径卷积块
	Conv3d(k×k×k) + InstanceNorm3d + ReLU

	使用深度可分离卷积减少计算量：
	- Depthwise: 每个通道独立的空间卷积
	- Pointwise: 1×1×1卷积混合通道

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数
		kernel_size: 卷积核大小 (3, 5, 或 7)
		use_depthwise_separable: 是否使用深度可分离卷积（默认True，减少计算量）
	"""
	
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int = 3,
			use_depthwise_separable: bool = True
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		padding = kernel_size // 2
		
		if use_depthwise_separable and in_channels >= 4:
			# 深度可分离卷积：减少参数量和计算量
			# Depthwise: 空间卷积，每个通道独立
			self.depthwise = nn.Conv3d(
				in_channels, in_channels,
				kernel_size=kernel_size,
				padding=padding,
				groups=in_channels,  # 每个通道独立卷积
				bias=False
			)
			# Pointwise: 1×1×1卷积混合通道
			self.pointwise = nn.Conv3d(
				in_channels, out_channels,
				kernel_size=1,
				bias=False
			)
			self.use_separable = True
		else:
			# 标准卷积
			self.conv = nn.Conv3d(
				in_channels, out_channels,
				kernel_size=kernel_size,
				padding=padding,
				bias=False
			)
			self.use_separable = False
		
		self.norm = nn.InstanceNorm3d(out_channels, affine=True)
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		参数:
			x: 输入特征 [B, C_in, D, H, W]

		返回:
			输出特征 [B, C_out, D, H, W]
		"""
		if self.use_separable:
			x = self.depthwise(x)
			x = self.pointwise(x)
		else:
			x = self.conv(x)
		
		x = self.norm(x)
		x = self.act(x)
		return x


class MultiScalePathConv(nn.Module):
	"""
	多尺度路径卷积组合
	包含3个并行路径，分别使用不同大小的卷积核

	- Path1: 3×3×3 卷积 → 捕获小血管/细节
	- Path2: 5×5×5 卷积 → 捕获中等血管
	- Path3: 7×7×7 卷积 → 捕获大血管/全局结构

	通道分配比例: 1/2 : 1/4 : 1/4

	参数:
		in_channels: 输入通道数
		out_channels: 总输出通道数（将按比例分配给各路径）
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		
		# 通道分配: 1/2, 1/4, 1/4
		# 确保通道数能被4整除
		assert out_channels % 4 == 0, f"out_channels({out_channels})必须能被4整除"
		
		self.ch1 = out_channels // 2  # 3×3×3 路径，1/2通道
		self.ch2 = out_channels // 4  # 5×5×5 路径，1/4通道
		self.ch3 = out_channels // 4  # 7×7×7 路径，1/4通道
		
		# 三个并行路径
		self.path1 = PathConv(in_channels, self.ch1, kernel_size=3)  # 小血管
		self.path2 = PathConv(in_channels, self.ch2, kernel_size=5)  # 中血管
		self.path3 = PathConv(in_channels, self.ch3, kernel_size=7)  # 大血管
	
	def forward(self, x: torch.Tensor) -> tuple:
		"""
		参数:
			x: 输入特征 [B, C_in, D, H, W]

		返回:
			(path1_out, path2_out, path3_out): 三个路径的输出
			- path1_out: [B, ch1, D, H, W]  (小尺度)
			- path2_out: [B, ch2, D, H, W]  (中尺度)
			- path3_out: [B, ch3, D, H, W]  (大尺度)
		"""
		out1 = self.path1(x)  # 3×3×3 → 小血管
		out2 = self.path2(x)  # 5×5×5 → 中血管
		out3 = self.path3(x)  # 7×7×7 → 大血管
		
		return out1, out2, out3
	
	def forward_concat(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播并拼接输出

		参数:
			x: 输入特征 [B, C_in, D, H, W]

		返回:
			拼接后的特征 [B, out_channels, D, H, W]
		"""
		out1, out2, out3 = self.forward(x)
		return torch.cat([out1, out2, out3], dim=1)