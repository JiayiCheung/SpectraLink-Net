# models/components/conv_blocks.py
"""
通用卷积块模块
包含网络中复用的基础卷积组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitConv(nn.Module):
	"""
	初始卷积块
	Conv3d(in_ch → out_ch) + InstanceNorm3d + ReLU

	参数:
		in_channels: 输入通道数，默认1（CT图像）
		out_channels: 输出通道数，默认32
		kernel_size: 卷积核大小，默认3
	"""
	
	def __init__(self, in_channels: int = 1, out_channels: int = 32, kernel_size: int = 3):
		super().__init__()
		padding = kernel_size // 2
		
		self.conv = nn.Conv3d(
			in_channels, out_channels,
			kernel_size=kernel_size,
			padding=padding,
			bias=False
		)
		self.norm = nn.InstanceNorm3d(out_channels, affine=True)
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
	"""
	U-Net标准双层卷积块
	Conv3d + IN + ReLU → Conv3d + IN + ReLU

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数
		mid_channels: 中间通道数，默认与out_channels相同
	"""
	
	def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
		super().__init__()
		mid_channels = mid_channels or out_channels
		
		self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
		self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
		
		self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
		
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.act(self.norm1(self.conv1(x)))
		x = self.act(self.norm2(self.conv2(x)))
		return x


class DownsampleBlock(nn.Module):
	"""
	下采样块
	MaxPool3d(2) + Conv3d(1×1×1) 通道扩展

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
		self.norm = nn.InstanceNorm3d(out_channels, affine=True)
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(x)
		x = self.act(self.norm(self.conv(x)))
		return x


class UpsampleBlock(nn.Module):
	"""
	上采样块（Decoder用）
	Trilinear插值(2x) + Conv3d(3×3×3) 通道压缩

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.norm = nn.InstanceNorm3d(out_channels, affine=True)
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
		# 上采样
		if target_size is not None:
			x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
		else:
			x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
		
		# 卷积调整通道
		x = self.act(self.norm(self.conv(x)))
		return x


class ResidualConv(nn.Module):
	"""
	残差卷积块
	带跳跃连接的双层卷积

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		
		self.conv_block = DoubleConv(in_channels, out_channels)
		
		# 通道对齐的shortcut
		if in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.InstanceNorm3d(out_channels, affine=True)
			)
		else:
			self.shortcut = nn.Identity()
		
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = self.shortcut(x)
		out = self.conv_block(x)
		return self.act(out + identity)