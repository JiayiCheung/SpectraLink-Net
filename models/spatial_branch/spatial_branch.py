# models/spatial_branch/spatial_branch.py
"""
空域分支主模块
EnhancedSpatialBranch: 多尺度空间特征提取
"""

import torch
import torch.nn as nn
from typing import Tuple

from .multi_scale_conv import PathConv


class EnhancedSpatialBranch(nn.Module):
	"""
	增强型空域分支

	使用三个并行路径提取不同尺度的空间特征：
	- Path1 (3×3×3): 小感受野，捕获细小血管和边缘细节
	- Path2 (5×5×5): 中感受野，捕获中等血管结构
	- Path3 (7×7×7): 大感受野，捕获大血管和全局上下文

	通道分配比例: 1/2 : 1/4 : 1/4

	设计理念：
	- 小卷积核分配更多通道（细节丰富，需要更多特征）
	- 大卷积核分配较少通道（全局信息，冗余度高）

	参数:
		in_channels: 输入通道数
		out_channels: 总输出通道数（必须能被4整除）

	输入:
		x: [B, in_channels, D, H, W]

	输出:
		(path1, path2, path3): 三个路径的特征
		- path1: [B, out_channels//2, D, H, W]   (3×3×3, 小尺度)
		- path2: [B, out_channels//4, D, H, W]   (5×5×5, 中尺度)
		- path3: [B, out_channels//4, D, H, W]   (7×7×7, 大尺度)
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		
		assert out_channels % 4 == 0, f"out_channels({out_channels})必须能被4整除"
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# 通道分配
		self.ch1 = out_channels // 2  # Path1: 1/2 通道
		self.ch2 = out_channels // 4  # Path2: 1/4 通道
		self.ch3 = out_channels // 4  # Path3: 1/4 通道
		
		# 三个并行路径
		# Path1: 3×3×3 → 小血管/细节 (高频空间信息)
		self.path1 = PathConv(
			in_channels=in_channels,
			out_channels=self.ch1,
			kernel_size=3,
			use_depthwise_separable=True
		)
		
		# Path2: 5×5×5 → 中血管 (中频空间信息)
		self.path2 = PathConv(
			in_channels=in_channels,
			out_channels=self.ch2,
			kernel_size=5,
			use_depthwise_separable=True
		)
		
		# Path3: 7×7×7 → 大血管/全局 (低频空间信息)
		self.path3 = PathConv(
			in_channels=in_channels,
			out_channels=self.ch3,
			kernel_size=7,
			use_depthwise_separable=True
		)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		前向传播，返回三个路径的独立输出

		参数:
			x: 输入特征 [B, in_channels, D, H, W]

		返回:
			(path1, path2, path3):
			- path1: [B, ch1, D, H, W] 小尺度特征 (3×3×3)
			- path2: [B, ch2, D, H, W] 中尺度特征 (5×5×5)
			- path3: [B, ch3, D, H, W] 大尺度特征 (7×7×7)
		"""
		path1_out = self.path1(x)  # 小血管
		path2_out = self.path2(x)  # 中血管
		path3_out = self.path3(x)  # 大血管
		
		return path1_out, path2_out, path3_out
	
	def forward_concat(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播，返回拼接后的特征

		参数:
			x: 输入特征 [B, in_channels, D, H, W]

		返回:
			拼接特征 [B, out_channels, D, H, W]
		"""
		path1_out, path2_out, path3_out = self.forward(x)
		return torch.cat([path1_out, path2_out, path3_out], dim=1)
	
	def get_channel_info(self) -> dict:
		"""
		获取通道分配信息

		返回:
			包含各路径通道数的字典
		"""
		return {
			'path1_channels': self.ch1,  # 3×3×3
			'path2_channels': self.ch2,  # 5×5×5
			'path3_channels': self.ch3,  # 7×7×7
			'total_channels': self.out_channels
		}