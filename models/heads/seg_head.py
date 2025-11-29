# models/heads/seg_head.py
"""
分割头模块
将解码器输出转换为分割预测
"""

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
	"""
	分割头

	两层卷积将特征映射到分割输出：
	Conv3d(in_ch → mid_ch, k=3) + IN + ReLU → Conv3d(mid_ch → out_ch, k=1)

	输出logits（不加sigmoid/softmax，由loss函数处理）

	参数:
		in_channels: 输入通道数（来自Decoder输出，默认32）
		mid_channels: 中间层通道数（默认16）
		out_channels: 输出通道数（类别数，二分类为1）

	输入:
		x: [B, in_channels, D, H, W]

	输出:
		logits: [B, out_channels, D, H, W]
	"""
	
	def __init__(
			self,
			in_channels: int = 32,
			mid_channels: int = 16,
			out_channels: int = 1
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.mid_channels = mid_channels
		self.out_channels = out_channels
		
		# 第一层：3×3×3卷积压缩通道
		self.conv1 = nn.Conv3d(
			in_channels, mid_channels,
			kernel_size=3,
			padding=1,
			bias=False
		)
		self.norm = nn.InstanceNorm3d(mid_channels, affine=True)
		self.act = nn.ReLU(inplace=True)
		
		# 第二层：1×1×1卷积输出logits
		self.conv2 = nn.Conv3d(
			mid_channels, out_channels,
			kernel_size=1,
			bias=True  # 最后一层加bias
		)
		
		# 初始化
		self._init_weights()
	
	def _init_weights(self):
		"""权重初始化"""
		# conv1: kaiming初始化
		nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		
		# conv2: xavier初始化，bias初始化为小负值（因为前景稀疏）
		nn.init.xavier_uniform_(self.conv2.weight)
		if self.conv2.bias is not None:
			# 初始化bias为小负值，使初始预测偏向背景
			# 这对于前景稀疏的血管分割有帮助
			nn.init.constant_(self.conv2.bias, -2.0)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播

		参数:
			x: 输入特征 [B, in_channels, D, H, W]

		返回:
			logits: [B, out_channels, D, H, W]
		"""
		x = self.conv1(x)
		x = self.norm(x)
		x = self.act(x)
		logits = self.conv2(x)
		
		return logits


class MultiScaleSegmentationHead(nn.Module):
	"""
	多尺度分割头（用于深度监督，可选）

	为Decoder的每层输出都生成预测，用于深度监督训练

	参数:
		decoder_channels: Decoder各层输出通道数列表
		out_channels: 输出类别数
	"""
	
	def __init__(
			self,
			decoder_channels: list = None,
			out_channels: int = 1
	):
		super().__init__()
		
		if decoder_channels is None:
			decoder_channels = [128, 64, 32]  # 默认3层
		
		self.heads = nn.ModuleList()
		for ch in decoder_channels:
			head = nn.Conv3d(ch, out_channels, kernel_size=1, bias=True)
			# 初始化bias为负值
			nn.init.constant_(head.bias, -2.0)
			self.heads.append(head)
	
	def forward(self, features: list) -> list:
		"""
		前向传播

		参数:
			features: Decoder各层输出列表

		返回:
			logits_list: 各层的logits列表
		"""
		logits_list = []
		for feat, head in zip(features, self.heads):
			logits_list.append(head(feat))
		return logits_list