# models/decoder/decoder.py
"""
解码器模块
U-Net风格的解码器，逐层上采样并融合skip connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..components import DoubleConv, UpsampleBlock


class DecoderBlock(nn.Module):
	"""
	解码器块

	单层解码器：上采样 → 拼接Skip → DoubleConv

	结构：
	Input → Upsample(×2) → Concat(skip) → DoubleConv → Output

	参数:
		in_channels: 输入通道数（来自上一层decoder或bottleneck）
		skip_channels: skip connection通道数
		out_channels: 输出通道数

	输入:
		x: [B, in_channels, D, H, W] 来自上一层
		skip: [B, skip_channels, D*2, H*2, W*2] skip connection

	输出:
		out: [B, out_channels, D*2, H*2, W*2]
	"""
	
	def __init__(
			self,
			in_channels: int,
			skip_channels: int,
			out_channels: int
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.skip_channels = skip_channels
		self.out_channels = out_channels
		
		# 上采样：插值 + 3×3卷积调整通道
		self.upsample = UpsampleBlock(
			in_channels=in_channels,
			out_channels=in_channels // 2  # 上采样时通道减半
		)
		
		# 拼接后的通道数
		concat_channels = in_channels // 2 + skip_channels
		
		# 双层卷积
		self.conv = DoubleConv(
			in_channels=concat_channels,
			out_channels=out_channels
		)
	
	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		"""
		前向传播

		参数:
			x: 输入特征 [B, in_ch, D, H, W]
			skip: skip connection [B, skip_ch, D', H', W']

		返回:
			out: [B, out_ch, D', H', W']
		"""
		# 1. 上采样，匹配skip的空间尺寸
		x = self.upsample(x, target_size=skip.shape[2:])
		
		# 2. 拼接skip connection
		x = torch.cat([x, skip], dim=1)
		
		# 3. 双层卷积
		x = self.conv(x)
		
		return x


class Decoder(nn.Module):
	"""
	完整解码器

	3层DecoderBlock，逐层上采样并融合skip connection

	结构（假设输入patch [64,128,128]，base_channels=32）：

	Bottleneck: [B, 256, 8, 16, 16]
		↓ DecoderBlock-2: Up + Cat(skip2) + Conv
	Level 2: [B, 128, 16, 32, 32]
		↓ DecoderBlock-1: Up + Cat(skip1) + Conv
	Level 1: [B, 64, 32, 64, 64]
		↓ DecoderBlock-0: Up + Cat(skip0) + Conv
	Level 0: [B, 32, 64, 128, 128]

	参数:
		bottleneck_channels: bottleneck输出通道数（默认256）
		skip_channels: 各层skip connection通道数列表 [skip0_ch, skip1_ch, skip2_ch]
					  默认 [32, 64, 128]
		out_channels: 最终输出通道数（默认32）
		num_levels: 解码器层数（默认3，与编码器对应）
	"""
	
	def __init__(
			self,
			bottleneck_channels: int = 256,
			skip_channels: List[int] = None,
			out_channels: int = 32,
			num_levels: int = 3
	):
		super().__init__()
		
		self.num_levels = num_levels
		
		# 默认skip通道数
		if skip_channels is None:
			skip_channels = [32 * (2 ** i) for i in range(num_levels)]  # [32, 64, 128]
		
		assert len(skip_channels) == num_levels, \
			f"skip_channels长度({len(skip_channels)})必须等于num_levels({num_levels})"
		
		self.skip_channels = skip_channels
		
		# 构建DecoderBlock（从深到浅）
		self.decoder_blocks = nn.ModuleList()
		
		current_ch = bottleneck_channels
		
		for level in reversed(range(num_levels)):
			skip_ch = skip_channels[level]
			
			# 输出通道数
			if level == 0:
				out_ch = out_channels  # 最后一层输出指定通道数
			else:
				out_ch = skip_channels[level - 1]  # 匹配下一层skip的通道数
			
			block = DecoderBlock(
				in_channels=current_ch,
				skip_channels=skip_ch,
				out_channels=out_ch
			)
			self.decoder_blocks.append(block)
			
			# 更新下一个block的输入通道数
			current_ch = out_ch
		
		self.out_channels = out_channels
	
	def forward(
			self,
			bottleneck: torch.Tensor,
			skips: List[torch.Tensor]
	) -> torch.Tensor:
		"""
		前向传播

		参数:
			bottleneck: bottleneck特征 [B, bottleneck_ch, D, H, W]
			skips: skip connection列表 [skip0, skip1, skip2]
				   注意：顺序是从浅到深，skip0对应最高分辨率

		返回:
			out: 解码器输出 [B, out_channels, D_orig, H_orig, W_orig]
		"""
		# skips顺序: [skip0, skip1, skip2] (浅→深)
		# 解码顺序: skip2 → skip1 → skip0 (深→浅)
		
		x = bottleneck
		
		for i, block in enumerate(self.decoder_blocks):
			# 从深到浅取skip
			skip_idx = self.num_levels - 1 - i
			skip = skips[skip_idx]
			x = block(x, skip)
		
		return x
	
	def forward_with_intermediates(
			self,
			bottleneck: torch.Tensor,
			skips: List[torch.Tensor]
	) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		"""
		前向传播，返回中间特征（用于深度监督等）

		返回:
			(out, intermediates):
			- out: 最终输出
			- intermediates: 各层输出列表
		"""
		x = bottleneck
		intermediates = []
		
		for i, block in enumerate(self.decoder_blocks):
			skip_idx = self.num_levels - 1 - i
			skip = skips[skip_idx]
			x = block(x, skip)
			intermediates.append(x)
		
		return x, intermediates