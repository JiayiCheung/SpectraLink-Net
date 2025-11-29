# models/frequency_branch/complex_block.py
"""
复数域增强模块
用于在频域中对各频带进行特征增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.complex_ops import ComplexLinear3D, ComplexGELU


class ComplexBlock(nn.Module):
	"""
	复数域增强块

	对频域特征进行非线性增强，保持复数表示：
	1. ComplexLinear 扩展通道 (C → C*expansion_ratio)
	2. ComplexGELU 激活
	3. ComplexLinear 压缩通道 (C*expansion_ratio → C)
	4. 残差连接 + 可学习缩放因子 γ

	设计理念：
	- 在频域中直接处理复数，保留幅度和相位信息
	- 残差连接确保梯度稳定，γ初始化为小值保证训练初期稳定

	参数:
		channels: 输入/输出通道数
		expansion_ratio: 中间层通道扩展倍数，默认2
		init_gamma: γ的初始值，默认0.1

	输入:
		x: 复数频谱 [B, C, D, H, W_half]

	输出:
		增强后的复数频谱 [B, C, D, H, W_half]
	"""
	
	def __init__(
			self,
			channels: int,
			expansion_ratio: int = 2,
			init_gamma: float = 0.1
	):
		super().__init__()
		
		self.channels = channels
		self.expansion_ratio = expansion_ratio
		hidden_channels = channels * expansion_ratio
		
		# 扩展层：C → C * expansion_ratio
		self.expand = ComplexLinear3D(channels, hidden_channels)
		
		# 激活函数
		self.act = ComplexGELU()
		
		# 压缩层：C * expansion_ratio → C
		self.compress = ComplexLinear3D(hidden_channels, channels)
		
		# 可学习缩放因子，控制残差分支的贡献
		# 初始化为小值，让训练初期接近恒等映射
		self.gamma = nn.Parameter(torch.tensor(init_gamma))
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播

		参数:
			x: 复数频谱 [B, C, D, H, W_half]

		返回:
			增强后的复数频谱 [B, C, D, H, W_half]
		"""
		# 保存输入用于残差连接
		identity = x
		
		# 扩展 → 激活 → 压缩
		out = self.expand(x)
		out = self.act(out)
		out = self.compress(out)
		
		# 残差连接：output = identity + γ * enhanced
		out = identity + self.gamma * out
		
		return out


class MultiScaleComplexBlock(nn.Module):
	"""
	多频带复数增强模块

	为3个频带分别配置独立的ComplexBlock进行增强。
	每个频带可以有不同的通道数。

	参数:
		band_channels: 各频带的通道数列表 [ch_low, ch_mid, ch_high]
		expansion_ratio: 通道扩展倍数，默认2

	输入:
		(spec_low, spec_mid, spec_high): 三个频带的复数频谱

	输出:
		(enhanced_low, enhanced_mid, enhanced_high): 增强后的三个频带
	"""
	
	def __init__(
			self,
			band_channels: list,
			expansion_ratio: int = 2
	):
		super().__init__()
		
		assert len(band_channels) == 3, "需要3个频带的通道数"
		
		self.band_channels = band_channels
		
		# 三个频带的独立增强模块
		self.block_low = ComplexBlock(
			channels=band_channels[0],
			expansion_ratio=expansion_ratio
		)
		self.block_mid = ComplexBlock(
			channels=band_channels[1],
			expansion_ratio=expansion_ratio
		)
		self.block_high = ComplexBlock(
			channels=band_channels[2],
			expansion_ratio=expansion_ratio
		)
	
	def forward(
			self,
			spec_low: torch.Tensor,
			spec_mid: torch.Tensor,
			spec_high: torch.Tensor
	) -> tuple:
		"""
		前向传播

		参数:
			spec_low: 低频频谱 [B, C0, D, H, W_half]
			spec_mid: 中频频谱 [B, C1, D, H, W_half]
			spec_high: 高频频谱 [B, C2, D, H, W_half]

		返回:
			(enhanced_low, enhanced_mid, enhanced_high): 增强后的频谱
		"""
		enhanced_low = self.block_low(spec_low)
		enhanced_mid = self.block_mid(spec_mid)
		enhanced_high = self.block_high(spec_high)
		
		return enhanced_low, enhanced_mid, enhanced_high