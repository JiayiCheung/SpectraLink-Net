# models/components/complex_ops.py
"""
复数运算基础模块
用于频域分支中的复数张量操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear3D(nn.Module):
	"""
	复数版 1×1×1 卷积/线性层，用于频域张量的通道混合

	实现复数矩阵乘法: Y = X @ W
	其中 X = Xr + i*Xi, W = Wr + i*Wi
	Y = (Xr*Wr - Xi*Wi) + i*(Xr*Wi + Xi*Wr)

	参数:
		in_channels: 输入通道数
		out_channels: 输出通道数

	输入:
		x: 复数张量 [B, C_in, D, H, W] (torch.complex64)

	输出:
		复数张量 [B, C_out, D, H, W] (torch.complex64)
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# 实部和虚部权重，分开存储
		self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
		self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: complex tensor [B, C_in, D, H, W]
		"""
		xr, xi = x.real, x.imag  # [B, C_in, D, H, W]
		
		# 复数矩阵乘法
		# Y_real = Xr @ Wr - Xi @ Wi
		# Y_imag = Xr @ Wi + Xi @ Wr
		yr = torch.einsum("bc..., oc -> bo...", xr, self.weight_real) - \
		     torch.einsum("bc..., oc -> bo...", xi, self.weight_imag)
		yi = torch.einsum("bc..., oc -> bo...", xr, self.weight_imag) + \
		     torch.einsum("bc..., oc -> bo...", xi, self.weight_real)
		
		return torch.complex(yr, yi)


class ComplexGELU(nn.Module):
	"""
	复数GELU激活函数
	分别对实部和虚部应用GELU
	"""
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if torch.is_complex(x):
			return torch.complex(F.gelu(x.real), F.gelu(x.imag))
		return F.gelu(x)


class ComplexReLU(nn.Module):
	"""
	复数ReLU激活函数
	分别对实部和虚部应用ReLU
	"""
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if torch.is_complex(x):
			return torch.complex(F.relu(x.real), F.relu(x.imag))
		return F.relu(x)


def complex_multiply(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	"""
	复数逐元素乘法

	参数:
		x: 复数张量
		w: 实数或复数权重张量（可广播）

	返回:
		复数张量
	"""
	if torch.is_complex(w):
		# (a + bi)(c + di) = (ac - bd) + (ad + bc)i
		return torch.complex(
			x.real * w.real - x.imag * w.imag,
			x.real * w.imag + x.imag * w.real
		)
	else:
		# 实数权重直接缩放
		return torch.complex(x.real * w, x.imag * w)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
	"""
	计算复数张量的幅值

	参数:
		x: 复数张量

	返回:
		实数张量，幅值 sqrt(real^2 + imag^2)
	"""
	if torch.is_complex(x):
		return x.abs()
	return x.abs()


def complex_phase(x: torch.Tensor) -> torch.Tensor:
	"""
	计算复数张量的相位角

	参数:
		x: 复数张量

	返回:
		实数张量，相位角 atan2(imag, real)
	"""
	if torch.is_complex(x):
		return torch.angle(x)
	return torch.zeros_like(x)