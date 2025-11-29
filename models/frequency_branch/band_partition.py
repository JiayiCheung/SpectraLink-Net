# models/frequency_branch/band_partition.py
"""
可学习频带划分模块
用于将3D频谱自适应地划分为3个频带
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List


class LearnableBandPartition(nn.Module):
	"""
	可学习的频带划分

	将频谱按径向频率划分为3个频带：
	- 低频带 Ω_low:   [0, r1)    → 大血管/全局结构 → 对应 path3 (7×7×7)
	- 中频带 Ω_mid:   [r1, r2)   → 中等血管       → 对应 path2 (5×5×5)
	- 高频带 Ω_high:  [r2, 1.0]  → 小血管/细节    → 对应 path1 (3×3×3)

	切割点 r1, r2 是可学习参数，训练时自动优化。
	通过参数化方式保证有序性：0 < r1 < r2 < 1

	频域-空域对应关系（核心设计）：
	- token0 (低频) guide → path3 (大感受野)
	- token1 (中频) guide → path2 (中感受野)
	- token2 (高频) guide → path1 (小感受野)

	参数:
		init_cuts: 初始切割点 (r1, r2)，默认 (0.2, 0.5)
		transition_width: 频带过渡区宽度，默认 0.05
	"""
	
	def __init__(
			self,
			init_cuts: Tuple[float, float] = (0.2, 0.5),
			transition_width: float = 0.05
	):
		super().__init__()
		
		self.transition_width = transition_width
		
		# 使用未约束参数，通过sigmoid和累加保证有序性
		# raw参数通过inverse_sigmoid初始化
		self.raw_r1 = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_cuts[0])))
		self.raw_r2 = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_cuts[1])))
	
	def _inverse_sigmoid(self, x: float) -> float:
		"""sigmoid的逆函数，用于初始化"""
		x = max(min(x, 0.999), 0.001)  # 数值稳定
		return math.log(x / (1 - x))
	
	def get_cutpoints(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		获取有序的切割点

		通过参数化保证：0 < r1 < r2 < 1

		返回:
			(r1, r2): 两个切割点张量
		"""
		# r1: 限制在 [0.05, 0.40]
		r1 = 0.05 + 0.35 * torch.sigmoid(self.raw_r1)
		
		# r2: 在 r1 之后，限制间距，确保 r2 < 0.95
		# r2 = r1 + delta, delta ∈ [0.10, 0.95-r1]
		remaining = 0.95 - r1
		delta_12 = 0.10 + (remaining - 0.10).clamp(min=0) * torch.sigmoid(self.raw_r2)
		r2 = r1 + delta_12
		
		return r1, r2
	
	def create_frequency_grid(
			self,
			shape: Tuple[int, int, int],
			device: torch.device
	) -> torch.Tensor:
		"""
		创建归一化径向频率网格

		参数:
			shape: 频谱空间尺寸 (D, H, W_half)，W_half = W//2 + 1
			device: 计算设备

		返回:
			k_radius: 归一化频率半径 [D, H, W_half]，范围 [0, 1]
		"""
		D, H, W_half = shape
		W = 2 * (W_half - 1)  # 原始空间宽度
		
		# 创建频率坐标
		# fftfreq返回 [-0.5, 0.5) 范围的归一化频率
		kz = torch.fft.fftfreq(D, device=device)[:, None, None]  # [D, 1, 1]
		ky = torch.fft.fftfreq(H, device=device)[None, :, None]  # [1, H, 1]
		kx = torch.fft.rfftfreq(W, device=device)[None, None, :]  # [1, 1, W_half]
		
		# 计算径向频率（欧几里得距离）
		k_radius = torch.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
		
		# 归一化到 [0, 1]
		# 3D频谱的最大可能频率半径是 sqrt(0.5^2 * 3) ≈ 0.866
		k_max = math.sqrt(0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2)
		k_radius = k_radius / k_max
		
		return k_radius.clamp(0, 1)
	
	def _smooth_step(
			self,
			x: torch.Tensor,
			edge: torch.Tensor,
			width: float
	) -> torch.Tensor:
		"""
		平滑阶跃函数（基于sigmoid）

		参数:
			x: 输入值
			edge: 阶跃边界
			width: 过渡宽度

		返回:
			平滑过渡的 0→1 掩码
		"""
		# 缩放因子，控制过渡陡峭程度
		scale = 4.0 / (width + 1e-6)
		return torch.sigmoid(scale * (x - edge))
	
	def create_band_masks(
			self,
			k_radius: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		创建三个频带的平滑掩码

		参数:
			k_radius: 归一化频率半径 [D, H, W_half]

		返回:
			(mask_low, mask_mid, mask_high): 三个频带掩码
			每个掩码形状: [D, H, W_half]，值域 [0, 1]
		"""
		r1, r2 = self.get_cutpoints()
		tw = self.transition_width
		
		# 使用平滑阶跃函数创建掩码
		# 低频: 1 - step(r1) → 在 r1 之前为 1，之后平滑过渡到 0
		mask_low = 1.0 - self._smooth_step(k_radius, r1, tw)
		
		# 中频: step(r1) - step(r2) → 在 [r1, r2] 区间为 1
		mask_mid = self._smooth_step(k_radius, r1, tw) - self._smooth_step(k_radius, r2, tw)
		
		# 高频: step(r2) → 在 r2 之后为 1
		mask_high = self._smooth_step(k_radius, r2, tw)
		
		return mask_low, mask_mid, mask_high
	
	def forward(
			self,
			spectrum: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		将输入频谱划分为三个频带

		参数:
			spectrum: 复数频谱 [B, C, D, H, W_half]

		返回:
			(spec_low, spec_mid, spec_high): 三个频带的频谱
			- spec_low:  低频，对应大血管 → token0 → guide path3
			- spec_mid:  中频，对应中血管 → token1 → guide path2
			- spec_high: 高频，对应小血管 → token2 → guide path1
			每个形状与输入相同
		"""
		B, C, D, H, W_half = spectrum.shape
		device = spectrum.device
		
		# 创建频率网格
		k_radius = self.create_frequency_grid((D, H, W_half), device)
		
		# 创建频带掩码
		mask_low, mask_mid, mask_high = self.create_band_masks(k_radius)
		
		# 扩展掩码维度以匹配频谱 [1, 1, D, H, W_half]
		mask_low = mask_low.unsqueeze(0).unsqueeze(0)
		mask_mid = mask_mid.unsqueeze(0).unsqueeze(0)
		mask_high = mask_high.unsqueeze(0).unsqueeze(0)
		
		# 应用掩码分离频带
		spec_low = spectrum * mask_low
		spec_mid = spectrum * mask_mid
		spec_high = spectrum * mask_high
		
		return spec_low, spec_mid, spec_high
	
	def get_band_info(self) -> dict:
		"""
		获取当前频带划分信息（用于可视化/调试）

		返回:
			包含切割点和频带范围的字典
		"""
		with torch.no_grad():
			r1, r2 = self.get_cutpoints()
			return {
				'r1': r1.item(),
				'r2': r2.item(),
				'bands': {
					'low': (0.0, r1.item()),  # → token0 → path3 (7×7×7)
					'mid': (r1.item(), r2.item()),  # → token1 → path2 (5×5×5)
					'high': (r2.item(), 1.0)  # → token2 → path1 (3×3×3)
				},
				'transition_width': self.transition_width
			}