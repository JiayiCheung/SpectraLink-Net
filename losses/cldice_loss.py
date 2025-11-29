# losses/cldice_loss.py
"""
clDice损失函数
基于中心线的拓扑保持损失，用于管状结构分割

参考文献：
    Shit et al., "clDice - a Novel Topology-Preserving Loss Function
    for Tubular Structure Segmentation", CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def soft_erode(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
	"""
	软腐蚀操作

	使用最小池化实现形态学腐蚀

	参数:
		img: 输入图像 [B, C, D, H, W]
		kernel_size: 腐蚀核大小

	返回:
		腐蚀后的图像
	"""
	padding = kernel_size // 2
	
	# 3D最小池化 = 取反 -> 最大池化 -> 取反
	# 但对于软分割，直接用-max(-x)
	return -F.max_pool3d(
		-img,
		kernel_size=kernel_size,
		stride=1,
		padding=padding
	)


def soft_dilate(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
	"""
	软膨胀操作

	使用最大池化实现形态学膨胀

	参数:
		img: 输入图像 [B, C, D, H, W]
		kernel_size: 膨胀核大小

	返回:
		膨胀后的图像
	"""
	padding = kernel_size // 2
	
	return F.max_pool3d(
		img,
		kernel_size=kernel_size,
		stride=1,
		padding=padding
	)


def soft_open(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
	"""
	软开运算

	先腐蚀后膨胀，去除小的突出物
	"""
	return soft_dilate(soft_erode(img, kernel_size), kernel_size)


def soft_skeleton(
		img: torch.Tensor,
		num_iterations: int = 10,
		kernel_size: int = 3
) -> torch.Tensor:
	"""
	软骨架化

	通过迭代腐蚀和开运算提取中心线

	算法：
		skeleton = 0
		for i in range(num_iterations):
			eroded = erode^i(img)
			opened = open(eroded)
			skeleton += eroded - opened

	参数:
		img: 输入图像 [B, C, D, H, W]，值在[0,1]之间
		num_iterations: 迭代次数，控制骨架化程度
		kernel_size: 形态学操作的核大小

	返回:
		骨架图像
	"""
	skeleton = torch.zeros_like(img)
	current = img.clone()
	
	for i in range(num_iterations):
		# 腐蚀
		eroded = soft_erode(current, kernel_size)
		
		# 开运算
		opened = soft_open(eroded, kernel_size)
		
		# 骨架 = 腐蚀 - 开运算
		skeleton = skeleton + F.relu(eroded - opened)
		
		# 更新当前图像
		current = eroded
		
		# 如果图像已经全部腐蚀掉，提前停止
		if current.max() < 1e-6:
			break
	
	return skeleton


def soft_clDice(
		pred: torch.Tensor,
		target: torch.Tensor,
		num_iterations: int = 10,
		smooth: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	计算软clDice

	clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)

	其中：
		Tprec = |S_pred ∩ V_target| / |S_pred|  (拓扑精确率)
		Tsens = |S_target ∩ V_pred| / |S_target|  (拓扑召回率)
		S = skeleton, V = volume

	参数:
		pred: 预测概率 [B, C, D, H, W]
		target: 真实标签 [B, C, D, H, W]
		num_iterations: 骨架化迭代次数
		smooth: 平滑项

	返回:
		(cldice, tprec, tsens)
	"""
	# 提取骨架
	skel_pred = soft_skeleton(pred, num_iterations)
	skel_target = soft_skeleton(target, num_iterations)
	
	# 拓扑精确率: 预测骨架在真实体积中的比例
	tprec_num = (skel_pred * target).sum(dim=(2, 3, 4))
	tprec_den = skel_pred.sum(dim=(2, 3, 4))
	tprec = (tprec_num + smooth) / (tprec_den + smooth)
	
	# 拓扑召回率: 真实骨架在预测体积中的比例
	tsens_num = (skel_target * pred).sum(dim=(2, 3, 4))
	tsens_den = skel_target.sum(dim=(2, 3, 4))
	tsens = (tsens_num + smooth) / (tsens_den + smooth)
	
	# clDice
	cldice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
	
	return cldice, tprec, tsens


class ClDiceLoss(nn.Module):
	"""
	clDice损失函数

	基于中心线的拓扑保持损失，特别适合血管等管状结构

	参数:
		num_iterations: 骨架化迭代次数（默认10）
		smooth: 平滑项（默认1e-5）
		sigmoid: 是否对输入做sigmoid（默认True）
		reduction: 归约方式（默认'mean'）

	输入:
		logits: [B, C, D, H, W] 预测logits
		targets: [B, C, D, H, W] 真实标签

	输出:
		loss: clDice损失 = 1 - clDice
	"""
	
	def __init__(
			self,
			num_iterations: int = 10,
			smooth: float = 1e-5,
			sigmoid: bool = True,
			reduction: str = 'mean'
	):
		super().__init__()
		self.num_iterations = num_iterations
		self.smooth = smooth
		self.sigmoid = sigmoid
		self.reduction = reduction
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算clDice损失"""
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		targets = targets.float()
		
		# 计算clDice
		cldice, _, _ = soft_clDice(
			probs,
			targets,
			self.num_iterations,
			self.smooth
		)
		
		# 损失 = 1 - clDice
		loss = 1.0 - cldice
		
		# 对通道维度求平均
		loss = loss.mean(dim=1)  # [B]
		
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'none':
			return loss
		else:
			raise ValueError(f"不支持的reduction: {self.reduction}")
	
	def forward_with_components(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		计算clDice损失，同时返回各分量

		返回:
			(loss, tprec, tsens)
		"""
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		targets = targets.float()
		
		cldice, tprec, tsens = soft_clDice(
			probs,
			targets,
			self.num_iterations,
			self.smooth
		)
		
		loss = 1.0 - cldice
		loss = loss.mean(dim=1)
		
		if self.reduction == 'mean':
			return loss.mean(), tprec.mean(), tsens.mean()
		else:
			return loss, tprec.mean(dim=1), tsens.mean(dim=1)


class DiceClDiceLoss(nn.Module):
	"""
	Dice + clDice组合损失

	结合区域重叠和拓扑连续性

	Loss = alpha * DiceLoss + (1 - alpha) * clDiceLoss

	参数:
		alpha: Dice损失权重（默认0.5）
		num_iterations: clDice骨架化迭代次数
		smooth: 平滑项
		sigmoid: 是否对输入做sigmoid
	"""
	
	def __init__(
			self,
			alpha: float = 0.5,
			num_iterations: int = 10,
			smooth: float = 1e-5,
			sigmoid: bool = True
	):
		super().__init__()
		self.alpha = alpha
		self.smooth = smooth
		self.sigmoid = sigmoid
		self.num_iterations = num_iterations
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算组合损失"""
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		targets = targets.float()
		
		# Dice
		probs_flat = probs.view(probs.size(0), -1)
		targets_flat = targets.view(targets.size(0), -1)
		
		intersection = (probs_flat * targets_flat).sum(dim=-1)
		union = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
		dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
		dice_loss = 1.0 - dice
		
		# clDice
		cldice, _, _ = soft_clDice(probs, targets, self.num_iterations, self.smooth)
		cldice_loss = 1.0 - cldice.mean(dim=1)
		
		# 组合
		loss = self.alpha * dice_loss + (1.0 - self.alpha) * cldice_loss
		
		return loss.mean()