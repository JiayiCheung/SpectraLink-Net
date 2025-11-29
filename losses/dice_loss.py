# losses/dice_loss.py
"""
Dice损失函数
用于处理类别不平衡的分割任务
"""

import torch
import torch.nn as nn
from typing import Optional


class DiceLoss(nn.Module):
	"""
	Dice损失函数

	Dice系数衡量预测和真实标签的重叠程度，Dice Loss = 1 - Dice

	公式：
		Dice = (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
		Loss = 1 - Dice

	参数:
		smooth: 平滑项，避免除零（默认1e-5）
		sigmoid: 是否对输入做sigmoid（默认True，因为输入是logits）
		reduction: 归约方式，'mean'或'none'（默认'mean'）

	输入:
		logits: [B, C, D, H, W] 预测logits
		targets: [B, C, D, H, W] 真实标签（0或1）

	输出:
		loss: 标量或[B]
	"""
	
	def __init__(
			self,
			smooth: float = 1e-5,
			sigmoid: bool = True,
			reduction: str = 'mean'
	):
		super().__init__()
		self.smooth = smooth
		self.sigmoid = sigmoid
		self.reduction = reduction
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""
		计算Dice损失

		参数:
			logits: 预测logits [B, C, D, H, W]
			targets: 真实标签 [B, C, D, H, W]

		返回:
			loss: Dice损失
		"""
		# 转换为概率
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		# 确保targets是float
		targets = targets.float()
		
		# 展平空间维度，保留batch和channel
		# [B, C, D, H, W] -> [B, C, D*H*W]
		probs_flat = probs.view(probs.size(0), probs.size(1), -1)
		targets_flat = targets.view(targets.size(0), targets.size(1), -1)
		
		# 计算交集和并集
		intersection = (probs_flat * targets_flat).sum(dim=-1)
		union = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
		
		# 计算Dice系数
		dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
		
		# Dice Loss = 1 - Dice
		loss = 1.0 - dice
		
		# 对通道维度求平均
		loss = loss.mean(dim=1)  # [B]
		
		# 归约
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'none':
			return loss
		else:
			raise ValueError(f"不支持的reduction: {self.reduction}")


class SoftDiceLoss(nn.Module):
	"""
	Soft Dice损失

	与标准Dice的区别在于使用平方项，对边界更敏感

	公式：
		Dice = (2 * sum(P*G) + smooth) / (sum(P^2) + sum(G^2) + smooth)
	"""
	
	def __init__(
			self,
			smooth: float = 1e-5,
			sigmoid: bool = True,
			reduction: str = 'mean'
	):
		super().__init__()
		self.smooth = smooth
		self.sigmoid = sigmoid
		self.reduction = reduction
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算Soft Dice损失"""
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		targets = targets.float()
		
		# 展平
		probs_flat = probs.view(probs.size(0), probs.size(1), -1)
		targets_flat = targets.view(targets.size(0), targets.size(1), -1)
		
		# 使用平方项
		intersection = (probs_flat * targets_flat).sum(dim=-1)
		probs_sum = (probs_flat ** 2).sum(dim=-1)
		targets_sum = (targets_flat ** 2).sum(dim=-1)
		
		dice = (2.0 * intersection + self.smooth) / (probs_sum + targets_sum + self.smooth)
		loss = 1.0 - dice
		
		loss = loss.mean(dim=1)
		
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'none':
			return loss
		else:
			raise ValueError(f"不支持的reduction: {self.reduction}")


class GeneralizedDiceLoss(nn.Module):
	"""
	广义Dice损失 (Generalized Dice Loss)

	对不同类别加权，缓解类别不平衡问题
	权重与每个类别的体积成反比

	公式：
		w_c = 1 / (sum(G_c)^2 + eps)
		GDL = 1 - 2 * sum(w_c * sum(P_c * G_c)) / sum(w_c * (sum(P_c) + sum(G_c)))

	参考：
		Sudre et al., "Generalised Dice overlap as a deep learning loss function
		for highly unbalanced segmentations", DLMIA 2017
	"""
	
	def __init__(
			self,
			smooth: float = 1e-5,
			sigmoid: bool = True,
			reduction: str = 'mean'
	):
		super().__init__()
		self.smooth = smooth
		self.sigmoid = sigmoid
		self.reduction = reduction
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算广义Dice损失"""
		if self.sigmoid:
			probs = torch.sigmoid(logits)
		else:
			probs = logits
		
		targets = targets.float()
		
		# 展平: [B, C, N]
		probs_flat = probs.view(probs.size(0), probs.size(1), -1)
		targets_flat = targets.view(targets.size(0), targets.size(1), -1)
		
		# 计算每个类别的权重（体积的倒数）
		# [B, C]
		target_sum = targets_flat.sum(dim=-1)
		weights = 1.0 / (target_sum ** 2 + self.smooth)
		
		# 加权交集和并集
		intersection = (probs_flat * targets_flat).sum(dim=-1)  # [B, C]
		union = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)  # [B, C]
		
		weighted_intersection = (weights * intersection).sum(dim=1)  # [B]
		weighted_union = (weights * union).sum(dim=1)  # [B]
		
		# 广义Dice
		gdl = 1.0 - (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
		
		if self.reduction == 'mean':
			return gdl.mean()
		elif self.reduction == 'none':
			return gdl
		else:
			raise ValueError(f"不支持的reduction: {self.reduction}")