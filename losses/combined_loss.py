# losses/combined_loss.py
"""
组合损失函数
将多种损失函数加权组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .dice_loss import DiceLoss, SoftDiceLoss
from .cldice_loss import ClDiceLoss


class BCEWithLogitsLoss(nn.Module):
	"""
	带logits的二元交叉熵损失

	封装nn.BCEWithLogitsLoss，支持正类权重

	参数:
		pos_weight: 正类权重，用于处理类别不平衡
		reduction: 归约方式
	"""
	
	def __init__(
			self,
			pos_weight: Optional[float] = None,
			reduction: str = 'mean'
	):
		super().__init__()
		self.pos_weight = pos_weight
		self.reduction = reduction
		
		if pos_weight is not None:
			pw = torch.tensor([pos_weight])
			self.register_buffer('pw_buffer', pw)
		else:
			self.pw_buffer = None
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算BCE损失"""
		targets = targets.float()
		
		if self.pw_buffer is not None:
			pw = self.pw_buffer.to(logits.device)
			return F.binary_cross_entropy_with_logits(
				logits, targets,
				pos_weight=pw,
				reduction=self.reduction
			)
		else:
			return F.binary_cross_entropy_with_logits(
				logits, targets,
				reduction=self.reduction
			)


class FocalLoss(nn.Module):
	"""
	Focal Loss

	对难分类样本加大权重，缓解类别不平衡

	公式:
		FL = -alpha * (1 - p)^gamma * log(p)    当y=1
		FL = -(1-alpha) * p^gamma * log(1-p)    当y=0

	参考:
		Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

	参数:
		alpha: 正类权重（默认0.25）
		gamma: 聚焦参数，越大越关注难样本（默认2.0）
		reduction: 归约方式
	"""
	
	def __init__(
			self,
			alpha: float = 0.25,
			gamma: float = 2.0,
			reduction: str = 'mean'
	):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算Focal Loss"""
		targets = targets.float()
		
		# BCE loss (不归约)
		bce_loss = F.binary_cross_entropy_with_logits(
			logits, targets, reduction='none'
		)
		
		# 概率
		probs = torch.sigmoid(logits)
		
		# pt = p if y=1 else 1-p
		pt = probs * targets + (1 - probs) * (1 - targets)
		
		# alpha_t = alpha if y=1 else 1-alpha
		alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
		
		# Focal weight
		focal_weight = alpha_t * (1 - pt) ** self.gamma
		
		# Focal loss
		focal_loss = focal_weight * bce_loss
		
		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		elif self.reduction == 'none':
			return focal_loss
		else:
			raise ValueError(f"不支持的reduction: {self.reduction}")


class CombinedLoss(nn.Module):
	"""
	组合损失函数

	将Dice、BCE、clDice等损失加权组合

	Loss = w_dice * DiceLoss + w_bce * BCELoss + w_cldice * clDiceLoss

	参数:
		w_dice: Dice损失权重（默认0.5）
		w_bce: BCE损失权重（默认0.5）
		w_cldice: clDice损失权重（默认0.0，不使用）
		pos_weight: BCE正类权重（默认6.0）
		cldice_iterations: clDice骨架化迭代次数
		smooth: 平滑项

	输入:
		logits: [B, C, D, H, W] 预测logits
		targets: [B, C, D, H, W] 真实标签

	输出:
		loss: 组合损失
		loss_dict: 各损失分量的字典（可选）
	"""
	
	def __init__(
			self,
			w_dice: float = 0.5,
			w_bce: float = 0.5,
			w_cldice: float = 0.0,
			pos_weight: float = 6.0,
			cldice_iterations: int = 10,
			smooth: float = 1e-5
	):
		super().__init__()
		
		self.w_dice = w_dice
		self.w_bce = w_bce
		self.w_cldice = w_cldice
		
		# 损失函数
		self.dice_loss = DiceLoss(smooth=smooth, sigmoid=True)
		self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
		
		if w_cldice > 0:
			self.cldice_loss = ClDiceLoss(
				num_iterations=cldice_iterations,
				smooth=smooth,
				sigmoid=True
			)
		else:
			self.cldice_loss = None
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor,
			return_dict: bool = False
	):
		"""
		计算组合损失

		参数:
			logits: 预测logits
			targets: 真实标签
			return_dict: 是否返回各损失分量

		返回:
			loss 或 (loss, loss_dict)
		"""
		loss_dict = {}
		total_loss = 0.0
		
		# Dice Loss
		if self.w_dice > 0:
			dice = self.dice_loss(logits, targets)
			loss_dict['dice'] = dice.item()
			total_loss = total_loss + self.w_dice * dice
		
		# BCE Loss
		if self.w_bce > 0:
			bce = self.bce_loss(logits, targets)
			loss_dict['bce'] = bce.item()
			total_loss = total_loss + self.w_bce * bce
		
		# clDice Loss
		if self.w_cldice > 0 and self.cldice_loss is not None:
			cldice = self.cldice_loss(logits, targets)
			loss_dict['cldice'] = cldice.item()
			total_loss = total_loss + self.w_cldice * cldice
		
		loss_dict['total'] = total_loss.item()
		
		if return_dict:
			return total_loss, loss_dict
		else:
			return total_loss


class DiceBCELoss(nn.Module):
	"""
	简化版Dice + BCE损失

	这是论文中使用的损失函数配置

	Loss = w_dice * DiceLoss + w_bce * BCELoss(pos_weight=6)
	"""
	
	def __init__(
			self,
			w_dice: float = 0.5,
			w_bce: float = 0.5,
			pos_weight: float = 6.0,
			smooth: float = 1e-5
	):
		super().__init__()
		self.w_dice = w_dice
		self.w_bce = w_bce
		self.smooth = smooth
		
		# BCE with pos_weight
		self.register_buffer(
			'pos_weight',
			torch.tensor([pos_weight])
		)
	
	def forward(
			self,
			logits: torch.Tensor,
			targets: torch.Tensor
	) -> torch.Tensor:
		"""计算Dice + BCE损失"""
		targets = targets.float()
		
		# BCE
		bce = F.binary_cross_entropy_with_logits(
			logits, targets,
			pos_weight=self.pos_weight.to(logits.device)
		)
		
		# Dice
		probs = torch.sigmoid(logits)
		probs_flat = probs.view(-1)
		targets_flat = targets.view(-1)
		
		intersection = (probs_flat * targets_flat).sum()
		union = probs_flat.sum() + targets_flat.sum()
		dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
		dice_loss = 1.0 - dice
		
		# Combined
		return self.w_dice * dice_loss + self.w_bce * bce


def build_loss(config: Dict) -> nn.Module:
	"""
	根据配置构建损失函数

	参数:
		config: 损失函数配置字典

	返回:
		损失函数实例
	"""
	loss_config = config.get('loss', {})
	loss_type = loss_config.get('type', 'combined')
	
	if loss_type == 'dice':
		return DiceLoss(
			smooth=loss_config.get('smooth', 1e-5),
			sigmoid=True
		)
	
	elif loss_type == 'bce':
		return BCEWithLogitsLoss(
			pos_weight=loss_config.get('pos_weight', 6.0)
		)
	
	elif loss_type == 'focal':
		return FocalLoss(
			alpha=loss_config.get('alpha', 0.25),
			gamma=loss_config.get('gamma', 2.0)
		)
	
	elif loss_type == 'cldice':
		return ClDiceLoss(
			num_iterations=loss_config.get('cldice_iterations', 10),
			smooth=loss_config.get('smooth', 1e-5),
			sigmoid=True
		)
	
	elif loss_type == 'dice_bce':
		return DiceBCELoss(
			w_dice=loss_config.get('w_dice', 0.5),
			w_bce=loss_config.get('w_bce', 0.5),
			pos_weight=loss_config.get('pos_weight', 6.0)
		)
	
	elif loss_type == 'combined':
		return CombinedLoss(
			w_dice=loss_config.get('w_dice', 0.5),
			w_bce=loss_config.get('w_bce', 0.5),
			w_cldice=loss_config.get('w_cldice', 0.0),
			pos_weight=loss_config.get('pos_weight', 6.0),
			cldice_iterations=loss_config.get('cldice_iterations', 10)
		)
	
	else:
		raise ValueError(f"不支持的损失类型: {loss_type}")