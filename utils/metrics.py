# utils/metrics.py
"""
评估指标
用于计算分割性能的各种指标
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.ndimage import distance_transform_edt


def calculate_dice(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		smooth: float = 1e-5
) -> float:
	"""
	计算Dice系数

	Dice = 2 * |P ∩ G| / (|P| + |G|)

	参数:
		pred: 预测概率或logits
		target: 真实标签
		threshold: 二值化阈值
		smooth: 平滑项

	返回:
		Dice系数 (0-1)
	"""
	# 转换为numpy
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	# 二值化
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	# 展平
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	# 计算
	intersection = np.sum(pred_flat * target_flat)
	union = np.sum(pred_flat) + np.sum(target_flat)
	
	dice = (2.0 * intersection + smooth) / (union + smooth)
	
	return float(dice)


def calculate_iou(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		smooth: float = 1e-5
) -> float:
	"""
	计算IoU (Intersection over Union)

	IoU = |P ∩ G| / |P ∪ G|

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		smooth: 平滑项

	返回:
		IoU系数 (0-1)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	intersection = np.sum(pred_flat * target_flat)
	union = np.sum(pred_flat) + np.sum(target_flat) - intersection
	
	iou = (intersection + smooth) / (union + smooth)
	
	return float(iou)


def calculate_precision(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		smooth: float = 1e-5
) -> float:
	"""
	计算精确率

	Precision = TP / (TP + FP)

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		smooth: 平滑项

	返回:
		精确率 (0-1)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	tp = np.sum(pred_flat * target_flat)
	fp = np.sum(pred_flat * (1 - target_flat))
	
	precision = (tp + smooth) / (tp + fp + smooth)
	
	return float(precision)


def calculate_recall(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		smooth: float = 1e-5
) -> float:
	"""
	计算召回率 (灵敏度)

	Recall = TP / (TP + FN)

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		smooth: 平滑项

	返回:
		召回率 (0-1)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	tp = np.sum(pred_flat * target_flat)
	fn = np.sum((1 - pred_flat) * target_flat)
	
	recall = (tp + smooth) / (tp + fn + smooth)
	
	return float(recall)


def calculate_specificity(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		smooth: float = 1e-5
) -> float:
	"""
	计算特异性

	Specificity = TN / (TN + FP)

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		smooth: 平滑项

	返回:
		特异性 (0-1)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	tn = np.sum((1 - pred_flat) * (1 - target_flat))
	fp = np.sum(pred_flat * (1 - target_flat))
	
	specificity = (tn + smooth) / (tn + fp + smooth)
	
	return float(specificity)


def calculate_hausdorff_distance(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		percentile: float = 95
) -> float:
	"""
	计算Hausdorff距离

	使用百分位数版本减少离群点影响

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		percentile: 距离百分位数 (默认95)

	返回:
		Hausdorff距离 (体素单位)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	# 去除batch和channel维度
	pred = pred.squeeze()
	target = target.squeeze()
	
	# 二值化
	pred_binary = (pred > threshold).astype(np.bool_)
	target_binary = (target > 0.5).astype(np.bool_)
	
	# 处理空预测或空标签
	if not np.any(pred_binary) or not np.any(target_binary):
		return float('inf')
	
	# 计算距离变换
	pred_dist = distance_transform_edt(~pred_binary)
	target_dist = distance_transform_edt(~target_binary)
	
	# 获取表面点的距离
	pred_surface_dist = target_dist[pred_binary]
	target_surface_dist = pred_dist[target_binary]
	
	# 计算百分位距离
	hd_pred = np.percentile(pred_surface_dist, percentile)
	hd_target = np.percentile(target_surface_dist, percentile)
	
	return float(max(hd_pred, hd_target))


def calculate_average_surface_distance(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5
) -> float:
	"""
	计算平均表面距离 (ASD)

	ASD = (mean(d(P→G)) + mean(d(G→P))) / 2

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值

	返回:
		平均表面距离 (体素单位)
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred = pred.squeeze()
	target = target.squeeze()
	
	pred_binary = (pred > threshold).astype(np.bool_)
	target_binary = (target > 0.5).astype(np.bool_)
	
	if not np.any(pred_binary) or not np.any(target_binary):
		return float('inf')
	
	# 距离变换
	pred_dist = distance_transform_edt(~pred_binary)
	target_dist = distance_transform_edt(~target_binary)
	
	# 表面点距离
	pred_to_target = target_dist[pred_binary]
	target_to_pred = pred_dist[target_binary]
	
	# 平均
	asd = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2.0
	
	return float(asd)


def calculate_false_rates(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5
) -> Dict[str, float]:
	"""
	计算假阳性率和假阴性率

	FPR = FP / (FP + TN)  过分割率
	FNR = FN / (FN + TP)  欠分割率

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值

	返回:
		包含FPR, FNR, OR, UR的字典
	"""
	if torch.is_tensor(pred):
		pred = pred.detach().cpu().numpy()
	if torch.is_tensor(target):
		target = target.detach().cpu().numpy()
	
	pred_binary = (pred > threshold).astype(np.float32)
	target_binary = (target > 0.5).astype(np.float32)
	
	pred_flat = pred_binary.flatten()
	target_flat = target_binary.flatten()
	
	tp = np.sum(pred_flat * target_flat)
	fp = np.sum(pred_flat * (1 - target_flat))
	tn = np.sum((1 - pred_flat) * (1 - target_flat))
	fn = np.sum((1 - pred_flat) * target_flat)
	
	# 假阳性率
	fpr = fp / (fp + tn + 1e-8)
	
	# 假阴性率
	fnr = fn / (fn + tp + 1e-8)
	
	# 过分割率 (相对于真实前景)
	over_rate = fp / (tp + fn + 1e-8)
	
	# 欠分割率 (相对于真实前景)
	under_rate = fn / (tp + fn + 1e-8)
	
	return {
		'fpr': float(fpr),
		'fnr': float(fnr),
		'over_rate': float(over_rate),
		'under_rate': float(under_rate)
	}


def calculate_all_metrics(
		pred: torch.Tensor,
		target: torch.Tensor,
		threshold: float = 0.5,
		include_surface: bool = True
) -> Dict[str, float]:
	"""
	计算所有指标

	参数:
		pred: 预测
		target: 真实标签
		threshold: 二值化阈值
		include_surface: 是否包含表面距离指标（计算较慢）

	返回:
		包含所有指标的字典
	"""
	metrics = {}
	
	# 基础指标
	metrics['dice'] = calculate_dice(pred, target, threshold)
	metrics['iou'] = calculate_iou(pred, target, threshold)
	metrics['precision'] = calculate_precision(pred, target, threshold)
	metrics['recall'] = calculate_recall(pred, target, threshold)
	metrics['specificity'] = calculate_specificity(pred, target, threshold)
	
	# 错误率
	false_rates = calculate_false_rates(pred, target, threshold)
	metrics.update(false_rates)
	
	# 表面距离指标（可选，计算较慢）
	if include_surface:
		metrics['hd95'] = calculate_hausdorff_distance(pred, target, threshold, percentile=95)
		metrics['asd'] = calculate_average_surface_distance(pred, target, threshold)
	
	return metrics


class MetricTracker:
	"""
	指标追踪器

	累积多个样本的指标，计算平均值
	"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		"""重置所有累积值"""
		self.metrics_sum = {}
		self.count = 0
	
	def update(self, metrics: Dict[str, float]):
		"""
		更新指标

		参数:
			metrics: 单个样本的指标字典
		"""
		for key, value in metrics.items():
			if key not in self.metrics_sum:
				self.metrics_sum[key] = 0.0
			
			# 跳过inf值
			if not np.isinf(value):
				self.metrics_sum[key] += value
		
		self.count += 1
	
	def get_average(self) -> Dict[str, float]:
		"""获取平均指标"""
		if self.count == 0:
			return {}
		
		return {key: value / self.count for key, value in self.metrics_sum.items()}
	
	def get_summary_string(self) -> str:
		"""获取指标摘要字符串"""
		avg_metrics = self.get_average()
		
		parts = []
		for key, value in avg_metrics.items():
			parts.append(f"{key}: {value:.4f}")
		
		return " | ".join(parts)