# utils/checkpoint.py
"""
检查点管理
保存和加载模型检查点
"""

import os
import glob
import shutil
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
	"""
	检查点管理器

	功能：
	- 保存最新检查点
	- 保存最佳检查点
	- 定期保存检查点
	- 自动清理旧检查点

	参数:
		checkpoint_dir: 检查点保存目录
		max_keep: 最多保留的检查点数量
		rank: 分布式训练的rank（只有rank=0保存）
	"""
	
	def __init__(
			self,
			checkpoint_dir: str,
			max_keep: int = 5,
			rank: int = 0
	):
		self.checkpoint_dir = checkpoint_dir
		self.max_keep = max_keep
		self.rank = rank
		self.is_main = (rank == 0)
		
		if self.is_main:
			os.makedirs(checkpoint_dir, exist_ok=True)
		
		self.best_metric = None
		self.best_epoch = None
	
	def save(
			self,
			model: nn.Module,
			optimizer: torch.optim.Optimizer,
			epoch: int,
			metrics: Optional[Dict[str, float]] = None,
			scheduler: Optional[Any] = None,
			is_best: bool = False,
			filename: Optional[str] = None
	) -> Optional[str]:
		"""
		保存检查点

		参数:
			model: 模型
			optimizer: 优化器
			epoch: 当前epoch
			metrics: 当前指标
			scheduler: 学习率调度器（可选）
			is_best: 是否是最佳模型
			filename: 自定义文件名（可选）

		返回:
			保存的文件路径
		"""
		if not self.is_main:
			return None
		
		# 获取模型状态（处理DDP包装）
		if hasattr(model, 'module'):
			model_state = model.module.state_dict()
		else:
			model_state = model.state_dict()
		
		# 构建检查点
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': model_state,
			'optimizer_state_dict': optimizer.state_dict(),
			'metrics': metrics,
		}
		
		if scheduler is not None:
			checkpoint['scheduler_state_dict'] = scheduler.state_dict()
		
		# 保存
		if filename is None:
			filename = f'checkpoint_epoch_{epoch:04d}.pth'
		
		filepath = os.path.join(self.checkpoint_dir, filename)
		torch.save(checkpoint, filepath)
		
		# 更新latest链接
		latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
		if os.path.exists(latest_path):
			os.remove(latest_path)
		shutil.copy(filepath, latest_path)
		
		# 如果是最佳模型，额外保存
		if is_best:
			best_path = os.path.join(self.checkpoint_dir, 'best.pth')
			shutil.copy(filepath, best_path)
			
			if metrics:
				self.best_metric = metrics.get('dice', 0)
			self.best_epoch = epoch
		
		# 清理旧检查点
		self._cleanup_old_checkpoints()
		
		return filepath
	
	def _cleanup_old_checkpoints(self):
		"""清理旧检查点，只保留最新的max_keep个"""
		pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
		checkpoints = sorted(glob.glob(pattern))
		
		if len(checkpoints) > self.max_keep:
			for ckpt in checkpoints[:-self.max_keep]:
				os.remove(ckpt)
	
	def load(
			self,
			filepath: str,
			model: nn.Module,
			optimizer: Optional[torch.optim.Optimizer] = None,
			scheduler: Optional[Any] = None,
			device: Optional[torch.device] = None,
			strict: bool = True
	) -> Dict[str, Any]:
		"""
		加载检查点

		参数:
			filepath: 检查点路径
			model: 模型
			optimizer: 优化器（可选）
			scheduler: 调度器（可选）
			device: 目标设备
			strict: 是否严格匹配参数

		返回:
			检查点信息
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		checkpoint = torch.load(filepath, map_location=device, weights_only=False)
		
		# 加载模型权重
		if hasattr(model, 'module'):
			model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
		else:
			model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
		
		# 加载优化器状态
		if optimizer is not None and 'optimizer_state_dict' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
		# 加载调度器状态
		if scheduler is not None and 'scheduler_state_dict' in checkpoint:
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		
		return {
			'epoch': checkpoint.get('epoch', 0),
			'metrics': checkpoint.get('metrics', {}),
		}
	
	def load_latest(
			self,
			model: nn.Module,
			optimizer: Optional[torch.optim.Optimizer] = None,
			scheduler: Optional[Any] = None,
			device: Optional[torch.device] = None
	) -> Tuple[bool, Dict[str, Any]]:
		"""
		加载最新检查点

		返回:
			(是否成功加载, 检查点信息)
		"""
		latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
		
		if not os.path.exists(latest_path):
			return False, {}
		
		info = self.load(latest_path, model, optimizer, scheduler, device)
		return True, info
	
	def load_best(
			self,
			model: nn.Module,
			device: Optional[torch.device] = None,
			strict: bool = True
	) -> Tuple[bool, Dict[str, Any]]:
		"""
		加载最佳检查点

		返回:
			(是否成功加载, 检查点信息)
		"""
		best_path = os.path.join(self.checkpoint_dir, 'best.pth')
		
		if not os.path.exists(best_path):
			return False, {}
		
		info = self.load(best_path, model, device=device, strict=strict)
		return True, info
	
	def get_best_metric(self) -> Tuple[Optional[float], Optional[int]]:
		"""获取最佳指标和对应epoch"""
		return self.best_metric, self.best_epoch


def save_model_only(
		model: nn.Module,
		filepath: str
):
	"""
	只保存模型权重（用于最终部署）

	参数:
		model: 模型
		filepath: 保存路径
	"""
	if hasattr(model, 'module'):
		model_state = model.module.state_dict()
	else:
		model_state = model.state_dict()
	
	torch.save(model_state, filepath)


def load_model_only(
		model: nn.Module,
		filepath: str,
		device: Optional[torch.device] = None,
		strict: bool = True
):
	"""
	只加载模型权重

	参数:
		model: 模型
		filepath: 权重文件路径
		device: 目标设备
		strict: 是否严格匹配
	"""
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	state_dict = torch.load(filepath, map_location=device, weights_only=True)
	
	if hasattr(model, 'module'):
		model.module.load_state_dict(state_dict, strict=strict)
	else:
		model.load_state_dict(state_dict, strict=strict)