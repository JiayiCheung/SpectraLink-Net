# utils/logger.py
"""
日志记录器
TensorBoard日志和控制台日志
"""

import os
import sys
import time
import datetime
import logging
from typing import Dict, Optional, Any
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
	"""
	统一日志记录器

	同时支持控制台输出、文件日志和TensorBoard

	参数:
		log_dir: 日志目录
		name: 日志器名称
		rank: 分布式训练的rank（只有rank=0记录）
	"""
	
	def __init__(
			self,
			log_dir: str,
			name: str = 'train',
			rank: int = 0
	):
		self.log_dir = log_dir
		self.name = name
		self.rank = rank
		self.is_main = (rank == 0)
		
		if self.is_main:
			os.makedirs(log_dir, exist_ok=True)
			
			# 设置Python日志
			self._setup_file_logger()
			
			# 设置TensorBoard
			self.tb_writer = SummaryWriter(log_dir)
			
			# 记录开始时间
			self.start_time = time.time()
			self.log(f"日志目录: {log_dir}")
			self.log(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		else:
			self.tb_writer = None
			self.file_logger = None
	
	def _setup_file_logger(self):
		"""设置文件日志"""
		self.file_logger = logging.getLogger(self.name)
		self.file_logger.setLevel(logging.INFO)
		
		# 文件处理器
		log_file = os.path.join(self.log_dir, f'{self.name}.log')
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(logging.INFO)
		
		# 格式
		formatter = logging.Formatter(
			'%(asctime)s - %(levelname)s - %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		file_handler.setFormatter(formatter)
		
		self.file_logger.addHandler(file_handler)
	
	def log(self, message: str, level: str = 'info'):
		"""
		记录文本日志

		参数:
			message: 日志消息
			level: 日志级别 ('info', 'warning', 'error')
		"""
		if not self.is_main:
			return
		
		# 控制台输出
		print(message)
		
		# 文件日志
		if self.file_logger:
			if level == 'info':
				self.file_logger.info(message)
			elif level == 'warning':
				self.file_logger.warning(message)
			elif level == 'error':
				self.file_logger.error(message)
	
	def log_scalar(self, tag: str, value: float, step: int):
		"""
		记录标量到TensorBoard

		参数:
			tag: 标签名
			value: 数值
			step: 步数
		"""
		if self.is_main and self.tb_writer:
			self.tb_writer.add_scalar(tag, value, step)
	
	def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
		"""
		记录多个标量到TensorBoard

		参数:
			main_tag: 主标签
			values: 数值字典
			step: 步数
		"""
		if self.is_main and self.tb_writer:
			self.tb_writer.add_scalars(main_tag, values, step)
	
	def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
		"""
		记录指标

		参数:
			metrics: 指标字典
			step: 步数
			prefix: 标签前缀
		"""
		if not self.is_main:
			return
		
		for name, value in metrics.items():
			tag = f"{prefix}/{name}" if prefix else name
			self.log_scalar(tag, value, step)
	
	def log_histogram(self, tag: str, values: torch.Tensor, step: int):
		"""记录直方图"""
		if self.is_main and self.tb_writer:
			self.tb_writer.add_histogram(tag, values, step)
	
	def log_image(self, tag: str, image: torch.Tensor, step: int):
		"""
		记录图像

		参数:
			tag: 标签
			image: 图像张量 [C, H, W]
			step: 步数
		"""
		if self.is_main and self.tb_writer:
			self.tb_writer.add_image(tag, image, step)
	
	def log_images(self, tag: str, images: torch.Tensor, step: int):
		"""
		记录多张图像

		参数:
			tag: 标签
			images: 图像张量 [N, C, H, W]
			step: 步数
		"""
		if self.is_main and self.tb_writer:
			self.tb_writer.add_images(tag, images, step)
	
	def log_epoch(
			self,
			epoch: int,
			train_loss: float,
			val_metrics: Optional[Dict[str, float]] = None,
			lr: Optional[float] = None
	):
		"""
		记录每个epoch的信息

		参数:
			epoch: 当前epoch
			train_loss: 训练损失
			val_metrics: 验证指标（可选）
			lr: 当前学习率（可选）
		"""
		if not self.is_main:
			return
		
		# 构建日志消息
		elapsed = time.time() - self.start_time
		elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
		
		msg = f"Epoch {epoch} | Loss: {train_loss:.4f}"
		
		if lr is not None:
			msg += f" | LR: {lr:.2e}"
		
		if val_metrics:
			dice = val_metrics.get('dice', 0)
			msg += f" | Val Dice: {dice:.4f}"
		
		msg += f" | Time: {elapsed_str}"
		
		self.log(msg)
		
		# TensorBoard
		self.log_scalar('train/loss', train_loss, epoch)
		if lr is not None:
			self.log_scalar('train/lr', lr, epoch)
		if val_metrics:
			self.log_metrics(val_metrics, epoch, prefix='val')
	
	def log_config(self, config: Dict[str, Any]):
		"""记录配置信息"""
		if not self.is_main:
			return
		
		self.log("=" * 50)
		self.log("配置信息:")
		self._log_dict(config)
		self.log("=" * 50)
		
		# 保存配置到TensorBoard
		if self.tb_writer:
			config_str = self._dict_to_str(config)
			self.tb_writer.add_text('config', config_str, 0)
	
	def _log_dict(self, d: Dict, indent: int = 0):
		"""递归记录字典"""
		prefix = "  " * indent
		for key, value in d.items():
			if isinstance(value, dict):
				self.log(f"{prefix}{key}:")
				self._log_dict(value, indent + 1)
			else:
				self.log(f"{prefix}{key}: {value}")
	
	def _dict_to_str(self, d: Dict, indent: int = 0) -> str:
		"""字典转字符串"""
		lines = []
		prefix = "  " * indent
		for key, value in d.items():
			if isinstance(value, dict):
				lines.append(f"{prefix}**{key}**:")
				lines.append(self._dict_to_str(value, indent + 1))
			else:
				lines.append(f"{prefix}- {key}: {value}")
		return "\n".join(lines)
	
	def log_model_info(self, model: torch.nn.Module):
		"""记录模型信息"""
		if not self.is_main:
			return
		
		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		
		self.log(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
		self.log(f"可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
	
	def close(self):
		"""关闭日志器"""
		if self.is_main:
			elapsed = time.time() - self.start_time
			elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
			self.log(f"总用时: {elapsed_str}")
			self.log(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
			
			if self.tb_writer:
				self.tb_writer.close()


class ProgressBar:
	"""
	进度条

	用于显示训练进度
	"""
	
	def __init__(self, total: int, desc: str = '', rank: int = 0):
		self.total = total
		self.desc = desc
		self.rank = rank
		self.is_main = (rank == 0)
		self.current = 0
		self.start_time = time.time()
	
	def update(self, n: int = 1, **kwargs):
		"""更新进度"""
		self.current += n
		
		if not self.is_main:
			return
		
		# 计算进度
		progress = self.current / self.total
		elapsed = time.time() - self.start_time
		
		if self.current > 0:
			eta = elapsed / self.current * (self.total - self.current)
		else:
			eta = 0
		
		# 构建进度条
		bar_width = 30
		filled = int(bar_width * progress)
		bar = '█' * filled + '░' * (bar_width - filled)
		
		# 构建信息
		info = f"\r{self.desc} |{bar}| {self.current}/{self.total}"
		info += f" [{elapsed:.0f}s<{eta:.0f}s]"
		
		for key, value in kwargs.items():
			if isinstance(value, float):
				info += f" {key}: {value:.4f}"
			else:
				info += f" {key}: {value}"
		
		print(info, end='', flush=True)
	
	def close(self):
		"""关闭进度条"""
		if self.is_main:
			print()  # 换行