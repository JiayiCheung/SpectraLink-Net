# scripts/train.py
"""
SpectraLink-Net 训练脚本
支持单GPU和多GPU分布式训练
"""

import os
import sys
import argparse
import yaml
import time
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SpectraLinkNet
from losses import build_loss, DiceBCELoss
from data import get_dataloaders
from utils import Logger, CheckpointManager, calculate_dice, MetricTracker


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='SpectraLink-Net Training')
	parser.add_argument('--config', type=str, default='config/config.yaml',
	                    help='配置文件路径')
	parser.add_argument('--resume', type=str, default=None,
	                    help='恢复训练的检查点路径')
	parser.add_argument('--local_rank', type=int, default=-1,
	                    help='分布式训练的local rank')
	return parser.parse_args()


def load_config(config_path: str) -> dict:
	"""加载配置文件"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config


def setup_distributed():
	"""设置分布式训练环境"""
	if 'LOCAL_RANK' in os.environ:
		local_rank = int(os.environ['LOCAL_RANK'])
		torch.cuda.set_device(local_rank)
		dist.init_process_group(backend='nccl')
		rank = dist.get_rank()
		world_size = dist.get_world_size()
		return local_rank, rank, world_size, True
	else:
		return 0, 0, 1, False


def cleanup_distributed():
	"""清理分布式环境"""
	if dist.is_initialized():
		dist.destroy_process_group()


def create_model(config: dict, device: torch.device) -> nn.Module:
	"""创建模型"""
	model_config = config.get('model', {})
	
	model = SpectraLinkNet(
		in_channels=model_config.get('in_channels', 1),
		out_channels=model_config.get('out_channels', 1),
		base_channels=model_config.get('base_channels', 32),
		num_levels=model_config.get('num_levels', 3),
		freq_token_base=model_config.get('freq_token_base', 4),
		expansion_ratio=model_config.get('expansion_ratio', 2),
		dropout=model_config.get('dropout', 0.0)
	)
	
	model = model.to(device)
	return model


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
	"""创建优化器"""
	train_config = config.get('training', {})
	
	optimizer = AdamW(
		model.parameters(),
		lr=float(train_config.get('learning_rate', 1e-3)),
		weight_decay=float(train_config.get('weight_decay', 1e-5)),
		betas=(0.9, 0.999)
	)
	
	return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_epochs: int):
	"""创建学习率调度器"""
	train_config = config.get('training', {})
	scheduler_type = train_config.get('scheduler', 'cosine')
	
	if scheduler_type == 'cosine':
		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=num_epochs,
			eta_min=float(train_config.get('min_lr', 1e-6))
		)
	elif scheduler_type == 'plateau':
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode='max',
			factor=0.5,
			patience=10,
			min_lr=float(train_config.get('min_lr', 1e-6))
		)
	else:
		scheduler = None
	
	return scheduler


def train_one_epoch(
		model: nn.Module,
		train_loader,
		criterion,
		optimizer,
		device: torch.device,
		epoch: int,
		logger: Logger
) -> float:
	"""训练一个epoch"""
	model.train()
	
	total_loss = 0.0
	num_batches = len(train_loader)
	
	for batch_idx, (images, labels) in enumerate(train_loader):
		# 移动数据到设备
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		
		# 前向传播
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# 反向传播
		optimizer.zero_grad()
		loss.backward()
		
		# 梯度裁剪
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		
		optimizer.step()
		
		total_loss += loss.item()
		
		# 日志
		if batch_idx % 10 == 0:
			current_lr = optimizer.param_groups[0]['lr']
			logger.log(
				f"Epoch {epoch} [{batch_idx}/{num_batches}] "
				f"Loss: {loss.item():.4f} LR: {current_lr:.2e}"
			)
	
	avg_loss = total_loss / num_batches
	return avg_loss


@torch.no_grad()
def validate(
		model: nn.Module,
		val_loader,
		device: torch.device,
		logger: Logger
) -> dict:
	"""验证"""
	model.eval()
	
	metric_tracker = MetricTracker()
	
	for images, labels, case_ids in val_loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		
		# 前向传播
		outputs = model(images)
		probs = torch.sigmoid(outputs)
		
		# 计算Dice
		dice = calculate_dice(probs, labels)
		metric_tracker.update({'dice': dice})
	
	metrics = metric_tracker.get_average()
	return metrics


def main():
	"""主函数"""
	args = parse_args()
	
	# 设置分布式
	local_rank, rank, world_size, distributed = setup_distributed()
	device = torch.device(f'cuda:{local_rank}')
	
	# 加载配置
	config = load_config(args.config)
	
	# 创建日志目录
	log_dir = config.get('logging', {}).get('log_dir', 'outputs/logs')
	checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'outputs/checkpoints')
	
	# 初始化日志和检查点管理器
	logger = Logger(log_dir, name='train', rank=rank)
	ckpt_manager = CheckpointManager(checkpoint_dir, max_keep=5, rank=rank)
	
	# 记录配置
	logger.log_config(config)
	
	# 创建模型
	model = create_model(config, device)
	logger.log_model_info(model)
	
	# 分布式包装
	if distributed:
		model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	
	# 创建优化器和调度器
	optimizer = create_optimizer(model, config)
	
	train_config = config.get('training', {})
	num_epochs = train_config.get('num_epochs', 200)
	scheduler = create_scheduler(optimizer, config, num_epochs)
	
	# 创建损失函数
	criterion = DiceBCELoss(
		w_dice=config.get('loss', {}).get('w_dice', 0.5),
		w_bce=config.get('loss', {}).get('w_bce', 0.5),
		pos_weight=config.get('loss', {}).get('pos_weight', 6.0)
	).to(device)
	
	# 创建数据加载器
	train_loader, val_loader = get_dataloaders(
		config, distributed=distributed, rank=rank, world_size=world_size
	)
	
	# 恢复训练
	start_epoch = 0
	best_dice = 0.0
	
	if args.resume:
		loaded, info = ckpt_manager.load(
			args.resume, model, optimizer, scheduler, device
		)
		if loaded:
			start_epoch = info['epoch'] + 1
			best_dice = info.get('metrics', {}).get('dice', 0.0)
			logger.log(f"从epoch {start_epoch}恢复训练，最佳Dice: {best_dice:.4f}")
	
	# 训练循环
	logger.log(f"开始训练，共{num_epochs}个epoch")
	
	validate_every = train_config.get('validate_every', 5)
	save_every = train_config.get('save_every', 10)
	
	for epoch in range(start_epoch, num_epochs):
		# 设置epoch（分布式采样器需要）
		if distributed:
			train_loader.sampler.set_epoch(epoch)
		
		# 训练
		train_loss = train_one_epoch(
			model, train_loader, criterion, optimizer, device, epoch, logger
		)
		
		# 更新学习率
		current_lr = optimizer.param_groups[0]['lr']
		if scheduler is not None:
			if isinstance(scheduler, ReduceLROnPlateau):
				pass  # 验证后更新
			else:
				scheduler.step()
		
		# 验证
		val_metrics = None
		if (epoch + 1) % validate_every == 0:
			val_metrics = validate(model, val_loader, device, logger)
			
			if isinstance(scheduler, ReduceLROnPlateau):
				scheduler.step(val_metrics['dice'])
			
			# 检查是否是最佳
			is_best = val_metrics['dice'] > best_dice
			if is_best:
				best_dice = val_metrics['dice']
				logger.log(f"新的最佳Dice: {best_dice:.4f}")
		else:
			is_best = False
		
		# 记录epoch
		logger.log_epoch(epoch, train_loss, val_metrics, current_lr)
		
		# 保存检查点
		if (epoch + 1) % save_every == 0 or is_best:
			ckpt_manager.save(
				model, optimizer, epoch,
				metrics=val_metrics,
				scheduler=scheduler,
				is_best=is_best
			)
		
		# 清理内存
		if (epoch + 1) % 10 == 0:
			torch.cuda.empty_cache()
			gc.collect()
		
		# 同步
		if distributed:
			dist.barrier()
	
	# 训练结束
	logger.log(f"训练完成！最佳Dice: {best_dice:.4f}")
	logger.close()
	
	cleanup_distributed()


if __name__ == '__main__':
	main()