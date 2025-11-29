# data/dataloader.py
"""
数据加载器
创建训练和验证的DataLoader
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Dict, Optional

from .dataset import VesselDataset, VesselDatasetFullVolume


def get_train_dataloader(
		config: Dict,
		distributed: bool = False,
		rank: int = 0,
		world_size: int = 1
) -> DataLoader:
	"""
	创建训练数据加载器

	参数:
		config: 配置字典
		distributed: 是否分布式训练
		rank: 当前进程rank
		world_size: 总进程数

	返回:
		训练DataLoader
	"""
	data_config = config.get('data', {})
	
	dataset = VesselDataset(
		data_dir=data_config.get('data_dir'),
		props_dir=data_config.get('props_dir'),
		split_file=data_config.get('split_file'),
		split='train',
		patch_size=tuple(data_config.get('patch_size', [64, 128, 128])),
		oversample_foreground=data_config.get('oversample_foreground', 0.7),
		samples_per_volume=data_config.get('samples_per_volume', 4),
		mode='train'
	)
	
	if distributed:
		sampler = DistributedSampler(
			dataset,
			num_replicas=world_size,
			rank=rank,
			shuffle=True
		)
		shuffle = False
	else:
		sampler = None
		shuffle = True
	
	dataloader = DataLoader(
		dataset,
		batch_size=data_config.get('batch_size', 2),
		shuffle=shuffle,
		sampler=sampler,
		num_workers=data_config.get('num_workers', 4),
		pin_memory=True,
		drop_last=True,
		persistent_workers=True if data_config.get('num_workers', 4) > 0 else False
	)
	
	return dataloader


def get_val_dataloader(
		config: Dict,
		distributed: bool = False,
		rank: int = 0,
		world_size: int = 1,
		full_volume: bool = False
) -> DataLoader:
	"""
	创建验证数据加载器

	参数:
		config: 配置字典
		distributed: 是否分布式训练
		rank: 当前进程rank
		world_size: 总进程数
		full_volume: 是否加载完整volume

	返回:
		验证DataLoader
	"""
	data_config = config.get('data', {})
	
	if full_volume:
		dataset = VesselDatasetFullVolume(
			data_dir=data_config.get('data_dir'),
			props_dir=data_config.get('props_dir'),
			split_file=data_config.get('split_file'),
			split='val'
		)
	else:
		dataset = VesselDataset(
			data_dir=data_config.get('data_dir'),
			props_dir=data_config.get('props_dir'),
			split_file=data_config.get('split_file'),
			split='val',
			patch_size=tuple(data_config.get('patch_size', [64, 128, 128])),
			oversample_foreground=0.0,  # 验证时不过采样
			samples_per_volume=1,
			mode='val'
		)
	
	if distributed:
		sampler = DistributedSampler(
			dataset,
			num_replicas=world_size,
			rank=rank,
			shuffle=False
		)
	else:
		sampler = None
	
	dataloader = DataLoader(
		dataset,
		batch_size=1,  # 验证时batch_size=1
		shuffle=False,
		sampler=sampler,
		num_workers=max(1, data_config.get('num_workers', 4) // 2),
		pin_memory=True,
		drop_last=False
	)
	
	return dataloader


def get_dataloaders(
		config: Dict,
		distributed: bool = False,
		rank: int = 0,
		world_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
	"""
	同时创建训练和验证数据加载器

	参数:
		config: 配置字典
		distributed: 是否分布式训练
		rank: 当前进程rank
		world_size: 总进程数

	返回:
		(train_loader, val_loader)
	"""
	train_loader = get_train_dataloader(config, distributed, rank, world_size)
	val_loader = get_val_dataloader(config, distributed, rank, world_size)
	
	return train_loader, val_loader


def get_test_dataloader(
		config: Dict,
		split: str = 'test'
) -> DataLoader:
	"""
	创建测试数据加载器

	参数:
		config: 配置字典
		split: 数据集划分（'test' 或 'val'）

	返回:
		测试DataLoader（加载完整volume）
	"""
	data_config = config.get('data', {})
	
	dataset = VesselDatasetFullVolume(
		data_dir=data_config.get('data_dir'),
		props_dir=data_config.get('props_dir'),
		split_file=data_config.get('split_file'),
		split=split
	)
	
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=2,
		pin_memory=True
	)
	
	return dataloader