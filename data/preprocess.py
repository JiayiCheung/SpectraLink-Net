# data/preprocess.py
"""
数据预处理
重采样到统一spacing，归一化，提取前景位置
"""

import os
import gc
import json
import pickle
import numpy as np
from typing import Tuple, List, Dict, Optional
from multiprocessing import Pool
from scipy.ndimage import binary_erosion, zoom
from pathlib import Path


def resample_data(
		data: np.ndarray,
		seg: Optional[np.ndarray],
		original_spacing: Tuple[float, float, float],
		target_spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	"""
	重采样数据到目标spacing

	参数:
		data: 图像数据 [D, H, W]
		seg: 标签数据 [D, H, W]（可选）
		original_spacing: 原始体素间距
		target_spacing: 目标体素间距

	返回:
		(resampled_data, resampled_seg)
	"""
	# 检查是否需要重采样
	if np.allclose(original_spacing, target_spacing, rtol=0.01):
		return data, seg
	
	# 计算缩放因子
	scale_factors = np.array(original_spacing) / np.array(target_spacing)
	
	# 重采样图像（三次插值）
	resampled_data = zoom(data, scale_factors, order=3, mode='nearest')
	
	# 重采样标签（最近邻插值）
	if seg is not None:
		resampled_seg = zoom(seg, scale_factors, order=0, mode='nearest')
	else:
		resampled_seg = None
	
	return resampled_data, resampled_seg


def normalize_ct(
		data: np.ndarray,
		clip_range: Tuple[float, float] = (-200, 300)
) -> Tuple[np.ndarray, Dict]:
	"""
	CT图像归一化

	步骤：
	1. 裁剪到指定HU值范围
	2. Z-score归一化

	参数:
		data: 图像数据
		clip_range: HU值裁剪范围

	返回:
		(normalized_data, norm_params)
	"""
	# 裁剪到合适的HU值范围
	data_clipped = np.clip(data, clip_range[0], clip_range[1])
	
	# Z-score归一化
	mean = np.mean(data_clipped)
	std = np.std(data_clipped)
	normalized = (data_clipped - mean) / (std + 1e-8)
	
	norm_params = {
		'mean': float(mean),
		'std': float(std),
		'clip_range': clip_range
	}
	
	return normalized, norm_params


def extract_foreground_locations(
		seg: np.ndarray,
		max_points: int = 10000
) -> Dict[str, np.ndarray]:
	"""
	提取前景位置，用于训练时的过采样

	参数:
		seg: 分割标签 [D, H, W]
		max_points: 每类最大点数

	返回:
		包含各类位置的字典
	"""
	foreground_mask = seg > 0
	
	if not np.any(foreground_mask):
		return {'foreground': np.array([]), 'boundary': np.array([]), 'interior': np.array([])}
	
	# 提取边界和内部区域
	eroded = binary_erosion(foreground_mask, iterations=2)
	boundary = foreground_mask & (~eroded)
	interior = eroded
	
	# 获取位置点
	foreground_points = np.argwhere(foreground_mask)
	boundary_points = np.argwhere(boundary)
	interior_points = np.argwhere(interior)
	
	# 下采样
	def subsample(points, max_n):
		if len(points) > max_n:
			indices = np.random.choice(len(points), max_n, replace=False)
			return points[indices]
		return points
	
	return {
		'foreground': subsample(foreground_points, max_points),
		'boundary': subsample(boundary_points, max_points // 3),
		'interior': subsample(interior_points, max_points // 3)
	}


def process_case(args: Tuple) -> None:
	"""
	处理单个案例的重采样和归一化

	参数:
		args: (cropped_image_file, cropped_label_file, properties_file,
			   output_dir, target_spacing, case_id)
	"""
	(cropped_image_file, cropped_label_file, properties_file,
	 output_dir, target_spacing, case_id) = args
	
	try:
		# 创建输出目录
		data_dir = os.path.join(output_dir, 'processed', 'data')
		props_dir = os.path.join(output_dir, 'processed', 'properties')
		
		os.makedirs(data_dir, exist_ok=True)
		os.makedirs(props_dir, exist_ok=True)
		
		# 加载裁剪后的数据
		cropped_image = np.load(cropped_image_file)
		if cropped_label_file and os.path.exists(cropped_label_file):
			cropped_label = np.load(cropped_label_file)
		else:
			cropped_label = None
		
		# 加载原始属性
		with open(properties_file, 'rb') as f:
			properties = pickle.load(f)
		
		original_spacing = properties['original_spacing']
		
		# 重采样
		resampled_image, resampled_label = resample_data(
			cropped_image, cropped_label, original_spacing, target_spacing
		)
		
		# 归一化
		normalized_image, norm_params = normalize_ct(resampled_image)
		
		# 提取前景位置
		class_locations = None
		if resampled_label is not None:
			class_locations = extract_foreground_locations(resampled_label)
		
		# 更新属性
		process_props = properties.copy()
		process_props.update({
			'target_spacing': target_spacing,
			'shape_after_resampling': normalized_image.shape,
			'normalization_params': norm_params,
			'class_locations': class_locations
		})
		
		# 保存处理后的数据（图像和标签堆叠）
		if resampled_label is not None:
			# [2, D, H, W]: 第0通道是图像，第1通道是标签
			combined_data = np.stack([normalized_image, resampled_label], axis=0)
		else:
			combined_data = normalized_image[np.newaxis]
		
		np.save(os.path.join(data_dir, f"{case_id}.npy"), combined_data.astype(np.float32))
		
		# 保存属性
		with open(os.path.join(props_dir, f"{case_id}.pkl"), 'wb') as f:
			pickle.dump(process_props, f)
		
		# 清理内存
		del cropped_image, normalized_image, resampled_image
		if cropped_label is not None:
			del cropped_label, resampled_label
		gc.collect()
		
		print(f"[预处理完成] {case_id}: shape={process_props['shape_after_resampling']}")
	
	except Exception as e:
		print(f"[错误] 处理 {case_id} 时出错: {e}")
		import traceback
		traceback.print_exc()


def create_splits(
		output_dir: str,
		val_ratio: float = 0.2,
		seed: int = 42
) -> None:
	"""
	创建训练/验证集划分

	参数:
		output_dir: 输出目录
		val_ratio: 验证集比例
		seed: 随机种子
	"""
	data_dir = os.path.join(output_dir, 'processed', 'data')
	
	# 获取所有case_id
	case_ids = sorted([
		f.replace('.npy', '')
		for f in os.listdir(data_dir)
		if f.endswith('.npy')
	])
	
	# 随机划分
	np.random.seed(seed)
	np.random.shuffle(case_ids)
	
	val_size = max(1, int(len(case_ids) * val_ratio))
	val_ids = case_ids[:val_size]
	train_ids = case_ids[val_size:]
	
	splits = {
		'train': train_ids,
		'val': val_ids
	}
	
	# 保存
	split_file = os.path.join(output_dir, 'splits.json')
	with open(split_file, 'w') as f:
		json.dump(splits, f, indent=2)
	
	print(f"数据集划分: 训练集 {len(train_ids)} 例, 验证集 {len(val_ids)} 例")
	print(f"保存到: {split_file}")


def run_preprocessing(
		cropped_dir: str,
		output_dir: str,
		target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
		num_workers: int = 8
) -> None:
	"""
	执行重采样和归一化

	参数:
		cropped_dir: 裁剪结果目录
		output_dir: 输出目录
		target_spacing: 目标体素间距
		num_workers: 工作进程数
	"""
	images_dir = os.path.join(cropped_dir, 'images')
	labels_dir = os.path.join(cropped_dir, 'labels')
	props_dir = os.path.join(cropped_dir, 'properties')
	
	# 准备参数
	args_list = []
	for img_file in os.listdir(images_dir):
		if img_file.endswith('.npy'):
			case_id = img_file.replace('.npy', '')
			
			cropped_image_file = os.path.join(images_dir, img_file)
			cropped_label_file = os.path.join(labels_dir, img_file)
			properties_file = os.path.join(props_dir, f"{case_id}.pkl")
			
			if not os.path.exists(cropped_label_file):
				cropped_label_file = None
			
			args_list.append((
				cropped_image_file,
				cropped_label_file,
				properties_file,
				output_dir,
				target_spacing,
				case_id
			))
	
	print(f"开始预处理 {len(args_list)} 个案例...")
	
	# 执行预处理
	if num_workers > 1:
		with Pool(num_workers) as p:
			p.map(process_case, args_list)
	else:
		for args in args_list:
			process_case(args)
	
	# 创建数据划分
	create_splits(output_dir)
	
	print(f"预处理完成，结果保存在 {output_dir}/processed/")


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='执行重采样和归一化')
	parser.add_argument('--cropped_dir', type=str, required=True, help='裁剪结果目录')
	parser.add_argument('--output_dir', type=str, default='data/preprocess', help='输出目录')
	parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
	                    help='目标体素间距')
	parser.add_argument('--num_workers', type=int, default=8, help='工作进程数')
	
	args = parser.parse_args()
	
	run_preprocessing(
		args.cropped_dir,
		args.output_dir,
		tuple(args.target_spacing),
		args.num_workers
	)