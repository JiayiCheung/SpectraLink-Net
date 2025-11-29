# data/crop.py
"""
非零区域裁剪
从原始NIfTI数据裁剪到非零区域，减少计算量
"""

import os
import gc
import pickle
import numpy as np
import nibabel as nib
from typing import Tuple, List, Optional
from multiprocessing import Pool
from pathlib import Path


def crop_to_nonzero(
		data: np.ndarray,
		seg: Optional[np.ndarray] = None,
		margin: int = 10
) -> Tuple:
	"""
	裁剪到非零区域，保留边距

	参数:
		data: 图像数据 [D, H, W]
		seg: 标签数据 [D, H, W]（可选）
		margin: 边距大小（像素）

	返回:
		(cropped_data, cropped_seg, bbox)
	"""
	# 创建前景掩码
	if seg is not None:
		mask = seg > 0
	else:
		mask = data > np.min(data)
	
	# 检查是否有前景
	if not np.any(mask):
		print("警告: 未检测到前景区域，返回原始数据")
		return data, seg, None
	
	# 获取边界框
	nonzero_indices = np.nonzero(mask)
	min_idx = [max(0, np.min(idx) - margin) for idx in nonzero_indices]
	max_idx = [min(mask.shape[i], np.max(nonzero_indices[i]) + margin + 1)
	           for i in range(len(nonzero_indices))]
	bbox = (min_idx, max_idx)
	
	# 裁剪数据
	slices = tuple(slice(min_idx[i], max_idx[i]) for i in range(len(min_idx)))
	cropped_data = data[slices]
	
	if seg is not None:
		cropped_seg = seg[slices]
		return cropped_data, cropped_seg, bbox
	else:
		return cropped_data, None, bbox


def process_case(args: Tuple) -> None:
	"""
	处理单个案例的裁剪

	参数:
		args: (image_file, label_file, output_dir, case_id)
	"""
	image_file, label_file, output_dir, case_id = args
	
	try:
		# 创建输出目录
		images_dir = os.path.join(output_dir, 'cropped', 'images')
		labels_dir = os.path.join(output_dir, 'cropped', 'labels')
		props_dir = os.path.join(output_dir, 'cropped', 'properties')
		
		os.makedirs(images_dir, exist_ok=True)
		os.makedirs(labels_dir, exist_ok=True)
		os.makedirs(props_dir, exist_ok=True)
		
		# 加载NIfTI数据
		image_nii = nib.load(image_file)
		image_data = image_nii.get_fdata().astype(np.float32)
		original_spacing = image_nii.header.get_zooms()[:3]
		affine = image_nii.affine
		
		if label_file and os.path.exists(label_file):
			label_nii = nib.load(label_file)
			label_data = label_nii.get_fdata().astype(np.float32)
		else:
			label_data = None
		
		# 执行裁剪
		cropped_image, cropped_label, bbox = crop_to_nonzero(
			image_data, label_data, margin=10
		)
		
		# 保存裁剪后的数据
		np.save(os.path.join(images_dir, f"{case_id}.npy"), cropped_image)
		if cropped_label is not None:
			np.save(os.path.join(labels_dir, f"{case_id}.npy"), cropped_label)
		
		# 保存属性
		properties = {
			'original_spacing': tuple(original_spacing),
			'original_shape': image_data.shape,
			'cropped_shape': cropped_image.shape,
			'bbox': bbox,
			'affine': affine,
			'case_id': case_id
		}
		
		with open(os.path.join(props_dir, f"{case_id}.pkl"), 'wb') as f:
			pickle.dump(properties, f)
		
		# 清理内存
		del image_data, cropped_image
		if label_data is not None:
			del label_data, cropped_label
		gc.collect()
		
		print(f"[裁剪完成] {case_id}: {properties['original_shape']} -> {properties['cropped_shape']}")
	
	except Exception as e:
		print(f"[错误] 处理 {case_id} 时出错: {e}")
		import traceback
		traceback.print_exc()


def run_cropping(
		images_dir: str,
		labels_dir: str,
		output_dir: str,
		num_workers: int = 8
) -> None:
	"""
	批量执行非零裁剪

	参数:
		images_dir: 图像目录
		labels_dir: 标签目录
		output_dir: 输出目录
		num_workers: 并行工作进程数
	"""
	# 获取所有图像文件
	image_files = sorted([
		os.path.join(images_dir, f)
		for f in os.listdir(images_dir)
		if f.endswith('.nii.gz') or f.endswith('.nii')
	])
	
	# 准备参数
	args_list = []
	for img_file in image_files:
		# 提取case_id
		case_id = os.path.basename(img_file).replace('.nii.gz', '').replace('.nii', '')
		
		# 查找对应的标签文件
		label_file = os.path.join(labels_dir, os.path.basename(img_file))
		if not os.path.exists(label_file):
			label_file = None
		
		args_list.append((img_file, label_file, output_dir, case_id))
	
	print(f"开始裁剪 {len(args_list)} 个案例...")
	
	# 执行裁剪
	if num_workers > 1:
		with Pool(num_workers) as p:
			p.map(process_case, args_list)
	else:
		for args in args_list:
			process_case(args)
	
	print(f"裁剪完成，结果保存在 {output_dir}/cropped/")


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='执行非零裁剪')
	parser.add_argument('--images_dir', type=str, required=True, help='图像目录')
	parser.add_argument('--labels_dir', type=str, required=True, help='标签目录')
	parser.add_argument('--output_dir', type=str, default='data/preprocess', help='输出目录')
	parser.add_argument('--num_workers', type=int, default=8, help='工作进程数')
	
	args = parser.parse_args()
	
	run_cropping(args.images_dir, args.labels_dir, args.output_dir, args.num_workers)