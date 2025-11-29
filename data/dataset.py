# data/dataset.py
"""
血管分割数据集
加载预处理后的数据，支持patch采样和前景过采样
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional


class VesselDataset(Dataset):
	"""
	血管分割数据集

	加载预处理后的npy文件，按patch采样

	数据格式：
		- data文件: [2, D, H, W]，第0通道是图像，第1通道是标签
		- properties文件: 包含class_locations等元信息

	参数:
		data_dir: 预处理后数据目录（包含.npy文件）
		props_dir: 属性文件目录（包含.pkl文件）
		split_file: 数据集划分文件（JSON格式）
		split: 'train' 或 'val'
		patch_size: 采样patch尺寸 (D, H, W)
		oversample_foreground: 前景过采样比例（0-1）
		samples_per_volume: 每个volume采样的patch数（仅训练时有效）
		mode: 'train', 'val', 或 'test'

	返回:
		train模式: (image, label)
		val/test模式: (image, label, case_id)
	"""
	
	def __init__(
			self,
			data_dir: str,
			props_dir: str,
			split_file: str,
			split: str = 'train',
			patch_size: Tuple[int, int, int] = (64, 128, 128),
			oversample_foreground: float = 0.7,
			samples_per_volume: int = 4,
			mode: str = 'train'
	):
		super().__init__()
		
		self.data_dir = data_dir
		self.props_dir = props_dir
		self.split = split
		self.patch_size = patch_size
		self.oversample_foreground = oversample_foreground
		self.samples_per_volume = samples_per_volume
		self.mode = mode
		
		# 加载数据划分
		with open(split_file, 'r') as f:
			splits = json.load(f)
		
		self.case_ids = splits[split]
		
		# 预加载属性文件
		self.properties = {}
		for case_id in self.case_ids:
			prop_path = os.path.join(props_dir, f"{case_id}.pkl")
			if os.path.exists(prop_path):
				with open(prop_path, 'rb') as f:
					self.properties[case_id] = pickle.load(f)
		
		print(f"[{split}] 加载了 {len(self.case_ids)} 个样本")
	
	def __len__(self) -> int:
		if self.mode == 'train':
			# 训练时：每个volume采样多个patch
			return len(self.case_ids) * self.samples_per_volume
		else:
			# 验证/测试时：每个volume一个样本
			return len(self.case_ids)
	
	def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
		"""加载单个case的数据"""
		data_path = os.path.join(self.data_dir, f"{case_id}.npy")
		data = np.load(data_path)  # [2, D, H, W]
		
		image = data[0:1]  # [1, D, H, W]
		label = data[1:2]  # [1, D, H, W]
		
		return image, label
	
	def _get_random_center(self, shape: Tuple[int, ...]) -> List[int]:
		"""获取随机采样中心点"""
		center = []
		for i, (s, p) in enumerate(zip(shape, self.patch_size)):
			if s <= p:
				# 图像尺寸小于patch，取中心
				center.append(s // 2)
			else:
				# 在有效范围内随机采样
				low = p // 2
				high = s - p // 2
				center.append(np.random.randint(low, high))
		return center
	
	def _get_foreground_center(self, case_id: str, shape: Tuple[int, ...]) -> List[int]:
		"""从前景区域采样中心点"""
		props = self.properties.get(case_id, {})
		class_locs = props.get('class_locations', None)
		
		if class_locs is None or 'foreground' not in class_locs:
			# 没有前景位置信息，退化为随机采样
			return self._get_random_center(shape)
		
		fg_points = class_locs['foreground']
		if len(fg_points) == 0:
			return self._get_random_center(shape)
		
		# 随机选择一个前景点作为中心
		idx = np.random.randint(0, len(fg_points))
		center = fg_points[idx].tolist()
		
		# 确保中心点在有效范围内
		valid_center = []
		for i, (c, s, p) in enumerate(zip(center, shape, self.patch_size)):
			half = p // 2
			c = max(half, min(c, s - half - 1))
			valid_center.append(int(c))
		
		return valid_center
	
	def _extract_patch(
			self,
			image: np.ndarray,
			label: np.ndarray,
			center: List[int]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""从volume中提取patch"""
		# 计算切片范围
		slices = []
		pad_before = []
		pad_after = []
		
		for i, (c, s, p) in enumerate(zip(center, image.shape[1:], self.patch_size)):
			half = p // 2
			start = c - half
			end = start + p
			
			# 处理边界
			pad_b = max(0, -start)
			pad_a = max(0, end - s)
			start = max(0, start)
			end = min(s, end)
			
			slices.append(slice(start, end))
			pad_before.append(pad_b)
			pad_after.append(pad_a)
		
		# 提取patch
		img_patch = image[:, slices[0], slices[1], slices[2]]
		lbl_patch = label[:, slices[0], slices[1], slices[2]]
		
		# 如果需要padding
		if any(pad_before) or any(pad_after):
			pad_width = [(0, 0)]  # 通道维度不padding
			for pb, pa in zip(pad_before, pad_after):
				pad_width.append((pb, pa))
			
			img_patch = np.pad(img_patch, pad_width, mode='constant', constant_values=0)
			lbl_patch = np.pad(lbl_patch, pad_width, mode='constant', constant_values=0)
		
		return img_patch, lbl_patch
	
	def __getitem__(self, idx: int):
		"""获取一个样本"""
		if self.mode == 'train':
			# 训练模式：计算对应的case和sample索引
			case_idx = idx // self.samples_per_volume
			case_id = self.case_ids[case_idx]
		else:
			# 验证/测试模式
			case_id = self.case_ids[idx]
		
		# 加载数据
		image, label = self._load_case(case_id)
		shape = image.shape[1:]  # (D, H, W)
		
		if self.mode == 'train':
			# 决定是否从前景采样
			if np.random.random() < self.oversample_foreground:
				center = self._get_foreground_center(case_id, shape)
			else:
				center = self._get_random_center(shape)
			
			# 提取patch
			img_patch, lbl_patch = self._extract_patch(image, label, center)
		else:
			# 验证/测试：从中心提取patch
			center = [s // 2 for s in shape]
			img_patch, lbl_patch = self._extract_patch(image, label, center)
		
		# 转换为tensor
		img_tensor = torch.from_numpy(img_patch.copy()).float()
		lbl_tensor = torch.from_numpy(lbl_patch.copy()).float()
		
		if self.mode == 'train':
			return img_tensor, lbl_tensor
		else:
			return img_tensor, lbl_tensor, case_id


class VesselDatasetFullVolume(Dataset):
	"""
	全volume数据集

	用于验证/推理时加载完整volume

	参数:
		data_dir: 数据目录
		props_dir: 属性目录
		split_file: 划分文件
		split: 数据集划分

	返回:
		(image, label, case_id, properties)
	"""
	
	def __init__(
			self,
			data_dir: str,
			props_dir: str,
			split_file: str,
			split: str = 'val'
	):
		super().__init__()
		
		self.data_dir = data_dir
		self.props_dir = props_dir
		
		with open(split_file, 'r') as f:
			splits = json.load(f)
		
		self.case_ids = splits[split]
		
		# 加载属性
		self.properties = {}
		for case_id in self.case_ids:
			prop_path = os.path.join(props_dir, f"{case_id}.pkl")
			if os.path.exists(prop_path):
				with open(prop_path, 'rb') as f:
					self.properties[case_id] = pickle.load(f)
		
		print(f"[{split}] 加载了 {len(self.case_ids)} 个完整volume")
	
	def __len__(self) -> int:
		return len(self.case_ids)
	
	def __getitem__(self, idx: int):
		case_id = self.case_ids[idx]
		
		# 加载数据
		data_path = os.path.join(self.data_dir, f"{case_id}.npy")
		data = np.load(data_path)
		
		image = data[0:1]  # [1, D, H, W]
		label = data[1:2]  # [1, D, H, W]
		
		# 转换为tensor
		img_tensor = torch.from_numpy(image.copy()).float()
		lbl_tensor = torch.from_numpy(label.copy()).float()
		
		# 获取属性
		props = self.properties.get(case_id, {})
		
		return img_tensor, lbl_tensor, case_id, props