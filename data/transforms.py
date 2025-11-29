# data/transforms.py
"""
数据增强
用于训练时的在线数据增强
"""

import numpy as np
from scipy.ndimage import rotate, zoom
from typing import Tuple, Optional, List, Callable
import torch


class Compose:
	"""组合多个变换"""
	
	def __init__(self, transforms: List[Callable]):
		self.transforms = transforms
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		for t in self.transforms:
			image, label = t(image, label)
		return image, label


class RandomFlip:
	"""
	随机翻转

	参数:
		axes: 可翻转的轴列表
		prob: 每个轴的翻转概率
	"""
	
	def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
		self.axes = axes
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		for axis in self.axes:
			if np.random.random() < self.prob:
				image = np.flip(image, axis=axis + 1).copy()  # +1跳过通道维
				if label is not None:
					label = np.flip(label, axis=axis + 1).copy()
		return image, label


class RandomRotate90:
	"""
	随机90度旋转

	在指定平面内随机旋转0/90/180/270度

	参数:
		axes: 旋转平面的两个轴
		prob: 旋转概率
	"""
	
	def __init__(self, axes: Tuple[int, int] = (1, 2), prob: float = 0.5):
		self.axes = (axes[0] + 1, axes[1] + 1)  # +1跳过通道维
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			k = np.random.randint(0, 4)  # 0, 1, 2, 3次90度
			image = np.rot90(image, k, axes=self.axes).copy()
			if label is not None:
				label = np.rot90(label, k, axes=self.axes).copy()
		return image, label


class RandomRotate:
	"""
	随机小角度旋转

	参数:
		angle_range: 旋转角度范围 (min, max)
		axes: 旋转平面
		prob: 旋转概率
	"""
	
	def __init__(
			self,
			angle_range: Tuple[float, float] = (-15, 15),
			axes: Tuple[int, int] = (1, 2),
			prob: float = 0.3
	):
		self.angle_range = angle_range
		self.axes = (axes[0] + 1, axes[1] + 1)
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			angle = np.random.uniform(*self.angle_range)
			image = rotate(
				image, angle, axes=self.axes,
				reshape=False, order=1, mode='constant', cval=0
			)
			if label is not None:
				label = rotate(
					label, angle, axes=self.axes,
					reshape=False, order=0, mode='constant', cval=0
				)
		return image, label


class RandomScale:
	"""
	随机缩放

	参数:
		scale_range: 缩放比例范围
		prob: 缩放概率
	"""
	
	def __init__(
			self,
			scale_range: Tuple[float, float] = (0.9, 1.1),
			prob: float = 0.3
	):
		self.scale_range = scale_range
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			scale = np.random.uniform(*self.scale_range)
			# 只缩放空间维度
			zoom_factors = [1] + [scale] * 3  # [C, D, H, W]
			
			image = zoom(image, zoom_factors, order=1, mode='constant', cval=0)
			if label is not None:
				label = zoom(label, zoom_factors, order=0, mode='constant', cval=0)
		return image, label


class RandomIntensityShift:
	"""
	随机强度偏移

	参数:
		shift_range: 偏移范围
		prob: 概率
	"""
	
	def __init__(
			self,
			shift_range: Tuple[float, float] = (-0.1, 0.1),
			prob: float = 0.5
	):
		self.shift_range = shift_range
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			shift = np.random.uniform(*self.shift_range)
			image = image + shift
		return image, label


class RandomIntensityScale:
	"""
	随机强度缩放

	参数:
		scale_range: 缩放范围
		prob: 概率
	"""
	
	def __init__(
			self,
			scale_range: Tuple[float, float] = (0.9, 1.1),
			prob: float = 0.5
	):
		self.scale_range = scale_range
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			scale = np.random.uniform(*self.scale_range)
			image = image * scale
		return image, label


class RandomGaussianNoise:
	"""
	随机高斯噪声

	参数:
		mean: 噪声均值
		std_range: 标准差范围
		prob: 概率
	"""
	
	def __init__(
			self,
			mean: float = 0.0,
			std_range: Tuple[float, float] = (0.01, 0.05),
			prob: float = 0.3
	):
		self.mean = mean
		self.std_range = std_range
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			std = np.random.uniform(*self.std_range)
			noise = np.random.normal(self.mean, std, image.shape).astype(np.float32)
			image = image + noise
		return image, label


class RandomGaussianBlur:
	"""
	随机高斯模糊

	参数:
		sigma_range: sigma范围
		prob: 概率
	"""
	
	def __init__(
			self,
			sigma_range: Tuple[float, float] = (0.5, 1.5),
			prob: float = 0.2
	):
		self.sigma_range = sigma_range
		self.prob = prob
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		if np.random.random() < self.prob:
			from scipy.ndimage import gaussian_filter
			sigma = np.random.uniform(*self.sigma_range)
			# 只对空间维度模糊
			for c in range(image.shape[0]):
				image[c] = gaussian_filter(image[c], sigma=sigma)
		return image, label


class ClipIntensity:
	"""
	裁剪强度到指定范围

	参数:
		min_val: 最小值
		max_val: 最大值
	"""
	
	def __init__(self, min_val: float = -5.0, max_val: float = 5.0):
		self.min_val = min_val
		self.max_val = max_val
	
	def __call__(
			self,
			image: np.ndarray,
			label: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		image = np.clip(image, self.min_val, self.max_val)
		return image, label


def get_train_transforms() -> Compose:
	"""获取训练数据增强"""
	return Compose([
		RandomFlip(axes=(0, 1, 2), prob=0.5),
		RandomRotate90(axes=(1, 2), prob=0.5),
		RandomRotate(angle_range=(-15, 15), axes=(1, 2), prob=0.3),
		RandomIntensityShift(shift_range=(-0.1, 0.1), prob=0.5),
		RandomIntensityScale(scale_range=(0.9, 1.1), prob=0.5),
		RandomGaussianNoise(std_range=(0.01, 0.03), prob=0.3),
		ClipIntensity(min_val=-5.0, max_val=5.0),
	])


def get_val_transforms() -> Compose:
	"""获取验证数据变换（只做必要的标准化）"""
	return Compose([
		ClipIntensity(min_val=-5.0, max_val=5.0),
	])