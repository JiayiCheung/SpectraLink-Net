#!/usr/bin/env python3
"""
最佳切片可视化脚本（修复版 - 统一输出尺寸）

两种风格：
1. Overlay: CT背景 + TP(绿)/FP(红)/FN(蓝) 叠加
2. Contour: CT | CT+GT轮廓(绿) | CT+Pred轮廓(红)

直接 python visualize_slices_best.py 运行
"""

import os
import sys
from pathlib import Path
import random

# ============================================================
# 配置
# ============================================================

CONFIG_PATH = "config/config.yaml"
CHECKPOINT_PATH = "outputs/checkpoints/best.pth"
OUTPUT_DIR = "outputs/viz_slices_best"
SPLIT = "val"

# 固定切片尺寸，保证所有case输出一致
TARGET_SLICE_SIZE = (256, 256)  # (H, W)

# 固定每个case的切片索引，保证不同模型对比时一致
# combined和contour用不同组切片，展示更多内容

CASE_SLICES_COMBINED = {
	'hepaticvessel_442': [25, 50, 75, 100, 125, 150],
	'hepaticvessel_429': [20, 45, 70, 95, 120, 145],
	'hepaticvessel_406': [15, 40, 65, 90, 115, 140],
	'hepaticvessel_404': [20, 45, 70, 95, 120, 145],
	'hepaticvessel_371': [25, 50, 75, 100, 125, 150],
}

CASE_SLICES_CONTOUR = {
	'hepaticvessel_442': [35, 60, 85, 110, 135, 160],
	'hepaticvessel_429': [30, 55, 80, 105, 130, 155],
	'hepaticvessel_406': [25, 50, 75, 100, 125, 150],
	'hepaticvessel_404': [30, 55, 80, 105, 130, 155],
	'hepaticvessel_371': [35, 60, 85, 110, 135, 160],
}

TARGET_CASES = list(CASE_SLICES_COMBINED.keys())

# ============================================================
# 导入
# ============================================================

import yaml
from tqdm import tqdm

import torch
import numpy as np
from skimage.transform import resize

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import SpectraLinkNet
from data import get_test_dataloader


# ============================================================
# 切片尺寸统一
# ============================================================

def resize_slice(slice_2d, target_size=TARGET_SLICE_SIZE, is_mask=False):
	"""将切片resize到目标尺寸"""
	if is_mask:
		# mask用最近邻插值保持二值
		return resize(slice_2d, target_size, order=0, preserve_range=True, anti_aliasing=False)
	else:
		# CT用双线性插值
		return resize(slice_2d, target_size, order=1, preserve_range=True, anti_aliasing=True)


# ============================================================
# 滑动窗口推理
# ============================================================

def _create_gaussian_weight(window_size, device):
	d, h, w = window_size
	zz = torch.linspace(-1, 1, d, device=device)
	yy = torch.linspace(-1, 1, h, device=device)
	xx = torch.linspace(-1, 1, w, device=device)
	zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')
	gaussian = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * 0.5 ** 2))
	gaussian = (gaussian / gaussian.max()).unsqueeze(0).unsqueeze(0)
	return gaussian


def sliding_window_inference(model, image, window_size, overlap=0.5, device=None):
	if device is None:
		device = next(model.parameters()).device
	
	_, C, D, H, W = image.shape
	wd, wh, ww = window_size
	
	sd = max(1, int(wd * (1 - overlap)))
	sh = max(1, int(wh * (1 - overlap)))
	sw = max(1, int(ww * (1 - overlap)))
	
	output = torch.zeros(1, 1, D, H, W, device=device)
	count = torch.zeros(1, 1, D, H, W, device=device)
	gaussian = _create_gaussian_weight(window_size, device)
	
	model.eval()
	with torch.no_grad():
		for d in range(0, max(1, D - wd + 1), sd):
			for h in range(0, max(1, H - wh + 1), sh):
				for w in range(0, max(1, W - ww + 1), sw):
					d_end = min(d + wd, D)
					h_end = min(h + wh, H)
					w_end = min(w + ww, W)
					d_start, h_start, w_start = d_end - wd, h_end - wh, w_end - ww
					
					patch = image[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)
					pred = torch.sigmoid(model(patch))
					
					output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += pred * gaussian
					count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += gaussian
	
	return (output / (count + 1e-8))


# ============================================================
# 切片选择
# ============================================================

def select_slices(gt_volume, num_slices=6, min_pixels=50):
	"""选择有内容的切片，随机采样"""
	# 找出有足够前景的切片
	z_sums = gt_volume.sum(axis=(1, 2))
	valid_z = np.where(z_sums > min_pixels)[0]
	
	if len(valid_z) == 0:
		# 没有前景，均匀采样
		return np.linspace(0, gt_volume.shape[0] - 1, num_slices).astype(int)
	
	if len(valid_z) <= num_slices:
		return valid_z
	
	# 随机采样，但保持一定间距
	random.seed(42)  # 固定随机种子保证可复现
	
	# 分段采样，每段随机取一个
	segment_size = len(valid_z) // num_slices
	selected = []
	for i in range(num_slices):
		start = i * segment_size
		end = (i + 1) * segment_size if i < num_slices - 1 else len(valid_z)
		idx = random.randint(start, end - 1)
		selected.append(valid_z[idx])
	
	return np.array(selected)


# ============================================================
# 风格1: Overlay (TP绿/FP红/FN蓝)
# ============================================================

def create_overlay_slice(ct_slice, gt_slice, pred_slice):
	"""创建单个切片的Overlay图"""
	# 归一化CT
	ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
	
	# 二值化
	gt_bin = (gt_slice > 0.5).astype(float)
	pred_bin = (pred_slice > 0.5).astype(float)
	
	# 计算TP/FP/FN
	tp = gt_bin * pred_bin
	fp = pred_bin * (1 - gt_bin)
	fn = gt_bin * (1 - pred_bin)
	
	# 创建RGB图像
	h, w = ct_slice.shape
	overlay = np.zeros((h, w, 3))
	
	# CT作为背景（稍微调暗）
	overlay[..., 0] = ct_norm * 0.4
	overlay[..., 1] = ct_norm * 0.4
	overlay[..., 2] = ct_norm * 0.4
	
	# 叠加分割结果
	overlay[..., 1] = np.where(tp > 0, 0.9, overlay[..., 1])  # 绿色 = TP
	overlay[..., 0] = np.where(fp > 0, 0.9, overlay[..., 0])  # 红色 = FP
	overlay[..., 2] = np.where(fn > 0, 0.9, overlay[..., 2])  # 蓝色 = FN
	
	return np.clip(overlay, 0, 1)


def visualize_overlay(image_np, gt_np, pred_np, case_id, output_dir, selected_z):
	"""风格1: Overlay可视化"""
	num_slices = len(selected_z)
	
	fig, axes = plt.subplots(num_slices, 1, figsize=(4, 3 * num_slices))
	fig.patch.set_facecolor('#1a1a1a')
	
	if num_slices == 1:
		axes = [axes]
	
	for row, z in enumerate(selected_z):
		# 提取并resize切片
		ct_slice = resize_slice(image_np[z], is_mask=False)
		gt_slice = resize_slice(gt_np[z], is_mask=True)
		pred_slice = resize_slice(pred_np[z], is_mask=True)
		
		overlay = create_overlay_slice(ct_slice, gt_slice, pred_slice)
		
		axes[row].imshow(overlay, aspect='equal')
		axes[row].axis('off')
	
	# 标题
	fig.text(0.5, 0.99, 'Overlay', ha='center', va='top',
	         color='white', fontsize=14, fontweight='bold')
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.96, hspace=0.02)
	
	output_path = output_dir / f'{case_id}_overlay.png'
	plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
	plt.close()
	
	return output_path


# ============================================================
# 风格2: Contour (轮廓线)
# ============================================================

def extract_contour(mask, thickness=2):
	"""提取轮廓"""
	mask_bin = (mask > 0.5).astype(np.uint8)
	
	# 多次腐蚀得到更粗的轮廓
	eroded = mask_bin.copy()
	for _ in range(thickness):
		eroded = binary_erosion(eroded).astype(np.uint8)
	
	contour = mask_bin - eroded
	return contour


def visualize_contour(image_np, gt_np, pred_np, case_id, output_dir, selected_z):
	"""风格2: Contour可视化"""
	num_slices = len(selected_z)
	
	fig, axes = plt.subplots(num_slices, 3, figsize=(12, 3 * num_slices))
	fig.patch.set_facecolor('#1a1a1a')
	
	if num_slices == 1:
		axes = axes.reshape(1, -1)
	
	for row, z in enumerate(selected_z):
		# 提取并resize切片
		ct_slice = resize_slice(image_np[z], is_mask=False)
		gt_slice = resize_slice(gt_np[z], is_mask=True)
		pred_slice = resize_slice(pred_np[z], is_mask=True)
		
		# 归一化CT
		ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
		
		# 提取轮廓
		gt_contour = extract_contour(gt_slice, thickness=2)
		pred_contour = extract_contour(pred_slice, thickness=2)
		
		# 列1: 纯CT
		axes[row, 0].imshow(ct_norm, cmap='gray', aspect='equal')
		axes[row, 0].axis('off')
		
		# 列2: CT + GT轮廓(绿色)
		overlay_gt = np.stack([ct_norm] * 3, axis=-1)
		overlay_gt[gt_contour > 0] = [0, 1, 0]  # 绿色
		axes[row, 1].imshow(overlay_gt, aspect='equal')
		axes[row, 1].axis('off')
		
		# 列3: CT + Pred轮廓(红色)
		overlay_pred = np.stack([ct_norm] * 3, axis=-1)
		overlay_pred[pred_contour > 0] = [1, 0, 0]  # 红色
		axes[row, 2].imshow(overlay_pred, aspect='equal')
		axes[row, 2].axis('off')
	
	# 列标题
	fig.text(0.18, 0.99, 'CT', ha='center', va='top',
	         color='white', fontsize=12, fontweight='bold')
	fig.text(0.50, 0.99, 'CT + GT (green)', ha='center', va='top',
	         color='white', fontsize=12, fontweight='bold')
	fig.text(0.82, 0.99, 'CT + Pred (red)', ha='center', va='top',
	         color='white', fontsize=12, fontweight='bold')
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.96, hspace=0.02, wspace=0.02)
	
	output_path = output_dir / f'{case_id}_contour.png'
	plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
	plt.close()
	
	return output_path


# ============================================================
# 合并：CT | Overlay（与baseline统一格式）
# ============================================================

def visualize_combined(image_np, gt_np, pred_np, case_id, output_dir, selected_z):
	"""合并可视化：CT | Overlay（与baseline统一格式）"""
	num_slices = len(selected_z)
	
	# 2列: CT | Overlay（和MedNeXt一致）
	fig, axes = plt.subplots(num_slices, 2, figsize=(8, 3 * num_slices))
	fig.patch.set_facecolor('#1a1a1a')
	
	if num_slices == 1:
		axes = axes.reshape(1, -1)
	
	for row, z in enumerate(selected_z):
		# 提取并resize切片到统一尺寸
		ct_slice = resize_slice(image_np[z], is_mask=False)
		gt_slice = resize_slice(gt_np[z], is_mask=True)
		pred_slice = resize_slice(pred_np[z], is_mask=True)
		
		# 归一化CT
		ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
		
		# 列1: 纯CT
		axes[row, 0].imshow(ct_norm, cmap='gray', aspect='equal')
		axes[row, 0].axis('off')
		
		# 列2: Overlay
		overlay = create_overlay_slice(ct_slice, gt_slice, pred_slice)
		axes[row, 1].imshow(overlay, aspect='equal')
		axes[row, 1].axis('off')
	
	# 列标题: CT | Ours
	fig.text(0.28, 0.99, 'CT', ha='center', va='top',
	         color='white', fontsize=12, fontweight='bold')
	fig.text(0.72, 0.99, 'Ours', ha='center', va='top',
	         color='white', fontsize=12, fontweight='bold')
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.96, hspace=0.02, wspace=0.02)
	
	output_path = output_dir / f'{case_id}_combined.png'
	plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
	plt.close()
	
	return output_path


# ============================================================
# 计算Dice
# ============================================================

def compute_dice(pred, gt):
	pred_bin = (pred > 0.5).astype(float)
	gt_bin = (gt > 0.5).astype(float)
	intersection = np.sum(pred_bin * gt_bin)
	union = np.sum(pred_bin) + np.sum(gt_bin)
	return (2 * intersection + 1e-5) / (union + 1e-5)


# ============================================================
# 主函数
# ============================================================

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")
	
	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	with open(CONFIG_PATH, 'r') as f:
		config = yaml.safe_load(f)
	
	model_config = config.get('model', {})
	model = SpectraLinkNet(
		in_channels=model_config.get('in_channels', 1),
		out_channels=model_config.get('out_channels', 1),
		base_channels=model_config.get('base_channels', 32),
		num_levels=model_config.get('num_levels', 3)
	)
	
	checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	print(f"模型: {CHECKPOINT_PATH}")
	
	patch_size = tuple(config.get('data', {}).get('patch_size', [64, 128, 128]))
	dataloader = get_test_dataloader(config, split=SPLIT)
	
	print("=" * 60)
	print("切片可视化 - Overlay & Contour")
	print("=" * 60)
	print(f"目标case: {TARGET_CASES}")
	print(f"统一切片尺寸: {TARGET_SLICE_SIZE}")
	print(f"Combined切片组: 6个/case")
	print(f"Contour切片组:  6个/case (不同切片)")
	print("=" * 60)
	
	results = []
	
	for image, label, case_id, props in tqdm(dataloader, desc='Processing'):
		case_id = case_id[0]
		
		if case_id not in TARGET_CASES:
			continue
		
		print(f"\n{'=' * 40}")
		print(f"处理: {case_id}")
		print(f"{'=' * 40}")
		
		# 推理
		pred = sliding_window_inference(model, image, patch_size, overlap=0.5, device=device)
		pred_np = pred.cpu().squeeze().numpy()
		gt_np = label.squeeze().numpy()
		image_np = image.squeeze().numpy()
		
		# 计算Dice
		dice = compute_dice(pred_np, gt_np)
		print(f"  Dice: {dice:.4f}")
		
		# 使用固定的切片索引（combined和contour用不同组）
		slices_combined = np.array(CASE_SLICES_COMBINED[case_id])
		slices_contour = np.array(CASE_SLICES_CONTOUR[case_id])
		
		# 裁剪超出范围的切片
		max_z = gt_np.shape[0] - 1
		slices_combined = slices_combined[slices_combined <= max_z]
		slices_contour = slices_contour[slices_contour <= max_z]
		print(f"  Combined切片: {slices_combined.tolist()}")
		print(f"  Contour切片:  {slices_contour.tolist()}")
		print(f"  (max_z={max_z})")
		
		# 生成可视化
		print("  生成 Overlay (用Combined切片)...")
		path1 = visualize_overlay(image_np, gt_np, pred_np, case_id, output_dir, slices_combined)
		
		print("  生成 Contour...")
		path2 = visualize_contour(image_np, gt_np, pred_np, case_id, output_dir, slices_contour)
		
		print("  生成 Combined...")
		path3 = visualize_combined(image_np, gt_np, pred_np, case_id, output_dir, slices_combined)
		
		results.append({
			'case_id': case_id,
			'dice': dice,
			'slices_combined': slices_combined.tolist(),
			'slices_contour': slices_contour.tolist(),
			'overlay': str(path1),
			'contour': str(path2),
			'combined': str(path3),
		})
		
		print(f"  ✓ 完成")
	
	# 保存结果摘要
	print(f"\n{'=' * 60}")
	print("结果摘要")
	print("=" * 60)
	for r in results:
		print(f"  {r['case_id']}: Dice={r['dice']:.4f}")
	
	print(f"\n输出目录: {output_dir}")
	print("=" * 60)


if __name__ == "__main__":
	main()