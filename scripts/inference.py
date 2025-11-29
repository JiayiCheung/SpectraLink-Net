# scripts/inference.py
"""
SpectraLink-Net 推理脚本
对新数据进行推理，输出分割结果
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import numpy as np
import nibabel as nib
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SpectraLinkNet


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='SpectraLink-Net Inference')
	parser.add_argument('--input', type=str, required=True,
	                    help='输入文件或目录')
	parser.add_argument('--output', type=str, required=True,
	                    help='输出目录')
	parser.add_argument('--checkpoint', type=str, required=True,
	                    help='模型检查点路径')
	parser.add_argument('--config', type=str, default='config/config.yaml',
	                    help='配置文件路径')
	parser.add_argument('--threshold', type=float, default=0.5,
	                    help='二值化阈值')
	parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 128],
	                    help='推理patch大小')
	parser.add_argument('--overlap', type=float, default=0.5,
	                    help='滑动窗口重叠比例')
	parser.add_argument('--save_prob', action='store_true',
	                    help='是否保存概率图')
	return parser.parse_args()


def load_config(config_path: str) -> dict:
	"""加载配置文件"""
	if os.path.exists(config_path):
		with open(config_path, 'r', encoding='utf-8') as f:
			config = yaml.safe_load(f)
		return config
	return {}


def preprocess(
		image: np.ndarray,
		clip_range: tuple = (-200, 300)
) -> np.ndarray:
	"""
	预处理图像

	参数:
		image: 原始CT图像
		clip_range: HU值裁剪范围

	返回:
		预处理后的图像
	"""
	# 裁剪HU值
	image = np.clip(image, clip_range[0], clip_range[1])
	
	# Z-score归一化
	mean = np.mean(image)
	std = np.std(image)
	image = (image - mean) / (std + 1e-8)
	
	return image.astype(np.float32)


def sliding_window_inference(
		model: torch.nn.Module,
		image: torch.Tensor,
		window_size: tuple,
		overlap: float = 0.5,
		device: torch.device = None
) -> torch.Tensor:
	"""滑动窗口推理"""
	if device is None:
		device = next(model.parameters()).device
	
	_, C, D, H, W = image.shape
	wd, wh, ww = window_size
	
	# 计算步长
	sd = max(1, int(wd * (1 - overlap)))
	sh = max(1, int(wh * (1 - overlap)))
	sw = max(1, int(ww * (1 - overlap)))
	
	# 初始化
	output = torch.zeros(1, 1, D, H, W, device=device)
	count = torch.zeros(1, 1, D, H, W, device=device)
	
	# 高斯权重
	gaussian = _create_gaussian_weight(window_size, device)
	
	model.eval()
	with torch.no_grad():
		# 计算总步数用于进度条
		d_steps = list(range(0, max(1, D - wd + 1), sd))
		h_steps = list(range(0, max(1, H - wh + 1), sh))
		w_steps = list(range(0, max(1, W - ww + 1), sw))
		total_steps = len(d_steps) * len(h_steps) * len(w_steps)
		
		with tqdm(total=total_steps, desc='Inference', leave=False) as pbar:
			for d in d_steps:
				for h in h_steps:
					for w in w_steps:
						# 边界处理
						d_end = min(d + wd, D)
						h_end = min(h + wh, H)
						w_end = min(w + ww, W)
						d_start = d_end - wd
						h_start = h_end - wh
						w_start = w_end - ww
						
						# 提取patch
						patch = image[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
						patch = patch.to(device)
						
						# 推理
						pred = model(patch)
						pred = torch.sigmoid(pred)
						
						# 累加
						output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += pred * gaussian
						count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += gaussian
						
						pbar.update(1)
	
	# 平均
	output = output / (count + 1e-8)
	
	return output


def _create_gaussian_weight(window_size: tuple, device: torch.device) -> torch.Tensor:
	"""创建高斯权重"""
	d, h, w = window_size
	
	zz = torch.linspace(-1, 1, d, device=device)
	yy = torch.linspace(-1, 1, h, device=device)
	xx = torch.linspace(-1, 1, w, device=device)
	
	zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')
	
	sigma = 0.5
	gaussian = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * sigma ** 2))
	gaussian = gaussian / gaussian.max()
	gaussian = gaussian.unsqueeze(0).unsqueeze(0)
	
	return gaussian


def process_single_file(
		input_path: str,
		output_dir: str,
		model: torch.nn.Module,
		device: torch.device,
		patch_size: tuple,
		overlap: float,
		threshold: float,
		save_prob: bool
):
	"""
	处理单个文件

	参数:
		input_path: 输入文件路径
		output_dir: 输出目录
		model: 模型
		device: 设备
		patch_size: patch大小
		overlap: 重叠比例
		threshold: 二值化阈值
		save_prob: 是否保存概率图
	"""
	filename = os.path.basename(input_path)
	case_id = filename.replace('.nii.gz', '').replace('.nii', '')
	
	print(f"Processing: {case_id}")
	
	# 加载图像
	nii = nib.load(input_path)
	image = nii.get_fdata().astype(np.float32)
	affine = nii.affine
	header = nii.header
	
	# 预处理
	image_processed = preprocess(image)
	
	# 转换为tensor
	image_tensor = torch.from_numpy(image_processed).unsqueeze(0).unsqueeze(0)
	
	# 推理
	pred_prob = sliding_window_inference(
		model, image_tensor, patch_size, overlap, device
	)
	
	# 转换为numpy
	pred_prob_np = pred_prob.squeeze().cpu().numpy()
	
	# 二值化
	pred_binary = (pred_prob_np > threshold).astype(np.uint8)
	
	# 保存结果
	# 二值分割
	seg_nii = nib.Nifti1Image(pred_binary, affine, header)
	seg_path = os.path.join(output_dir, f'{case_id}_seg.nii.gz')
	nib.save(seg_nii, seg_path)
	
	# 概率图（可选）
	if save_prob:
		prob_nii = nib.Nifti1Image(pred_prob_np.astype(np.float32), affine, header)
		prob_path = os.path.join(output_dir, f'{case_id}_prob.nii.gz')
		nib.save(prob_nii, prob_path)
	
	print(f"  Saved: {seg_path}")
	if save_prob:
		print(f"  Saved: {prob_path}")


def main():
	"""主函数"""
	args = parse_args()
	
	# 设置设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	# 创建输出目录
	os.makedirs(args.output, exist_ok=True)
	
	# 加载配置
	config = load_config(args.config)
	
	# 创建模型
	model_config = config.get('model', {})
	model = SpectraLinkNet(
		in_channels=model_config.get('in_channels', 1),
		out_channels=model_config.get('out_channels', 1),
		base_channels=model_config.get('base_channels', 32),
		num_levels=model_config.get('num_levels', 3)
	)
	
	# 加载检查点
	checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	print(f"Loaded checkpoint: {args.checkpoint}")
	
	# 获取输入文件列表
	if os.path.isfile(args.input):
		input_files = [args.input]
	else:
		input_files = sorted([
			os.path.join(args.input, f)
			for f in os.listdir(args.input)
			if f.endswith('.nii.gz') or f.endswith('.nii')
		])
	
	print(f"Found {len(input_files)} files to process")
	
	# 处理每个文件
	patch_size = tuple(args.patch_size)
	
	for input_path in input_files:
		process_single_file(
			input_path=input_path,
			output_dir=args.output,
			model=model,
			device=device,
			patch_size=patch_size,
			overlap=args.overlap,
			threshold=args.threshold,
			save_prob=args.save_prob
		)
	
	print(f"\nDone! Results saved to: {args.output}")


if __name__ == '__main__':
	main()