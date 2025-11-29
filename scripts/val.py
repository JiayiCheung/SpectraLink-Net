# scripts/val.py
"""
SpectraLink-Net 验证脚本
对验证集进行完整评估，支持滑动窗口推理
"""

import os
import sys
import argparse
import yaml
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SpectraLinkNet
from data import get_test_dataloader
from utils import calculate_all_metrics, MetricTracker


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='SpectraLink-Net Validation')
	parser.add_argument('--config', type=str, default='config/config.yaml',
	                    help='配置文件路径')
	parser.add_argument('--checkpoint', type=str, required=True,
	                    help='模型检查点路径')
	parser.add_argument('--output_dir', type=str, default='outputs/validation',
	                    help='结果输出目录')
	parser.add_argument('--split', type=str, default='val',
	                    help='数据集划分 (val 或 test)')
	parser.add_argument('--save_predictions', action='store_true',
	                    help='是否保存预测结果')
	return parser.parse_args()


def load_config(config_path: str) -> dict:
	"""加载配置文件"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config


def sliding_window_inference(
		model: torch.nn.Module,
		image: torch.Tensor,
		window_size: tuple,
		overlap: float = 0.5,
		device: torch.device = None
) -> torch.Tensor:
	"""
	滑动窗口推理

	参数:
		model: 模型
		image: 输入图像 [1, C, D, H, W]
		window_size: 窗口大小 (D, H, W)
		overlap: 重叠比例
		device: 设备

	返回:
		预测结果 [1, 1, D, H, W]
	"""
	if device is None:
		device = next(model.parameters()).device
	
	_, C, D, H, W = image.shape
	wd, wh, ww = window_size
	
	# 计算步长
	sd = max(1, int(wd * (1 - overlap)))
	sh = max(1, int(wh * (1 - overlap)))
	sw = max(1, int(ww * (1 - overlap)))
	
	# 初始化输出和计数
	output = torch.zeros(1, 1, D, H, W, device=device)
	count = torch.zeros(1, 1, D, H, W, device=device)
	
	# 创建高斯权重
	gaussian = _create_gaussian_weight(window_size, device)
	
	# 滑动窗口
	model.eval()
	with torch.no_grad():
		for d in range(0, max(1, D - wd + 1), sd):
			for h in range(0, max(1, H - wh + 1), sh):
				for w in range(0, max(1, W - ww + 1), sw):
					# 处理边界
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
					
					# 加权累加
					output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += pred * gaussian
					count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += gaussian
	
	# 平均
	output = output / (count + 1e-8)
	
	return output


def _create_gaussian_weight(window_size: tuple, device: torch.device) -> torch.Tensor:
	"""创建高斯权重"""
	d, h, w = window_size
	
	# 创建坐标
	zz = torch.linspace(-1, 1, d, device=device)
	yy = torch.linspace(-1, 1, h, device=device)
	xx = torch.linspace(-1, 1, w, device=device)
	
	zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')
	
	# 高斯权重
	sigma = 0.5
	gaussian = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * sigma ** 2))
	
	# 归一化
	gaussian = gaussian / gaussian.max()
	
	# 添加batch和channel维度
	gaussian = gaussian.unsqueeze(0).unsqueeze(0)
	
	return gaussian


def evaluate(
		model: torch.nn.Module,
		dataloader,
		config: dict,
		device: torch.device,
		output_dir: str,
		save_predictions: bool = False
) -> dict:
	"""
	评估模型

	参数:
		model: 模型
		dataloader: 数据加载器
		config: 配置
		device: 设备
		output_dir: 输出目录
		save_predictions: 是否保存预测

	返回:
		平均指标
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# 获取窗口大小
	patch_size = tuple(config.get('data', {}).get('patch_size', [64, 128, 128]))
	
	# 准备CSV记录
	csv_path = os.path.join(output_dir, 'results.csv')
	csv_file = open(csv_path, 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow([
		'case_id', 'dice', 'iou', 'precision', 'recall',
		'hd95', 'asd', 'fnr', 'over_rate', 'under_rate'
	])
	
	# 指标追踪
	metric_tracker = MetricTracker()
	
	model.eval()
	
	for image, label, case_id, props in tqdm(dataloader, desc='Evaluating'):
		case_id = case_id[0]  # 解包
		
		# 滑动窗口推理
		pred = sliding_window_inference(
			model, image, patch_size, overlap=0.5, device=device
		)
		
		# 移到CPU计算指标
		pred_cpu = pred.cpu()
		label_cpu = label.cpu()
		
		# 计算指标
		metrics = calculate_all_metrics(
			pred_cpu, label_cpu,
			threshold=0.5,
			include_surface=True
		)
		
		# 记录
		metric_tracker.update(metrics)
		
		csv_writer.writerow([
			case_id,
			metrics['dice'],
			metrics['iou'],
			metrics['precision'],
			metrics['recall'],
			metrics.get('hd95', float('inf')),
			metrics.get('asd', float('inf')),
			metrics['fnr'],
			metrics['over_rate'],
			metrics['under_rate']
		])
		
		print(f"{case_id}: Dice={metrics['dice']:.4f}, HD95={metrics.get('hd95', 0):.2f}")
		
		# 保存预测
		if save_predictions:
			pred_dir = os.path.join(output_dir, 'predictions')
			os.makedirs(pred_dir, exist_ok=True)
			
			pred_np = (pred_cpu.squeeze().numpy() > 0.5).astype(np.uint8)
			np.save(os.path.join(pred_dir, f'{case_id}_pred.npy'), pred_np)
	
	csv_file.close()
	
	# 计算平均指标
	avg_metrics = metric_tracker.get_average()
	
	# 保存汇总
	summary_path = os.path.join(output_dir, 'summary.txt')
	with open(summary_path, 'w') as f:
		f.write("=" * 50 + "\n")
		f.write("Validation Summary\n")
		f.write("=" * 50 + "\n")
		for name, value in avg_metrics.items():
			f.write(f"{name}: {value:.4f}\n")
		f.write("=" * 50 + "\n")
	
	print("\n" + "=" * 50)
	print("Average Metrics:")
	for name, value in avg_metrics.items():
		print(f"  {name}: {value:.4f}")
	print("=" * 50)
	
	return avg_metrics


def main():
	"""主函数"""
	args = parse_args()
	
	# 设置设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
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
	print(f"Loaded checkpoint from: {args.checkpoint}")
	
	if 'epoch' in checkpoint:
		print(f"Checkpoint epoch: {checkpoint['epoch']}")
	if 'metrics' in checkpoint and checkpoint['metrics']:
		print(f"Checkpoint metrics: {checkpoint['metrics']}")
	
	# 创建数据加载器
	dataloader = get_test_dataloader(config, split=args.split)
	print(f"Loaded {len(dataloader)} samples from {args.split} set")
	
	# 评估
	avg_metrics = evaluate(
		model, dataloader, config, device,
		args.output_dir, args.save_predictions
	)
	
	print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
	main()