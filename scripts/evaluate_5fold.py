# scripts/evaluate_5fold.py
"""
五折交叉验证评估汇总脚本
计算Mean±Std，进行统计检验
"""

import os
import sys
import argparse
import yaml
import json
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from scipy.stats import wilcoxon, ttest_rel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SpectraLinkNet
from data import get_test_dataloader
from utils import calculate_all_metrics, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description='5-Fold Evaluation Summary')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--split_file', type=str, required=True,
                        help='五折划分文件路径')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='包含fold_0到fold_4的检查点目录')
    parser.add_argument('--output', type=str, default='5fold_results.csv',
                        help='结果输出文件')
    parser.add_argument('--method_name', type=str, default='SpectraLink-Net',
                        help='方法名称')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
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
    """滑动窗口推理"""
    if device is None:
        device = next(model.parameters()).device
    
    _, C, D, H, W = image.shape
    wd, wh, ww = window_size
    
    sd = max(1, int(wd * (1 - overlap)))
    sh = max(1, int(wh * (1 - overlap)))
    sw = max(1, int(ww * (1 - overlap)))
    
    output = torch.zeros(1, 1, D, H, W, device=device)
    count = torch.zeros(1, 1, D, H, W, device=device)
    
    for d in range(0, max(1, D - wd + 1), sd):
        for h in range(0, max(1, H - wh + 1), sh):
            for w in range(0, max(1, W - ww + 1), sw):
                d_end = min(d + wd, D)
                h_end = min(h + wh, H)
                w_end = min(w + ww, W)
                
                d_start = d_end - wd
                h_start = h_end - wh
                w_start = w_end - ww
                
                patch = image[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                patch = patch.to(device)
                
                with torch.no_grad():
                    pred = model(patch)
                    pred = torch.sigmoid(pred)
                
                output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += pred
                count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
    
    output = output / (count + 1e-8)
    return output


def evaluate_fold(
        model: torch.nn.Module,
        dataloader,
        config: dict,
        device: torch.device
) -> dict:
    """评估单个fold，返回每个样本的指标"""
    model.eval()
    
    patch_size = tuple(config.get('data', {}).get('patch_size', [128, 128, 128]))
    
    all_metrics = []
    
    for image, label, case_id, props in tqdm(dataloader, desc="Evaluating"):
        pred = sliding_window_inference(
            model, image, patch_size, overlap=0.5, device=device
        )
        
        # 计算指标
        metrics = calculate_all_metrics(
            pred.cpu(), label,
            include_surface=True,
            include_cldice=True
        )
        metrics['case_id'] = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
        all_metrics.append(metrics)
    
    return all_metrics


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = load_config(args.config)
    
    # 加载五折划分
    with open(args.split_file, 'r') as f:
        folds = json.load(f)
    
    # 收集所有fold的结果
    all_fold_results = []
    fold_summary = []
    
    for fold_idx in range(5):
        print(f"\n{'='*60}")
        print(f"Evaluating Fold {fold_idx}")
        print(f"{'='*60}")
        
        # 加载模型
        model_config = config.get('model', {})
        model = SpectraLinkNet(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_channels=model_config.get('base_channels', 32),
            num_levels=model_config.get('num_levels', 3)
        )
        
        # 加载检查点
        ckpt_path = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}', 'checkpoints', 'best_model.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}', 'checkpoints', 'latest_model.pth')
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found for fold {fold_idx}, skipping...")
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Loaded checkpoint: {ckpt_path}")
        
        # 创建验证数据加载器
        fold_key = f"fold_{fold_idx}"
        temp_split = {
            'train': folds[fold_key]['train'],
            'val': folds[fold_key]['val']
        }
        temp_split_path = f"/tmp/temp_split_fold{fold_idx}_eval.json"
        with open(temp_split_path, 'w') as f:
            json.dump(temp_split, f)
        
        config_copy = config.copy()
        config_copy['data'] = config['data'].copy()
        config_copy['data']['split_file'] = temp_split_path
        
        dataloader = get_test_dataloader(config_copy, split='val')
        
        # 评估
        fold_metrics = evaluate_fold(model, dataloader, config, device)
        all_fold_results.extend(fold_metrics)
        
        # 计算fold平均
        tracker = MetricTracker()
        for m in fold_metrics:
            tracker.update({k: v for k, v in m.items() if k != 'case_id'})
        
        avg = tracker.get_average()
        fold_summary.append({
            'fold': fold_idx,
            **avg
        })
        
        print(f"Fold {fold_idx} Results:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
    
    # 汇总所有fold
    print(f"\n{'='*60}")
    print("5-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    
    # 计算总体Mean±Std
    metric_names = ['dice', 'hd95', 'asd', 'sens', 'cldice']
    summary = {}
    
    for metric in metric_names:
        values = [r.get(metric, 0) for r in all_fold_results if metric in r]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            summary[metric] = {'mean': mean, 'std': std}
            print(f"{metric}: {mean:.4f} ± {std:.4f}")
    
    # 保存结果
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存CSV（每个fold）
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Fold', 'Dice', 'HD95', 'ASD', 'Sens', 'clDice'])
        for fold_result in fold_summary:
            writer.writerow([
                args.method_name,
                fold_result['fold'],
                f"{fold_result.get('dice', 0):.4f}",
                f"{fold_result.get('hd95', 0):.2f}",
                f"{fold_result.get('asd', 0):.2f}",
                f"{fold_result.get('sens', 0):.4f}",
                f"{fold_result.get('cldice', 0):.4f}"
            ])
        # 写入汇总行
        writer.writerow([
            args.method_name,
            'Mean±Std',
            f"{summary['dice']['mean']:.4f}±{summary['dice']['std']:.4f}",
            f"{summary['hd95']['mean']:.2f}±{summary['hd95']['std']:.2f}",
            f"{summary['asd']['mean']:.2f}±{summary['asd']['std']:.2f}",
            f"{summary['sens']['mean']:.4f}±{summary['sens']['std']:.4f}",
            f"{summary['cldice']['mean']:.4f}±{summary['cldice']['std']:.4f}"
        ])
    
    print(f"\nResults saved to: {args.output}")
    
    # 保存详细JSON
    json_output = args.output.replace('.csv', '_detailed.json')
    with open(json_output, 'w') as f:
        json.dump({
            'method': args.method_name,
            'fold_summary': fold_summary,
            'overall_summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in summary.items()},
            'all_samples': all_fold_results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {json_output}")


def compare_methods(results_file1: str, results_file2: str, metric: str = 'dice'):
    """
    比较两个方法，进行Wilcoxon符号秩检验
    
    用法:
        python evaluate_5fold.py --compare method1.json method2.json --metric dice
    """
    with open(results_file1, 'r') as f:
        results1 = json.load(f)
    with open(results_file2, 'r') as f:
        results2 = json.load(f)
    
    # 提取对应样本的指标值
    values1 = {r['case_id']: r[metric] for r in results1['all_samples']}
    values2 = {r['case_id']: r[metric] for r in results2['all_samples']}
    
    # 找到共同的样本
    common_cases = set(values1.keys()) & set(values2.keys())
    
    v1 = [values1[c] for c in sorted(common_cases)]
    v2 = [values2[c] for c in sorted(common_cases)]
    
    # Wilcoxon符号秩检验
    stat, p_value = wilcoxon(v1, v2)
    
    # t检验
    t_stat, t_p_value = ttest_rel(v1, v2)
    
    print(f"\nStatistical Comparison: {results1['method']} vs {results2['method']}")
    print(f"Metric: {metric}")
    print(f"Number of samples: {len(common_cases)}")
    print(f"\n{results1['method']}: {np.mean(v1):.4f} ± {np.std(v1):.4f}")
    print(f"{results2['method']}: {np.mean(v2):.4f} ± {np.std(v2):.4f}")
    print(f"\nWilcoxon signed-rank test: p = {p_value:.6f}")
    print(f"Paired t-test: p = {t_p_value:.6f}")
    
    if p_value < 0.05:
        print("结论: 差异显著 (p < 0.05)")
    else:
        print("结论: 差异不显著 (p >= 0.05)")


if __name__ == '__main__':
    main()
