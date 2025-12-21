# scripts/generate_5fold.py
"""
生成五折交叉验证划分文件
只需运行一次，所有方法共用同一份划分
"""

import os
import sys
import json
import argparse
from pathlib import Path
from sklearn.model_selection import KFold

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 5-Fold Split')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='预处理后的数据目录（包含.npy文件）')
    parser.add_argument('--output', type=str, default='5fold_splits.json',
                        help='输出文件路径')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='折数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def generate_5fold_split(data_dir: str, output_path: str, n_splits: int = 5, seed: int = 42):
    """
    生成五折交叉验证划分
    
    参数:
        data_dir: 数据目录
        output_path: 输出JSON文件路径
        n_splits: 折数
        seed: 随机种子（只在生成时使用一次）
    """
    # 获取所有case
    cases = sorted([f.replace('.npy', '') for f in os.listdir(data_dir) if f.endswith('.npy')])
    print(f"找到 {len(cases)} 个样本")
    
    # 五折划分
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    folds = {}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cases)):
        folds[f"fold_{fold_idx}"] = {
            "train": [cases[i] for i in train_idx],
            "val": [cases[i] for i in val_idx]
        }
        print(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(folds, f, indent=2)
    
    print(f"\n五折划分已保存到: {output_path}")
    print("=" * 50)
    print("重要: 所有方法必须使用同一份划分文件!")
    print("=" * 50)
    
    return folds


def main():
    args = parse_args()
    generate_5fold_split(
        data_dir=args.data_dir,
        output_path=args.output,
        n_splits=args.n_splits,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
