#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1: Motivation 图可视化脚本 (v4 - 高斯滤波 + 热力图)

修复问题：
1. 使用空域高斯滤波代替频域硬截断，消除振铃效应
2. 统一使用热力图配色，更专业直观
3. 改进归一化方式，提高对比度

使用方法:
    python scripts/visualize_motivation_v4.py \
        --data_path /path/to/xxx.npy \
        --output_dir ./outputs/motivation_fig
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frequency_decomposition_spatial(image_slice, sigma_low=10, sigma_high=2):
    """
    使用空域高斯滤波进行分解（更直观，无伪影）
    
    参数:
        image_slice: 2D numpy数组
        sigma_low: 低频滤波器的sigma（像素单位）
        sigma_high: 高频滤波器的sigma
    
    返回:
        dict: 包含各频带结果
    """
    # 低频：大sigma高斯模糊 -> 只保留大尺度结构
    low = gaussian_filter(image_slice, sigma=sigma_low)
    
    # 高频：原图 - 小sigma模糊 -> 保留细节边缘
    smoothed = gaussian_filter(image_slice, sigma=sigma_high)
    high = image_slice - smoothed
    
    # 中频：小sigma模糊 - 大sigma模糊 -> 中等尺度结构
    mid = smoothed - low
    
    # FFT频谱（用于可视化）
    fft_image = np.fft.fft2(image_slice)
    fft_shifted = np.fft.fftshift(fft_image)
    spectrum = np.log1p(np.abs(fft_shifted))
    
    return {
        'original': image_slice,
        'spectrum': spectrum,
        'low': low,
        'mid': mid,
        'high': high,
        'sigma_low': sigma_low,
        'sigma_high': sigma_high
    }


def normalize_for_display(img, method='percentile', **kwargs):
    """归一化图像用于显示"""
    if method == 'percentile':
        p_low = kwargs.get('p_low', 1)
        p_high = kwargs.get('p_high', 99)
        vmin = np.percentile(img, p_low)
        vmax = np.percentile(img, p_high)
    elif method == 'symmetric':
        abs_max = np.percentile(np.abs(img), 99)
        vmin, vmax = -abs_max, abs_max
    elif method == 'positive':
        img = np.abs(img)
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
    else:
        vmin, vmax = img.min(), img.max()
    
    img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return img_norm


def find_best_slice(ct_volume, label_volume):
    """找到血管最丰富的切片"""
    vessel_counts = np.array([label_volume[i].sum() for i in range(ct_volume.shape[0])])
    top_indices = np.argsort(vessel_counts)[-10:]
    best_idx = top_indices[len(top_indices)//2]
    return best_idx, vessel_counts


def create_motivation_figure_v4(ct_volume, label_volume=None,
                                 sigma_low=10, sigma_high=2,
                                 slice_idx=None,
                                 output_path=None):
    """
    创建 Motivation 图 - v4 热力图版
    """
    print("\n[INFO] 开始频率分解 (v4 - 热力图版)...")
    
    # 选择切片
    if slice_idx is None and label_volume is not None:
        slice_idx, vessel_counts = find_best_slice(ct_volume, label_volume)
        print(f"  自动选择切片: {slice_idx} (血管像素: {vessel_counts[slice_idx]:.0f})")
    elif slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2
    
    ct_slice = ct_volume[slice_idx]
    label_slice = label_volume[slice_idx] if label_volume is not None else None
    
    # 频率分解
    decomp = frequency_decomposition_spatial(ct_slice, sigma_low, sigma_high)
    print(f"  空域高斯: sigma_low={sigma_low}, sigma_high={sigma_high}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.2, wspace=0.12)
    
    # ==================== 第一行 ====================
    # (a) 原始CT - 灰度图
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(normalize_for_display(ct_slice), cmap='gray')
    ax_a.set_title('(a) Original CT', fontsize=14, fontweight='bold')
    ax_a.axis('off')
    if label_slice is not None and label_slice.max() > 0:
        ax_a.contour(label_slice, levels=[0.5], colors='#00FF00', linewidths=1.5)
    
    # (b) FFT 频谱 - 热力图
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(decomp['spectrum'], cmap='hot')
    ax_b.set_title('(b) FFT Magnitude Spectrum', fontsize=14, fontweight='bold')
    ax_b.axis('off')
    
    # (c) 频带划分示意图
    ax_c = fig.add_subplot(gs[0, 2])
    
    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    energy_dist = np.exp(-R**2 / 0.3)
    
    ax_c.imshow(energy_dist, cmap='hot', extent=[-1, 1, -1, 1])
    
    theta = np.linspace(0, 2*np.pi, 100)
    for r, label, offset in [(0.25, 'Low', 0), (0.55, 'Mid', 0.4), (0.85, 'High', 0.7)]:
        ax_c.plot(r * np.cos(theta), r * np.sin(theta), 
                  color='white', linewidth=2, linestyle='--')
        ax_c.text(0, offset if offset > 0 else 0, label, 
                  ha='center', va='center', fontsize=11, fontweight='bold', 
                  color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='black', alpha=0.6))
    
    ax_c.set_xlim(-1.1, 1.1)
    ax_c.set_ylim(-1.1, 1.1)
    ax_c.set_title('(c) Frequency Band Partition', fontsize=14, fontweight='bold')
    ax_c.axis('off')
    
    # ==================== 第二行 ====================
    band_configs = [
        ('(d) Low Frequency -> Large Vessels', 'low', 'percentile'),
        ('(e) Mid Frequency -> Medium Details', 'mid', 'symmetric'),
        ('(f) High Frequency -> Fine Edges', 'high', 'positive')
    ]
    
    for i, (title, key, norm_method) in enumerate(band_configs):
        ax = fig.add_subplot(gs[1, i])
        
        recon = decomp[key]
        
        if norm_method == 'symmetric':
            img_norm = normalize_for_display(np.abs(recon), method='percentile', p_low=1, p_high=99)
        elif norm_method == 'positive':
            img_norm = normalize_for_display(np.abs(recon), method='percentile', p_low=1, p_high=99)
        else:
            img_norm = normalize_for_display(recon, method='percentile', p_low=0.5, p_high=99.5)
        
        ax.imshow(img_norm, cmap='hot')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        
        if label_slice is not None and label_slice.max() > 0:
            ax.contour(label_slice, levels=[0.5], colors='#00FF00', linewidths=1.5, alpha=0.9)
    
    fig.suptitle('Figure 1: Frequency Decomposition Reveals Multi-Scale Vessel Structures',
                 fontsize=16, fontweight='bold', y=0.97)
    
    fig.text(0.5, 0.02,
             'Key Insight: Low frequency captures global structure; High frequency captures fine vessel edges.',
             ha='center', fontsize=12, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] 保存到: {output_path}")
    
    plt.close()
    return fig


def create_motivation_simple_v4(ct_volume, label_volume=None,
                                 sigma_low=10, sigma_high=2,
                                 slice_idx=None,
                                 output_path=None):
    """简洁版 (1x4 布局) - 热力图版"""
    print("\n[INFO] 生成简洁版 (热力图)...")
    
    if slice_idx is None and label_volume is not None:
        slice_idx, _ = find_best_slice(ct_volume, label_volume)
    elif slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2
    
    ct_slice = ct_volume[slice_idx]
    label_slice = label_volume[slice_idx] if label_volume is not None else None
    
    decomp = frequency_decomposition_spatial(ct_slice, sigma_low, sigma_high)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    configs = [
        ('Original CT', 'original', 'gray', 'percentile'),
        ('Low Frequency\n(Large Vessels)', 'low', 'hot', 'percentile'),
        ('Mid Frequency\n(Medium Details)', 'mid', 'hot', 'abs'),
        ('High Frequency\n(Fine Edges)', 'high', 'hot', 'abs')
    ]
    
    for ax, (title, key, cmap, norm) in zip(axes, configs):
        img = decomp[key]
        
        if norm == 'abs':
            img_norm = normalize_for_display(np.abs(img), method='percentile', p_low=1, p_high=99)
        else:
            img_norm = normalize_for_display(img)
        
        ax.imshow(img_norm, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        
        if label_slice is not None and label_slice.max() > 0:
            contour_color = '#FF0000' if cmap == 'gray' else '#00FF00'
            ax.contour(label_slice, levels=[0.5], colors=contour_color, linewidths=1.5)
    
    fig.text(0.5, 0.02,
             f'Gaussian Decomposition: sigma_low={sigma_low} (blur), sigma_high={sigma_high} (detail)',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] 保存到: {output_path}")
    
    plt.close()
    return fig


def create_motivation_gray_v4(ct_volume, label_volume=None,
                               sigma_low=10, sigma_high=2,
                               slice_idx=None,
                               output_path=None):
    """灰度版 (1x4 布局)"""
    print("\n[INFO] 生成灰度版...")
    
    if slice_idx is None and label_volume is not None:
        slice_idx, _ = find_best_slice(ct_volume, label_volume)
    elif slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2
    
    ct_slice = ct_volume[slice_idx]
    label_slice = label_volume[slice_idx] if label_volume is not None else None
    
    decomp = frequency_decomposition_spatial(ct_slice, sigma_low, sigma_high)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    configs = [
        ('Original CT', 'original', 'percentile'),
        ('Low Frequency\n(Large Vessels)', 'low', 'percentile'),
        ('Mid Frequency\n(Medium Details)', 'mid', 'abs'),
        ('High Frequency\n(Fine Edges)', 'high', 'abs')
    ]
    
    for ax, (title, key, norm) in zip(axes, configs):
        img = decomp[key]
        
        if norm == 'abs':
            img_norm = normalize_for_display(np.abs(img), method='percentile', p_low=1, p_high=99)
        else:
            img_norm = normalize_for_display(img)
        
        ax.imshow(img_norm, cmap='gray')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        
        if label_slice is not None and label_slice.max() > 0:
            ax.contour(label_slice, levels=[0.5], colors='#FF0000', linewidths=1.5)
    
    fig.text(0.5, 0.02,
             f'Gaussian Decomposition: sigma_low={sigma_low}, sigma_high={sigma_high}',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] 保存到: {output_path}")
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description='F1 Motivation Figure (v4)')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/motivation_fig')
    parser.add_argument('--slice_idx', type=int, default=None)
    parser.add_argument('--sigma_low', type=float, default=10)
    parser.add_argument('--sigma_high', type=float, default=2)
    parser.add_argument('--style', type=str, default='all', 
                        choices=['full', 'simple', 'gray', 'all'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 加载数据: {args.data_path}")
    data = np.load(args.data_path)
    
    if data.ndim == 4 and data.shape[0] >= 2:
        ct_volume = data[0]
        label_volume = (data[1] == 1).astype(float)
        print(f"  CT shape: {ct_volume.shape}")
    else:
        ct_volume = data
        label_volume = None
    
    if args.style in ['full', 'all']:
        create_motivation_figure_v4(
            ct_volume, label_volume,
            sigma_low=args.sigma_low,
            sigma_high=args.sigma_high,
            slice_idx=args.slice_idx,
            output_path=output_dir / 'motivation_v4_full.png'
        )
    
    if args.style in ['simple', 'all']:
        create_motivation_simple_v4(
            ct_volume, label_volume,
            sigma_low=args.sigma_low,
            sigma_high=args.sigma_high,
            slice_idx=args.slice_idx,
            output_path=output_dir / 'motivation_v4_simple.png'
        )
    
    if args.style in ['gray', 'all']:
        create_motivation_gray_v4(
            ct_volume, label_volume,
            sigma_low=args.sigma_low,
            sigma_high=args.sigma_high,
            slice_idx=args.slice_idx,
            output_path=output_dir / 'motivation_v4_gray.png'
        )
    
    print("\n[DONE] 完成！")


if __name__ == '__main__':
    main()
