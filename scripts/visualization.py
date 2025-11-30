

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import os

# 设置样式
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def create_vessel_cmap():
    """创建血管专用colormap - 红橙渐变"""
    colors = ['#000000', '#8B0000', '#FF4500', '#FF6347', '#FFA500', '#FFD700']
    return LinearSegmentedColormap.from_list('vessel', colors, N=256)


def create_overlay_image(pred, gt):
    """
    创建叠加对比图
    - 绿色: True Positive (正确检测)
    - 红色: False Positive (过分割)
    - 蓝色: False Negative (漏检)
    - 黄色: 预测和GT重叠的边界
    """
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    rgb = np.zeros((*pred_bin.shape, 3), dtype=np.float32)
    
    tp = pred_bin * gt_bin
    fp = pred_bin * (1 - gt_bin)
    fn = (1 - pred_bin) * gt_bin
    
    # 绿色通道: TP
    rgb[..., 1] = tp * 0.9
    # 红色通道: FP
    rgb[..., 0] = fp * 0.9
    # 蓝色通道: FN
    rgb[..., 2] = fn * 0.9
    
    return rgb


def visualize_mip_comparison(pred, gt, case_id, output_dir):
    """
    美化版MIP对比图
    左右对比 + 中间叠加误差图
    """
    pred = pred.squeeze()
    gt = gt.squeeze()
    
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    # 计算Dice
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    
    # 创建图像 - 3行3列
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), facecolor='black')
    
    views = [
        (0, 'Axial (Top-Down)'),
        (1, 'Coronal (Front)'),
        (2, 'Sagittal (Side)')
    ]
    
    vessel_cmap = create_vessel_cmap()
    
    for col, (axis, view_name) in enumerate(views):
        # Prediction MIP
        pred_mip = np.max(pred_bin, axis=axis)
        axes[0, col].imshow(pred_mip, cmap=vessel_cmap, vmin=0, vmax=1)
        axes[0, col].set_title(f'Prediction\n{view_name}', color='white', fontsize=11, pad=10)
        axes[0, col].axis('off')
        
        # Ground Truth MIP
        gt_mip = np.max(gt_bin, axis=axis)
        axes[1, col].imshow(gt_mip, cmap=vessel_cmap, vmin=0, vmax=1)
        axes[1, col].set_title(f'Ground Truth\n{view_name}', color='white', fontsize=11, pad=10)
        axes[1, col].axis('off')
        
        # Overlay - 误差图
        overlay = create_overlay_image(
            np.max(pred_bin, axis=axis),
            np.max(gt_bin, axis=axis)
        )
        axes[2, col].imshow(overlay)
        axes[2, col].set_title(f'Error Map\n{view_name}', color='white', fontsize=11, pad=10)
        axes[2, col].axis('off')
    
    # 总标题
    fig.suptitle(f'{case_id}\nDice Score: {dice:.4f}',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    
    # 添加图例
    legend_text = '■ Green: Correct (TP)    ■ Red: Over-seg (FP)    ■ Blue: Missed (FN)'
    fig.text(0.5, 0.01, legend_text, ha='center', color='white', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.04, hspace=0.15, wspace=0.05)
    
    # 保存
    output_path = os.path.join(output_dir, f'{case_id}_MIP_comparison.png')
    plt.savefig(output_path, dpi=200, facecolor='black', edgecolor='none',
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    return dice


def visualize_slices_grid(pred, gt, image, case_id, output_dir, num_slices=8):
    """
    切片网格视图 - 每个切片显示CT+预测叠加
    """
    pred = pred.squeeze()
    gt = gt.squeeze()
    if image is not None:
        image = image.squeeze()
    
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    # 计算Dice
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    
    # 选择有内容的切片
    z_sums = gt_bin.sum(axis=(1, 2))
    z_indices = np.where(z_sums > 10)[0]  # 至少10个前景像素
    
    if len(z_indices) < num_slices:
        z_indices = np.arange(pred.shape[0])
    
    # 均匀选择
    step = max(1, len(z_indices) // num_slices)
    selected_z = z_indices[::step][:num_slices]
    
    # 创建网格 - 2行4列
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8), facecolor='black')
    
    for idx, z in enumerate(selected_z[:n_rows * n_cols]):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        # 背景：CT图像（如果有）
        if image is not None:
            # 归一化CT图像用于显示
            ct_slice = image[z]
            ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
            ax.imshow(ct_norm, cmap='gray', alpha=0.6)
        
        # 叠加预测和GT的对比
        overlay = create_overlay_image(pred_bin[z], gt_bin[z])
        ax.imshow(overlay, alpha=0.7)
        
        # 切片编号
        ax.text(5, 15, f'Z={z}', color='white', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')
    
    # 隐藏多余的axes
    for idx in range(len(selected_z), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # 标题和图例
    fig.suptitle(f'{case_id} - Slice View | Dice: {dice:.4f}',
                 color='white', fontsize=14, fontweight='bold')
    
    legend_text = '■ Green: Correct    ■ Red: Over-segmentation    ■ Blue: Missed'
    fig.text(0.5, 0.02, legend_text, ha='center', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.06, hspace=0.08, wspace=0.02)
    
    output_path = os.path.join(output_dir, f'{case_id}_slices.png')
    plt.savefig(output_path, dpi=200, facecolor='black', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return dice


def visualize_single_best_view(pred, gt, case_id, output_dir):
    """
    单张最佳视角图 - 适合放论文
    """
    pred = pred.squeeze()
    gt = gt.squeeze()
    
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    # 计算指标
    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    fn = np.sum((1 - pred_bin) * gt_bin)
    
    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    
    # 选择Sagittal视角（侧面，血管树最清晰）
    pred_mip = np.max(pred_bin, axis=2)
    gt_mip = np.max(gt_bin, axis=2)
    
    # 创建图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    
    vessel_cmap = create_vessel_cmap()
    
    # Prediction
    axes[0].imshow(pred_mip, cmap='hot')
    axes[0].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mip, cmap='hot')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Error Map
    overlay = create_overlay_image(pred_mip, gt_mip)
    axes[2].imshow(overlay)
    axes[2].set_title('Error Analysis', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 指标文本
    metrics_text = f'Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    fig.suptitle(f'{case_id}\n{metrics_text}', fontsize=12, y=0.98)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{case_id}_paper_view.png')
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    return dice, precision, recall


def load_ground_truth(data_path):
    """加载预处理数据"""
    data = np.load(data_path)
    if data.ndim == 4 and data.shape[0] == 2:
        image = data[0]
        label = (data[1] == 1).astype(np.float32)  # 只保留血管(1)
        return image, label
    return None, (data == 1).astype(np.float32)


def main():
    # ============================================
    # 配置路径
    # ============================================
    
    # 预测结果目录
    pred_dir = Path("/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/validation/predictions")
    
    # GT数据目录 - 使用你有权限访问的路径
    gt_dir = Path("/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/processed/data")
    
    # 输出目录
    output_dir = Path("/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # 处理
    # ============================================
    
    pred_files = sorted(pred_dir.glob("*_pred.npy"))
    print(f"Found {len(pred_files)} prediction files")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    results = []
    
    for pred_file in pred_files:
        case_id = pred_file.stem.replace("_pred", "")
        
        # 加载预测
        pred = np.load(pred_file)
        
        # 加载GT
        gt_file = gt_dir / f"{case_id}.npy"
        if gt_file.exists():
            image, gt = load_ground_truth(gt_file)
        else:
            print(f"  Skipping {case_id}: GT not found")
            continue
        
        # 生成三种可视化
        print(f"  Processing {case_id}...")
        
        # 1. MIP对比图（黑色背景，醒目）
        dice1 = visualize_mip_comparison(pred, gt, case_id, output_dir)
        
        # 2. 切片网格图
        dice2 = visualize_slices_grid(pred, gt, image, case_id, output_dir)
        
        # 3. 论文用单图
        dice3, prec, recall = visualize_single_best_view(pred, gt, case_id, output_dir)
        
        results.append({'case': case_id, 'dice': dice3, 'precision': prec, 'recall': recall})
        print(f"    Dice={dice3:.4f}, Precision={prec:.4f}, Recall={recall:.4f}")
    
    # 汇总
    print("\n" + "=" * 60)
    if results:
        dices = [r['dice'] for r in results]
        print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Range: [{np.min(dices):.4f}, {np.max(dices):.4f}]")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()