# scripts/train_5fold.py
"""
SpectraLink-Net 五折交叉验证训练脚本 (v4.2)

显存优化特性：
1. 梯度检查点 (Gradient Checkpointing)
2. 混合精度训练 (AMP)
3. 梯度累积 (Gradient Accumulation)
4. 及时释放显存

v4.2 新增:
- 深度监督支持
"""

import os
import sys
import argparse
import yaml
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_spectralink_net
from losses import DiceBCELoss, DeepSupervisionWrapper
from data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='SpectraLink-Net 5-Fold Training')
    parser.add_argument('--config', type=str, default='config/config_5fold.yaml')
    parser.add_argument('--fold', type=int, required=True, help='Fold (0-4)')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return local_rank, rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_dice(pred, target, threshold=0.5):
    """计算 Dice 系数"""
    # 如果是多尺度输出，取第一个（主输出）
    if isinstance(pred, list):
        pred = pred[0]
    
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / (union + 1e-8)).item()


def create_criterion(config: dict, device: torch.device) -> nn.Module:
    """创建损失函数 (v4.2 - 支持深度监督)"""
    loss_cfg = config.get('loss', {})
    
    # 基础损失
    base_loss = DiceBCELoss(
        w_dice=loss_cfg.get('w_dice', 0.7),
        w_bce=loss_cfg.get('w_bce', 0.3),
        pos_weight=loss_cfg.get('pos_weight', 6.0)
    )
    
    # 是否使用深度监督
    use_ds = loss_cfg.get('deep_supervision', False)
    model_ds = config.get('model', {}).get('deep_supervision', False)
    
    if use_ds and model_ds:
        weights = loss_cfg.get('ds_weights', [1.0, 0.5, 0.25])
        criterion = DeepSupervisionWrapper(
            base_loss=base_loss,
            weights=weights,
            downsample_mode='max'  # 血管分割用 max 保证正类不丢失
        )
    else:
        criterion = base_loss
    
    return criterion.to(device)


def print_config(config, fold, rank):
    """打印配置信息"""
    if rank != 0:
        return

    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    loss_cfg = config.get('loss', {})

    print("\n" + "=" * 70)
    print("  SpectraLink-Net Training (v4.2)")
    print("=" * 70)

    print("\n" + "=" * 28 + "[Basic Info]" + "=" * 30)
    print(f"  Fold:              {fold}")
    print(f"  Output dir:        {config.get('output', {}).get('base_dir', 'outputs')}/fold_{fold}")
    print(f"  GPUs:              {torch.cuda.device_count()}")

    print("\n" + "=" * 27 + "[Model Config]" + "=" * 29)
    print(f"  Model:             SpectraLink-Net v4.2")
    print(f"  Base channels:     {model_cfg.get('base_channels', 32)}")
    print(f"  Num levels:        {model_cfg.get('num_levels', 3)}")
    print(f"  Freq token base:   {model_cfg.get('freq_token_base', 4)}")
    print(f"  Num heads:         {model_cfg.get('num_heads', 4)}")
    print(f"  Expansion ratio:   {model_cfg.get('expansion_ratio', 2)}")
    print(f"  Dropout:           {model_cfg.get('dropout', 0.0)}")
    print(f"  Use checkpoint:    {model_cfg.get('use_checkpoint', False)}")
    print(f"  Deep supervision:  {model_cfg.get('deep_supervision', False)}")
    
    # 稀疏注意力配置
    use_sparse = model_cfg.get('use_sparse_attention', False)
    print(f"  Sparse attention:  {use_sparse}")
    if use_sparse:
        print(f"    - Sample mode:   {model_cfg.get('sample_mode', 'auto')}")
        if model_cfg.get('sample_mode') == 'fixed':
            print(f"    - Num samples:   {model_cfg.get('num_samples', 4096)}")
        elif model_cfg.get('sample_mode') == 'ratio':
            print(f"    - Sample ratio:  {model_cfg.get('sample_ratio', 0.005)}")
        sparse_levels = model_cfg.get('sparse_levels', None)
        print(f"    - Sparse levels: {sparse_levels if sparse_levels else 'all'}")

    print("\n" + "=" * 28 + "[Data Config]" + "=" * 29)
    print(f"  Patch size:        {data_cfg.get('patch_size', [64, 128, 128])}")
    print(f"  Batch size:        {data_cfg.get('batch_size', 2)}")
    print(f"  Samples/volume:    {data_cfg.get('samples_per_volume', 4)}")
    print(f"  Oversample FG:     {data_cfg.get('oversample_foreground', 0.7)}")
    print(f"  Num workers:       {data_cfg.get('num_workers', 8)}")

    print("\n" + "=" * 26 + "[Training Config]" + "=" * 27)
    print(f"  Epochs:            {train_cfg.get('num_epochs', 250)}")
    print(f"  Warmup epochs:     {train_cfg.get('warmup_epochs', 10)}")
    print(f"  Learning rate:     {train_cfg.get('learning_rate', 0.001)}")
    print(f"  Min LR:            {train_cfg.get('min_lr', 1e-6)}")
    print(f"  Weight decay:      {train_cfg.get('weight_decay', 1e-5)}")
    print(f"  Gradient clip:     {train_cfg.get('gradient_clip', 1.0)}")
    print(f"  Early stop:        {train_cfg.get('early_stop_patience', 100)} epochs")

    print("\n" + "=" * 25 + "[Memory Optimization]" + "=" * 24)
    print(f"  Use AMP:           {train_cfg.get('use_amp', True)}")
    print(f"  Accumulation:      {train_cfg.get('accumulation_steps', 1)} steps")
    print(f"  Grad checkpoint:   {model_cfg.get('use_checkpoint', False)}")

    print("\n" + "=" * 28 + "[Loss Config]" + "=" * 29)
    print(f"  W_dice:            {loss_cfg.get('w_dice', 0.7)}")
    print(f"  W_bce:             {loss_cfg.get('w_bce', 0.3)}")
    print(f"  Pos weight:        {loss_cfg.get('pos_weight', 6.0)}")
    print(f"  Deep supervision:  {loss_cfg.get('deep_supervision', False)}")
    if loss_cfg.get('deep_supervision', False):
        print(f"    - DS weights:    {loss_cfg.get('ds_weights', [1.0, 0.5, 0.25])}")

    print("\n" + "=" * 70)


def train_one_epoch(
        model, train_loader, criterion, optimizer, scaler,
        device, epoch, config, rank, accumulation_steps
):
    """训练一个epoch (v4.2 - 支持深度监督)"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    use_amp = config.get('training', {}).get('use_amp', True)
    grad_clip = config.get('training', {}).get('gradient_clip', 1.0)

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train", ncols=120)
    else:
        pbar = train_loader

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 混合精度前向传播
        if use_amp:
            with autocast():
                outputs = model(images)
                # DeepSupervisionWrapper 自动处理 List[Tensor] 或 Tensor
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

        # 梯度累积：每 accumulation_steps 步更新一次
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # 计算 dice (使用主输出)
        with torch.no_grad():
            dice = compute_dice(outputs, labels)

        total_loss += loss.item() * accumulation_steps
        total_dice += dice
        num_batches += 1

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_dice = total_dice / num_batches
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'dice': f'{avg_dice:.4f}'
            })

    # 处理最后不足 accumulation_steps 的 batch
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / num_batches, total_dice / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, rank):
    """验证 (v4.2 - 支持深度监督)"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} Val  ", ncols=120)
    else:
        pbar = val_loader

    for batch_data in pbar:
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            
            # 验证时 model.eval() 返回单输出
            # 如果 criterion 是 DeepSupervisionWrapper，使用其 base_loss
            if isinstance(criterion, DeepSupervisionWrapper) and not isinstance(outputs, list):
                loss = criterion.base_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)

        dice = compute_dice(outputs, labels)

        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_dice = total_dice / num_batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'dice': f'{avg_dice:.4f}'
            })

    return total_loss / num_batches, total_dice / num_batches


def get_fold_dataloaders(config, fold, distributed, rank, world_size):
    """获取指定fold的数据加载器"""
    data_cfg = config.get('data', {})
    split_file = data_cfg.get('split_file')

    with open(split_file, 'r') as f:
        folds = json.load(f)

    fold_key = f"fold_{fold}"
    if fold_key not in folds:
        raise ValueError(f"Fold {fold} not found in {split_file}")

    temp_split = {
        'train': folds[fold_key]['train'],
        'val': folds[fold_key]['val']
    }

    temp_split_path = f"/tmp/temp_split_fold{fold}.json"
    with open(temp_split_path, 'w') as f:
        json.dump(temp_split, f)

    config_copy = config.copy()
    config_copy['data'] = config['data'].copy()
    config_copy['data']['split_file'] = temp_split_path

    train_loader, val_loader = get_dataloaders(
        config_copy, distributed=distributed, rank=rank, world_size=world_size
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    # 分布式设置
    local_rank, rank, world_size, distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 输出目录
    output_dir = os.path.join(
        config.get('output', {}).get('base_dir', 'outputs/5fold'),
        f'fold_{args.fold}'
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # 打印配置
    print_config(config, args.fold, rank)

    # 创建模型
    model = build_spectralink_net(config).to(device)

    # 打印参数量
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n" + "=" * 29 + "[Model]" + "=" * 34)
        print(f"  Parameters:        {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
        print(f"  Trainable:         {trainable_params:,}")

    # 分布式包装
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 数据加载器
    train_loader, val_loader = get_fold_dataloaders(
        config, args.fold, distributed, rank, world_size
    )

    if rank == 0:
        print(f"\n" + "=" * 29 + "[Dataset]" + "=" * 32)
        print(f"  Train samples:     {len(train_loader.dataset)}")
        print(f"  Val samples:       {len(val_loader.dataset)}")
        print(f"  Train batches:     {len(train_loader)}")
        print(f"  Val batches:       {len(val_loader)}")
        print("=" * 70 + "\n")

    # 损失函数 (v4.2 - 支持深度监督)
    criterion = create_criterion(config, device)
    
    if rank == 0:
        loss_cfg = config.get('loss', {})
        if loss_cfg.get('deep_supervision', False):
            print(f"  使用深度监督损失, weights={loss_cfg.get('ds_weights', [1.0, 0.5, 0.25])}")
        else:
            print(f"  使用标准 DiceBCE 损失")

    # 优化器
    train_cfg = config.get('training', {})
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get('learning_rate', 0.001),
        weight_decay=train_cfg.get('weight_decay', 1e-5)
    )

    # 学习率调度器
    num_epochs = train_cfg.get('num_epochs', 250)
    warmup_epochs = train_cfg.get('warmup_epochs', 10)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=train_cfg.get('min_lr', 1e-6)
    )

    # 混合精度 scaler
    use_amp = train_cfg.get('use_amp', True)
    scaler = GradScaler() if use_amp else None

    # 梯度累积步数
    accumulation_steps = train_cfg.get('accumulation_steps', 1)

    # 恢复训练
    start_epoch = 0
    best_dice = 0.0
    epochs_without_improvement = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")

    # 训练循环
    early_stop_patience = train_cfg.get('early_stop_patience', 100)

    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = train_cfg.get('learning_rate', 0.001) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # 训练
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, config, rank, accumulation_steps
        )

        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch, rank)

        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f} | "
                  f"LR: {current_lr:.2e}")

            # 保存最佳模型
            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0

                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': config
                }
                torch.save(save_dict, os.path.join(output_dir, 'best_model.pth'))
                print(f"  ★ NEW BEST: Dice={best_dice:.4f} at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            # 定期保存
            save_every = train_cfg.get('save_every', 10)
            if (epoch + 1) % save_every == 0:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': config
                }
                torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch:03d}.pth'))

        # 同步 epochs_without_improvement
        if distributed:
            tensor = torch.tensor([epochs_without_improvement], device=device)
            dist.broadcast(tensor, 0)
            epochs_without_improvement = tensor.item()

        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            if rank == 0:
                print(f"\nEarly stopping at epoch {epoch}. Best dice: {best_dice:.4f}")
            break

        # 及时释放显存
        if (epoch + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if distributed:
            dist.barrier()

    # 训练结束
    if rank == 0:
        print(f"\nTraining finished. Best dice: {best_dice:.4f}")

        # 保存最终模型
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'config': config
        }
        torch.save(save_dict, os.path.join(output_dir, 'final_model.pth'))

        # 保存结果
        result = {'fold': args.fold, 'best_dice': best_dice}
        with open(os.path.join(output_dir, 'result.json'), 'w') as f:
            json.dump(result, f)

    cleanup_distributed()


if __name__ == '__main__':
    main()
