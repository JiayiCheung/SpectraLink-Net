# losses/deep_supervision.py
"""Deep supervision loss for multi-scale training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss with multi-scale predictions.
    
    Args:
        base_loss: Base loss function (e.g., DiceBCELoss)
        weights: Scale weights [w_1x, w_1/4, ...], default [1.0, 0.5, 0.25]
        scales: Downsampling factors, default [1, 4, 2]
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
        scales: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.weights = weights if weights is not None else [1.0, 0.5, 0.25]
        self.scales = scales if scales is not None else [1, 4, 2]
    
    def _downsample_target(self, target: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample GT using max_pool to preserve positive labels."""
        if scale == 1:
            return target
        return F.max_pool3d(target.float(), kernel_size=scale, stride=scale)
    
    def forward(
        self,
        preds: List[torch.Tensor],
        target: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """Compute deep supervision loss."""
        assert len(preds) <= len(self.weights)
        assert len(preds) <= len(self.scales)
        
        total_loss = 0.0
        loss_dict = {}
        
        for i, (pred, weight, scale) in enumerate(zip(preds, self.weights, self.scales)):
            gt = self._downsample_target(target, scale)
            
            if pred.shape[2:] != gt.shape[2:]:
                gt = F.interpolate(gt.float(), size=pred.shape[2:], mode='nearest')
            
            loss = self.base_loss(pred, gt)
            total_loss = total_loss + weight * loss
            
            scale_name = f'ds_scale_{scale}x' if scale > 1 else 'ds_full'
            loss_dict[scale_name] = loss.item()
        
        loss_dict['ds_total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        if return_dict:
            return total_loss, loss_dict
        return total_loss


class DeepSupervisionWrapper(nn.Module):
    """
    Flexible deep supervision wrapper with auto-downsampling.
    
    Args:
        base_loss: Base loss function
        weights: Scale weights (None for auto exponential decay)
        downsample_mode: 'max', 'avg', or 'nearest'
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
        downsample_mode: str = 'max'
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.weights = weights
        self.downsample_mode = downsample_mode
    
    def _get_weights(self, num_outputs: int) -> List[float]:
        if self.weights is not None:
            return self.weights[:num_outputs]
        return [0.5 ** i for i in range(num_outputs)]
    
    def _downsample(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        if x.shape[2:] == target_size:
            return x
        
        x_float = x.float()
        factors = [s // t for s, t in zip(x.shape[2:], target_size)]
        
        if self.downsample_mode == 'max' and all(f > 0 for f in factors):
            return F.max_pool3d(x_float, kernel_size=factors, stride=factors)
        elif self.downsample_mode == 'avg' and all(f > 0 for f in factors):
            return F.avg_pool3d(x_float, kernel_size=factors, stride=factors)
        else:
            return F.interpolate(x_float, size=target_size, mode='nearest')
    
    def forward(
        self,
        preds: List[torch.Tensor],
        target: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """Compute deep supervision loss."""
        weights = self._get_weights(len(preds))
        
        total_loss = 0.0
        loss_dict = {}
        
        for i, (pred, weight) in enumerate(zip(preds, weights)):
            gt = self._downsample(target, pred.shape[2:])
            loss = self.base_loss(pred, gt)
            total_loss = total_loss + weight * loss
            loss_dict[f'ds_{i}'] = loss.item()
        
        loss_dict['ds_total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        if return_dict:
            return total_loss, loss_dict
        return total_loss


def build_deep_supervision_loss(
    base_loss: nn.Module,
    num_outputs: int = 3,
    weight_mode: str = 'exponential',
    downsample_mode: str = 'max'
) -> DeepSupervisionWrapper:
    """
    Build deep supervision loss.
    
    Args:
        base_loss: Base loss function
        num_outputs: Number of outputs (main + auxiliary)
        weight_mode: 'exponential', 'linear', or 'uniform'
        downsample_mode: GT downsampling mode
    """
    if weight_mode == 'exponential':
        weights = [0.5 ** i for i in range(num_outputs)]
    elif weight_mode == 'linear':
        weights = [1.0 - i * (1.0 / num_outputs) for i in range(num_outputs)]
    elif weight_mode == 'uniform':
        weights = [1.0] * num_outputs
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    
    return DeepSupervisionWrapper(
        base_loss=base_loss,
        weights=weights,
        downsample_mode=downsample_mode
    )
