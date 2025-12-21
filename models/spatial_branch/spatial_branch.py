# models/spatial_branch/spatial_branch.py
"""Multi-scale spatial branch with parallel convolution paths."""

import torch
import torch.nn as nn
from typing import Tuple

from .multi_scale_conv import PathConv


class EnhancedSpatialBranch(nn.Module):
    """
    Multi-scale spatial branch with three parallel paths.
    
    Paths with different receptive fields:
    - Path1 (3x3x3): Small receptive field, fine details
    - Path2 (5x5x5): Medium receptive field, main structures
    - Path3 (7x7x7): Large receptive field, global context
    
    Channel allocation: 1/4 : 1/2 : 1/4
    
    Args:
        in_channels: Input channels
        out_channels: Total output channels (must be divisible by 4)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        assert out_channels % 4 == 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.ch1 = out_channels // 4
        self.ch2 = out_channels // 2
        self.ch3 = out_channels // 4
        
        self.path1 = PathConv(
            in_channels=in_channels,
            out_channels=self.ch1,
            kernel_size=3,
            use_depthwise_separable=True
        )
        
        self.path2 = PathConv(
            in_channels=in_channels,
            out_channels=self.ch2,
            kernel_size=5,
            use_depthwise_separable=True
        )
        
        self.path3 = PathConv(
            in_channels=in_channels,
            out_channels=self.ch3,
            kernel_size=7,
            use_depthwise_separable=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_channels, D, H, W]
        Returns:
            (path1, path2, path3): Features from each path
        """
        return self.path1(x), self.path2(x), self.path3(x)
    
    def forward_concat(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning concatenated features."""
        path1_out, path2_out, path3_out = self.forward(x)
        return torch.cat([path1_out, path2_out, path3_out], dim=1)
    
    def get_channel_info(self) -> dict:
        return {
            'path1_channels': self.ch1,
            'path2_channels': self.ch2,
            'path3_channels': self.ch3,
            'total_channels': self.out_channels
        }
