# models/components/conv_blocks.py
"""Common convolution blocks used across the network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitConv(nn.Module):
    """Initial convolution: Conv3d -> InstanceNorm -> ReLU"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """U-Net style double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class DownsampleBlock(nn.Module):
    """Downsample block: MaxPool3d(2) + 1x1 Conv for channel expansion."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.act(self.norm(self.conv(x)))
        return x


class UpsampleBlock(nn.Module):
    """Upsample block: Trilinear interpolation + 3x3 Conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        
        x = self.act(self.norm(self.conv(x)))
        return x


class ResidualConv(nn.Module):
    """Residual convolution block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv_block = DoubleConv(in_channels, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv_block(x)
        return self.act(out + identity)
