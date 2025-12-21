# models/frequency_branch/frequency_branch.py
"""3D FFT-based frequency branch with learnable band partition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from typing import Tuple, Optional

from .band_partition import LearnableBandPartition
from .complex_block import ComplexBlock


class ImprovedFrequencyBranch(nn.Module):
    """
    Frequency branch with 3D FFT and learnable band partition.
    
    Pipeline: FFT -> Band Partition -> ComplexBlock Enhancement -> IFFT -> Spatial Compress -> Token Projection
    
    Outputs 3 token sequences corresponding to spatial branch paths:
    - token0 (low-freq) -> guide path3 (large receptive field)
    - token1 (mid-freq) -> guide path2 (medium receptive field)
    - token2 (high-freq) -> guide path1 (small receptive field)
    
    Args:
        in_channels: Input channels
        token_channels: Output token channels
        expansion_ratio: ComplexBlock expansion ratio
        init_cuts: Initial band cut points (r1, r2)
        transition_width: Band transition width
        pool_sizes: Pooling output sizes for each band
    """
    
    def __init__(
            self,
            in_channels: int,
            token_channels: int,
            expansion_ratio: int = 2,
            init_cuts: Tuple[float, float] = (0.2, 0.5),
            transition_width: float = 0.05,
            pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.token_channels = token_channels
        
        if pool_sizes is None:
            pool_sizes = ((2, 2, 2), (4, 4, 4), (4, 4, 4))
        
        self.pool_sizes = pool_sizes
        
        self.p_low = pool_sizes[0][0] * pool_sizes[0][1] * pool_sizes[0][2]
        self.p_mid = pool_sizes[1][0] * pool_sizes[1][1] * pool_sizes[1][2]
        self.p_high = pool_sizes[2][0] * pool_sizes[2][1] * pool_sizes[2][2]
        
        self.band_partition = LearnableBandPartition(
            init_cuts=init_cuts,
            transition_width=transition_width
        )
        
        self.enhance_low = ComplexBlock(channels=in_channels, expansion_ratio=expansion_ratio)
        self.enhance_mid = ComplexBlock(channels=in_channels, expansion_ratio=expansion_ratio)
        self.enhance_high = ComplexBlock(channels=in_channels, expansion_ratio=expansion_ratio)
        
        self.spatial_compress_low = self._make_spatial_compressor(in_channels, pool_sizes[0])
        self.spatial_compress_mid = self._make_spatial_compressor(in_channels, pool_sizes[1])
        self.spatial_compress_high = self._make_spatial_compressor(in_channels, pool_sizes[2])
        
        self.proj_low = nn.Linear(in_channels, token_channels)
        self.proj_mid = nn.Linear(in_channels, token_channels)
        self.proj_high = nn.Linear(in_channels, token_channels)
    
    def _make_spatial_compressor(
            self,
            channels: int,
            output_size: Tuple[int, int, int]
    ) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.LayerNorm([channels, *output_size]),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            (token0, token1, token2): Each [B, p, token_channels]
        """
        B, C, D, H, W = x.shape
        
        with amp.autocast(enabled=False):
            x_fp32 = x.float()
            
            spectrum = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1))
            spec_low, spec_mid, spec_high = self.band_partition(spectrum)
            
            spec_low = self.enhance_low(spec_low)
            spec_mid = self.enhance_mid(spec_mid)
            spec_high = self.enhance_high(spec_high)
            
            spatial_low = torch.fft.irfftn(spec_low, s=(D, H, W), dim=(-3, -2, -1))
            spatial_mid = torch.fft.irfftn(spec_mid, s=(D, H, W), dim=(-3, -2, -1))
            spatial_high = torch.fft.irfftn(spec_high, s=(D, H, W), dim=(-3, -2, -1))
        
        compressed_low = self.spatial_compress_low(spatial_low)
        compressed_mid = self.spatial_compress_mid(spatial_mid)
        compressed_high = self.spatial_compress_high(spatial_high)
        
        tokens_low = compressed_low.flatten(2).transpose(1, 2)
        tokens_mid = compressed_mid.flatten(2).transpose(1, 2)
        tokens_high = compressed_high.flatten(2).transpose(1, 2)
        
        token0 = self.proj_low(tokens_low)
        token1 = self.proj_mid(tokens_mid)
        token2 = self.proj_high(tokens_high)
        
        return token0, token1, token2
    
    def forward_with_spatial(self, x: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Forward pass returning both tokens and spatial features."""
        B, C, D, H, W = x.shape
        
        with amp.autocast(enabled=False):
            x_fp32 = x.float()
            
            spectrum = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1))
            spec_low, spec_mid, spec_high = self.band_partition(spectrum)
            
            spec_low = self.enhance_low(spec_low)
            spec_mid = self.enhance_mid(spec_mid)
            spec_high = self.enhance_high(spec_high)
            
            spatial_low = torch.fft.irfftn(spec_low, s=(D, H, W), dim=(-3, -2, -1))
            spatial_mid = torch.fft.irfftn(spec_mid, s=(D, H, W), dim=(-3, -2, -1))
            spatial_high = torch.fft.irfftn(spec_high, s=(D, H, W), dim=(-3, -2, -1))
        
        compressed_low = self.spatial_compress_low(spatial_low)
        compressed_mid = self.spatial_compress_mid(spatial_mid)
        compressed_high = self.spatial_compress_high(spatial_high)
        
        tokens_low = compressed_low.flatten(2).transpose(1, 2)
        tokens_mid = compressed_mid.flatten(2).transpose(1, 2)
        tokens_high = compressed_high.flatten(2).transpose(1, 2)
        
        token0 = self.proj_low(tokens_low)
        token1 = self.proj_mid(tokens_mid)
        token2 = self.proj_high(tokens_high)
        
        return (token0, token1, token2), (spatial_low, spatial_mid, spatial_high)
    
    def get_band_info(self) -> dict:
        """Get current band partition information."""
        info = self.band_partition.get_band_info()
        info['token_counts'] = {'low': self.p_low, 'mid': self.p_mid, 'high': self.p_high}
        info['pool_sizes'] = self.pool_sizes
        return info
