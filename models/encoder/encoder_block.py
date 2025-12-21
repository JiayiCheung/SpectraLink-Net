# models/encoder/encoder_block.py
"""Encoder block with frequency-spatial fusion."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List

from ..spatial_branch import EnhancedSpatialBranch
from ..frequency_branch import ImprovedFrequencyBranch
from ..fusion import FreqGuidedFusion
from ..fusion.cross_attention_sparse import SparseFreqGuidedAttentionPair
from ..components import DownsampleBlock


class FreqGuidedFusionSparse(nn.Module):
    """Sparse frequency-guided fusion module."""
    
    def __init__(
        self,
        spatial_channels: list,
        freq_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        sample_mode: str = 'auto',
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 512,
        max_samples: int = 16384
    ):
        super().__init__()
        
        assert len(spatial_channels) == 3
        
        self.spatial_channels = spatial_channels
        self.freq_channels = freq_channels
        self.out_channels = out_channels
        self.sample_mode = sample_mode
        
        ch1, ch2, ch3 = spatial_channels
        
        sparse_kwargs = {
            'sample_mode': sample_mode,
            'num_samples': num_samples,
            'sample_ratio': sample_ratio,
            'min_samples': min_samples,
            'max_samples': max_samples,
        }
        
        self.attn_pair_0 = SparseFreqGuidedAttentionPair(
            spatial_channels=ch3, freq_channels=freq_channels,
            num_heads=num_heads, dropout=dropout, **sparse_kwargs
        )
        self.attn_pair_1 = SparseFreqGuidedAttentionPair(
            spatial_channels=ch2, freq_channels=freq_channels,
            num_heads=num_heads, dropout=dropout, **sparse_kwargs
        )
        self.attn_pair_2 = SparseFreqGuidedAttentionPair(
            spatial_channels=ch1, freq_channels=freq_channels,
            num_heads=num_heads, dropout=dropout, **sparse_kwargs
        )
        
        concat_channels = ch1 + ch2 + ch3
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        spatial_paths: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        freq_tokens: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        path1, path2, path3 = spatial_paths
        token0, token1, token2 = freq_tokens
        
        enhanced_path3 = self.attn_pair_0(path3, token0)
        enhanced_path2 = self.attn_pair_1(path2, token1)
        enhanced_path1 = self.attn_pair_2(path1, token2)
        
        concat = torch.cat([enhanced_path1, enhanced_path2, enhanced_path3], dim=1)
        return self.fusion_conv(concat)


class EncoderBlock(nn.Module):
    """
    Encoder block with frequency-spatial dual branches.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        freq_token_channels: Frequency token channels
        expansion_ratio: ComplexBlock expansion ratio
        num_heads: Number of attention heads
        dropout: Dropout rate
        downsample: Enable downsampling
        pool_sizes: Frequency branch pool sizes
        use_checkpoint: Enable gradient checkpointing
        use_sparse_attention: Enable sparse attention
        sample_mode: Sparse sampling mode
        num_samples: Fixed sample count
        sample_ratio: Sample ratio
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            freq_token_channels: int,
            expansion_ratio: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            downsample: bool = True,
            pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None,
            use_checkpoint: bool = False,
            use_sparse_attention: bool = False,
            sample_mode: str = 'auto',
            num_samples: Optional[int] = None,
            sample_ratio: Optional[float] = None,
            min_samples: int = 512,
            max_samples: int = 16384
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_flag = downsample
        self.use_checkpoint = use_checkpoint
        self.use_sparse_attention = use_sparse_attention
        self.sample_mode = sample_mode

        # Spatial branch
        self.spatial_branch = EnhancedSpatialBranch(
            in_channels=in_channels,
            out_channels=out_channels
        )

        # Frequency branch
        self.frequency_branch = ImprovedFrequencyBranch(
            in_channels=in_channels,
            token_channels=freq_token_channels,
            expansion_ratio=expansion_ratio,
            pool_sizes=pool_sizes
        )

        # Get spatial path channels
        spatial_info = self.spatial_branch.get_channel_info()
        ch1, ch2, ch3 = spatial_info['path1_channels'], spatial_info['path2_channels'], spatial_info['path3_channels']

        # Fusion module
        if use_sparse_attention:
            self.fusion = FreqGuidedFusionSparse(
                spatial_channels=[ch1, ch2, ch3],
                freq_channels=freq_token_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                sample_mode=sample_mode,
                num_samples=num_samples,
                sample_ratio=sample_ratio,
                min_samples=min_samples,
                max_samples=max_samples
            )
        else:
            self.fusion = FreqGuidedFusion(
                spatial_channels=[ch1, ch2, ch3],
                freq_channels=freq_token_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                dropout=dropout
            )

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Residual scale factor
        self.gamma = nn.Parameter(torch.ones(1) * 1e-5)
        
        # Downsample
        if downsample:
            self.downsample = DownsampleBlock(
                in_channels=out_channels,
                out_channels=out_channels * 2
            )
        else:
            self.downsample = None

    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        identity = self.shortcut(x)
        
        path1, path2, path3 = self.spatial_branch(x)
        token0, token1, token2 = self.frequency_branch(x)

        fused = self.fusion(
            spatial_paths=(path1, path2, path3),
            freq_tokens=(token0, token1, token2)
        )

        skip = identity + self.gamma * fused

        if self.downsample is not None:
            out = self.downsample(skip)
        else:
            out = None

        return skip, out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_checkpoint and self.training:
            skip, out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            skip, out = self._forward_impl(x)
        return skip, out

    def forward_with_details(self, x: torch.Tensor) -> dict:
        """Forward with detailed intermediate outputs."""
        identity = self.shortcut(x)
        
        path1, path2, path3 = self.spatial_branch(x)

        (token0, token1, token2), (spatial_low, spatial_mid, spatial_high) = \
            self.frequency_branch.forward_with_spatial(x)

        if hasattr(self.fusion, 'forward_with_details'):
            fused, fusion_details = self.fusion.forward_with_details(
                spatial_paths=(path1, path2, path3),
                freq_tokens=(token0, token1, token2)
            )
        else:
            fused = self.fusion(
                spatial_paths=(path1, path2, path3),
                freq_tokens=(token0, token1, token2)
            )
            fusion_details = {'sparse_attention': self.use_sparse_attention, 'sample_mode': self.sample_mode}

        skip = identity + self.gamma * fused
        out = self.downsample(skip) if self.downsample is not None else None

        return {
            'path1': path1, 'path2': path2, 'path3': path3,
            'token0': token0, 'token1': token1, 'token2': token2,
            'freq_spatial_low': spatial_low, 'freq_spatial_mid': spatial_mid, 'freq_spatial_high': spatial_high,
            'fusion_details': fusion_details, 'fused': fused, 'skip': skip, 'out': out,
            'band_info': self.frequency_branch.get_band_info()
        }


class Encoder(nn.Module):
    """
    Multi-level encoder.
    
    Args:
        in_channels: Input channels
        base_channels: Base channel count
        freq_token_channels: Frequency token base channels
        num_levels: Number of levels
        expansion_ratio: ComplexBlock expansion ratio
        num_heads: Number of attention heads
        dropout: Dropout rate
        pool_sizes: Frequency branch pool sizes
        use_checkpoint: Enable gradient checkpointing
        use_sparse_attention: Enable sparse attention
        sample_mode: Sparse sampling mode
        num_samples: Fixed sample count
        sample_ratio: Sample ratio
        sparse_levels: Levels using sparse attention
    """

    def __init__(
            self,
            in_channels: int = 32,
            base_channels: int = 32,
            freq_token_channels: int = 4,
            num_levels: int = 3,
            expansion_ratio: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None,
            use_checkpoint: bool = False,
            use_sparse_attention: bool = False,
            sample_mode: str = 'auto',
            num_samples: Optional[int] = None,
            sample_ratio: Optional[float] = None,
            sparse_levels: Optional[List[int]] = None
    ):
        super().__init__()

        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.use_sparse_attention = use_sparse_attention
        self.sample_mode = sample_mode

        if sparse_levels is None:
            sparse_levels = list(range(num_levels)) if use_sparse_attention else []
        self.sparse_levels = sparse_levels

        self.encoder_blocks = nn.ModuleList()

        current_ch = in_channels
        for level in range(num_levels):
            out_ch = base_channels * (2 ** level)
            freq_ch = freq_token_channels * (2 ** level)
            use_sparse = (level in sparse_levels)

            block = EncoderBlock(
                in_channels=current_ch,
                out_channels=out_ch,
                freq_token_channels=freq_ch,
                expansion_ratio=expansion_ratio,
                num_heads=num_heads,
                dropout=dropout,
                downsample=True,
                pool_sizes=pool_sizes,
                use_checkpoint=use_checkpoint,
                use_sparse_attention=use_sparse,
                sample_mode=sample_mode,
                num_samples=num_samples,
                sample_ratio=sample_ratio
            )
            self.encoder_blocks.append(block)
            current_ch = out_ch * 2

        # Bottleneck
        bottleneck_in = base_channels * (2 ** (num_levels - 1)) * 2
        self.bottleneck = nn.Sequential(
            nn.Conv3d(bottleneck_in, bottleneck_in, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_in, affine=True),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_channels = bottleneck_in

    def forward(self, x: torch.Tensor) -> Tuple[list, torch.Tensor]:
        skips = []
        current = x

        for block in self.encoder_blocks:
            skip, current = block(current)
            skips.append(skip)

        bottleneck_out = self.bottleneck(current)
        return skips, bottleneck_out

    def set_checkpoint(self, enabled: bool):
        self.use_checkpoint = enabled
        for block in self.encoder_blocks:
            block.use_checkpoint = enabled
    
    def get_attention_info(self) -> dict:
        info = {}
        for i, block in enumerate(self.encoder_blocks):
            info[f'level_{i}'] = {
                'use_sparse': block.use_sparse_attention,
                'sample_mode': block.sample_mode if block.use_sparse_attention else 'N/A'
            }
        return info
