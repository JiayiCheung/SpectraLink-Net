# models/decoder/decoder.py
"""U-Net style decoder with gradient checkpointing support."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple

from ..components import DoubleConv, UpsampleBlock


class DecoderBlock(nn.Module):
    """
    Single decoder block: Upsample -> Concat Skip -> DoubleConv
    
    Args:
        in_channels: Input channels
        skip_channels: Skip connection channels
        out_channels: Output channels
        use_checkpoint: Enable gradient checkpointing
    """

    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint

        self.upsample = UpsampleBlock(
            in_channels=in_channels,
            out_channels=in_channels // 2
        )

        concat_channels = in_channels // 2 + skip_channels
        self.conv = DoubleConv(
            in_channels=concat_channels,
            out_channels=out_channels
        )

    def _forward_impl(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x, target_size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, skip, use_reentrant=False)
        return self._forward_impl(x, skip)


class Decoder(nn.Module):
    """
    Multi-level decoder with skip connections.
    
    Args:
        bottleneck_channels: Bottleneck output channels
        skip_channels: Skip connection channels per level
        out_channels: Final output channels
        num_levels: Number of decoder levels
        use_checkpoint: Enable gradient checkpointing
    """

    def __init__(
            self,
            bottleneck_channels: int = 256,
            skip_channels: List[int] = None,
            out_channels: int = 32,
            num_levels: int = 3,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint

        if skip_channels is None:
            skip_channels = [32 * (2 ** i) for i in range(num_levels)]

        assert len(skip_channels) == num_levels

        self.skip_channels = skip_channels
        self.decoder_blocks = nn.ModuleList()

        current_ch = bottleneck_channels

        for level in reversed(range(num_levels)):
            skip_ch = skip_channels[level]
            out_ch = out_channels if level == 0 else skip_ch

            block = DecoderBlock(
                in_channels=current_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_checkpoint=use_checkpoint
            )
            self.decoder_blocks.append(block)
            current_ch = out_ch

        self.out_channels = out_channels

    def forward(
            self,
            bottleneck: torch.Tensor,
            skips: List[torch.Tensor]
    ) -> torch.Tensor:
        x = bottleneck

        for i, block in enumerate(self.decoder_blocks):
            skip_idx = self.num_levels - 1 - i
            x = block(x, skips[skip_idx])

        return x

    def forward_with_intermediates(
            self,
            bottleneck: torch.Tensor,
            skips: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning intermediate features for deep supervision."""
        x = bottleneck
        intermediates = []

        for i, block in enumerate(self.decoder_blocks):
            skip_idx = self.num_levels - 1 - i
            x = block(x, skips[skip_idx])
            intermediates.append(x)

        return x, intermediates

    def set_checkpoint(self, enabled: bool):
        self.use_checkpoint = enabled
        for block in self.decoder_blocks:
            block.use_checkpoint = enabled
