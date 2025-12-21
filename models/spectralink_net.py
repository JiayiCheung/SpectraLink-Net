# models/spectralink_net.py
"""
SpectraLink-Net: Frequency-guided 3D vessel segmentation network.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path

from .components import InitConv
from .encoder import Encoder
from .decoder import Decoder
from .heads import SegmentationHead


class SpectraLinkNet(nn.Module):
    """
    SpectraLink-Net for 3D medical image segmentation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base channel count
        num_levels: Number of encoder/decoder levels
        freq_token_base: Base channels for frequency tokens
        expansion_ratio: Expansion ratio in ComplexBlock
        num_heads: Number of attention heads
        dropout: Dropout rate
        pool_sizes: Pooling sizes for frequency branch
        use_checkpoint: Enable gradient checkpointing
        deep_supervision: Enable deep supervision
        use_sparse_attention: Enable sparse attention
        sample_mode: Sampling mode ('fixed', 'ratio', 'auto')
        num_samples: Fixed sample count (mode='fixed')
        sample_ratio: Sample ratio (mode='ratio')
        sparse_levels: Levels using sparse attention (None=all)
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            base_channels: int = 32,
            num_levels: int = 3,
            freq_token_base: int = 4,
            expansion_ratio: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None,
            use_checkpoint: bool = False,
            deep_supervision: bool = False,
            use_sparse_attention: bool = False,
            sample_mode: str = 'auto',
            num_samples: Optional[int] = None,
            sample_ratio: Optional[float] = None,
            sparse_levels: Optional[List[int]] = None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.pool_sizes = pool_sizes
        self.deep_supervision = deep_supervision
        self.use_sparse_attention = use_sparse_attention
        self.sample_mode = sample_mode
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio
        self.sparse_levels = sparse_levels

        # Initial convolution
        self.init_conv = InitConv(
            in_channels=in_channels,
            out_channels=base_channels
        )

        # Encoder
        self.encoder = Encoder(
            in_channels=base_channels,
            base_channels=base_channels,
            freq_token_channels=freq_token_base,
            num_levels=num_levels,
            expansion_ratio=expansion_ratio,
            num_heads=num_heads,
            dropout=dropout,
            pool_sizes=pool_sizes,
            use_checkpoint=use_checkpoint,
            use_sparse_attention=use_sparse_attention,
            sample_mode=sample_mode,
            num_samples=num_samples,
            sample_ratio=sample_ratio,
            sparse_levels=sparse_levels
        )

        # Decoder
        skip_channels = [base_channels * (2 ** i) for i in range(num_levels)]
        self.decoder = Decoder(
            bottleneck_channels=self.encoder.bottleneck_channels,
            skip_channels=skip_channels,
            out_channels=base_channels,
            num_levels=num_levels,
            use_checkpoint=use_checkpoint
        )

        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=base_channels,
            mid_channels=base_channels // 2,
            out_channels=out_channels
        )

        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for level in range(num_levels - 1):
                ds_in_channels = base_channels * (2 ** (num_levels - 1 - level))
                self.ds_heads.append(
                    nn.Conv3d(ds_in_channels, out_channels, kernel_size=1, bias=True)
                )
        else:
            self.ds_heads = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            List[Tensor] if deep_supervision and training, else single Tensor
        """
        x = self.init_conv(x)
        skips, bottleneck = self.encoder(x)
        
        if self.deep_supervision and self.training:
            x, intermediates = self.decoder.forward_with_intermediates(bottleneck, skips)
            
            logits = self.seg_head(x)
            outputs = [logits]
            
            for i, head in enumerate(self.ds_heads):
                ds_logits = head(intermediates[i])
                outputs.append(ds_logits)
            
            return outputs
        else:
            x = self.decoder(bottleneck, skips)
            return self.seg_head(x)

    def set_checkpoint(self, enabled: bool):
        """Toggle gradient checkpointing."""
        self.use_checkpoint = enabled
        self.encoder.set_checkpoint(enabled)
        self.decoder.set_checkpoint(enabled)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning intermediate features."""
        features = {}

        x = self.init_conv(x)
        features['init_conv'] = x

        skips, bottleneck = self.encoder(x)
        features['skips'] = skips
        features['bottleneck'] = bottleneck

        x = self.decoder(bottleneck, skips)
        features['decoder_out'] = x

        logits = self.seg_head(x)
        features['logits'] = logits

        return features

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_name': 'SpectraLinkNet',
            'version': 'v4.2',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'base_channels': self.base_channels,
            'num_levels': self.num_levels,
            'pool_sizes': self.pool_sizes,
            'use_checkpoint': self.use_checkpoint,
            'deep_supervision': self.deep_supervision,
            'use_sparse_attention': self.use_sparse_attention,
            'sample_mode': self.sample_mode if self.use_sparse_attention else 'N/A',
            'num_samples': self.num_samples,
            'sample_ratio': self.sample_ratio,
            'sparse_levels': self.sparse_levels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_mb': total_params * 4 / (1024 * 1024),
        }
        
        if hasattr(self.encoder, 'get_attention_info'):
            info['attention_info'] = self.encoder.get_attention_info()
        
        return info

    def save_checkpoint(
            self,
            filepath: Union[str, Path],
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            epoch: Optional[int] = None,
            best_dice: Optional[float] = None,
            **kwargs
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'base_channels': self.base_channels,
                'num_levels': self.num_levels,
                'pool_sizes': self.pool_sizes,
                'use_checkpoint': self.use_checkpoint,
                'deep_supervision': self.deep_supervision,
                'use_sparse_attention': self.use_sparse_attention,
                'sample_mode': self.sample_mode,
                'num_samples': self.num_samples,
                'sample_ratio': self.sample_ratio,
                'sparse_levels': self.sparse_levels,
            },
            'model_info': self.get_model_info(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_dice is not None:
            checkpoint['best_dice'] = best_dice

        checkpoint.update(kwargs)
        torch.save(checkpoint, filepath)

    @classmethod
    def load_from_checkpoint(
            cls,
            filepath: Union[str, Path],
            device: Optional[torch.device] = None,
            strict: bool = True
    ) -> Tuple['SpectraLinkNet', Dict]:
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        config = checkpoint.get('model_config', {})
        model = cls(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            base_channels=config.get('base_channels', 32),
            num_levels=config.get('num_levels', 3),
            pool_sizes=config.get('pool_sizes', None),
            use_checkpoint=config.get('use_checkpoint', False),
            deep_supervision=config.get('deep_supervision', False),
            use_sparse_attention=config.get('use_sparse_attention', False),
            sample_mode=config.get('sample_mode', 'auto'),
            num_samples=config.get('num_samples', None),
            sample_ratio=config.get('sample_ratio', None),
            sparse_levels=config.get('sparse_levels', None),
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)

        checkpoint_info = {
            'epoch': checkpoint.get('epoch'),
            'best_dice': checkpoint.get('best_dice'),
            'model_info': checkpoint.get('model_info'),
        }

        return model, checkpoint_info


def build_spectralink_net(config: Dict = None) -> SpectraLinkNet:
    """Build SpectraLinkNet from config dict."""
    if config is None:
        config = {}

    model_config = config.get('model', {})

    pool_sizes_cfg = model_config.get('pool_sizes', None)
    if pool_sizes_cfg is not None:
        pool_sizes = tuple(tuple(ps) for ps in pool_sizes_cfg)
    else:
        pool_sizes = None
    
    sparse_levels_cfg = model_config.get('sparse_levels', None)
    if sparse_levels_cfg is not None:
        sparse_levels = list(sparse_levels_cfg)
    else:
        sparse_levels = None

    return SpectraLinkNet(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 1),
        base_channels=model_config.get('base_channels', 32),
        num_levels=model_config.get('num_levels', 3),
        freq_token_base=model_config.get('freq_token_base', 4),
        expansion_ratio=model_config.get('expansion_ratio', 2),
        num_heads=model_config.get('num_heads', 4),
        dropout=model_config.get('dropout', 0.0),
        pool_sizes=pool_sizes,
        use_checkpoint=model_config.get('use_checkpoint', False),
        deep_supervision=model_config.get('deep_supervision', False),
        use_sparse_attention=model_config.get('use_sparse_attention', False),
        sample_mode=model_config.get('sample_mode', 'auto'),
        num_samples=model_config.get('num_samples', None),
        sample_ratio=model_config.get('sample_ratio', None),
        sparse_levels=sparse_levels,
    )
