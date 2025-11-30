# models/fusion/__init__.py
"""
融合模块 (v2.0)

修改说明:
- CrossAttention: 支持多token频域特征 [B, p, C]
- FreqGuidedAttentionPair: 适配多token版本
- FreqGuidedFusion: 适配多token版本
- 保留 CrossAttentionLegacy 用于向后兼容
"""

from .cross_attention import (
    CrossAttention,
    FreqGuidedAttentionPair,
    CrossAttentionLegacy  # 兼容性接口
)
from .freq_guided_fusion import FreqGuidedFusion

__all__ = [
    'CrossAttention',
    'FreqGuidedAttentionPair',
    'FreqGuidedFusion',
    'CrossAttentionLegacy',
]