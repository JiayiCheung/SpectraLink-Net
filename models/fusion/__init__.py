# models/fusion/__init__.py
"""
融合模块 (v4.1)

模块说明:
- CrossAttention: 标准交叉注意力，支持多token频域特征 [B, p, C]
- FreqGuidedAttentionPair: 频域引导注意力对
- FreqGuidedFusion: 频域引导融合模块
- SparseCrossAttention: 稀疏采样交叉注意力 (v4.0)
- SparseFreqGuidedAttentionPair: 稀疏版频域引导注意力对 (v4.0)
- AdaptiveSampleCalculator: 自适应采样数量计算器 (v4.1)
- ContentAwareSparseAttention: 基于内容的自适应稀疏注意力 (v4.1)
"""

from .cross_attention import (
    CrossAttention,
    FreqGuidedAttentionPair,
    CrossAttentionLegacy  # 兼容性接口
)
from .freq_guided_fusion import FreqGuidedFusion

# 稀疏采样注意力 (v4.0+)
from .cross_attention_sparse import (
    # 核心类
    SparseCrossAttention,
    SparseFreqGuidedAttentionPair,
    
    # 辅助类
    SparseImportanceScorer,
    AdaptiveSampleCalculator,
    ContentAwareSparseAttention,
    
    # 工厂函数
    get_attention_module,
)

__all__ = [
    # 标准注意力
    'CrossAttention',
    'FreqGuidedAttentionPair',
    'FreqGuidedFusion',
    'CrossAttentionLegacy',
    
    # 稀疏注意力 (v4.0+)
    'SparseCrossAttention',
    'SparseFreqGuidedAttentionPair',
    'SparseImportanceScorer',
    'AdaptiveSampleCalculator',
    'ContentAwareSparseAttention',
    'get_attention_module',
]
