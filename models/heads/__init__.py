# models/heads/__init__.py
"""
分割头模块
"""

from .seg_head import SegmentationHead, MultiScaleSegmentationHead

__all__ = [
    'SegmentationHead',
    'MultiScaleSegmentationHead',
]