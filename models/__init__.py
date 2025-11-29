# models/__init__.py
"""
SpectraLink-Net 模型包

主要组件：
- SpectraLinkNet: 主模型
- Encoder: 编码器（双分支 + 融合）
- Decoder: 解码器
- SegmentationHead: 分割头
"""

from .spectralink_net import SpectraLinkNet, build_spectralink_net

# 子模块
from . import components
from . import spatial_branch
from . import frequency_branch
from . import fusion
from . import encoder
from . import decoder
from . import heads

__all__ = [
    # 主模型
    'SpectraLinkNet',
    'build_spectralink_net',
    # 子模块
    'components',
    'spatial_branch',
    'frequency_branch',
    'fusion',
    'encoder',
    'decoder',
    'heads',
]