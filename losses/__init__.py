# losses/__init__.py
"""
损失函数模块

包含：
- DiceLoss: 标准Dice损失
- SoftDiceLoss: 软Dice损失
- GeneralizedDiceLoss: 广义Dice损失
- ClDiceLoss: 中心线Dice损失（拓扑保持）
- FocalLoss: Focal损失
- CombinedLoss: 组合损失
- DiceBCELoss: Dice + BCE损失
"""

from .dice_loss import DiceLoss, SoftDiceLoss, GeneralizedDiceLoss
from .cldice_loss import ClDiceLoss, DiceClDiceLoss, soft_skeleton
from .combined_loss import (
    BCEWithLogitsLoss,
    FocalLoss,
    CombinedLoss,
    DiceBCELoss,
    build_loss
)

__all__ = [
    # Dice系列
    'DiceLoss',
    'SoftDiceLoss',
    'GeneralizedDiceLoss',
    # clDice系列
    'ClDiceLoss',
    'DiceClDiceLoss',
    'soft_skeleton',
    # 其他损失
    'BCEWithLogitsLoss',
    'FocalLoss',
    # 组合损失
    'CombinedLoss',
    'DiceBCELoss',
    # 工厂函数
    'build_loss',
]