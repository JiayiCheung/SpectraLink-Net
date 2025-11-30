# models/encoder/encoder_block.py
"""
编码器块模块
整合空域分支、频域分支和融合模块

修改说明 (v2.0):
- 适配新的 ImprovedFrequencyBranch 输出：[B, p, C] token序列
- 适配新的 FreqGuidedFusion 接口
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..spatial_branch import EnhancedSpatialBranch
from ..frequency_branch import ImprovedFrequencyBranch
from ..fusion import FreqGuidedFusion
from ..components import DownsampleBlock


class EncoderBlock(nn.Module):
    """
    编码器块 (v2.0)

    单层编码器，整合双分支特征提取和融合：
    1. EnhancedSpatialBranch: 提取3个空域path特征
    2. ImprovedFrequencyBranch: 提取3个频域token序列（多token版本）
    3. FreqGuidedFusion: 频域引导空域融合（真正的注意力选择）
    4. 输出 skip connection 特征

    结构：
    Input → SpatialBranch → (path1, path2, path3)
          → FrequencyBranch → (token0, token1, token2) 各为 [B, p, C]
          → FreqGuidedFusion → fused_feature (skip)
          → Downsample → output (to next level)

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数（融合后的通道数，也是skip的通道数）
        freq_token_channels: 频域token通道数
        expansion_ratio: ComplexBlock的扩展比例
        num_heads: 注意力头数（新增参数）
        dropout: dropout比例
        downsample: 是否下采样（最后一层可能不需要）
        pool_sizes: 频域分支的池化尺寸（新增参数）

    输入:
        x: [B, in_channels, D, H, W]

    输出:
        (skip, out):
        - skip: [B, out_channels, D, H, W] 用于decoder的skip connection
        - out: [B, out_channels*2, D//2, H//2, W//2] 下采样后传给下一层
               如果downsample=False，则out=None
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
            pool_sizes: Optional[Tuple[Tuple[int, int, int], ...]] = None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_flag = downsample

        # 空域分支：输出3个path
        # 通道分配：1/2, 1/4, 1/4
        self.spatial_branch = EnhancedSpatialBranch(
            in_channels=in_channels,
            out_channels=out_channels
        )

        # 频域分支：输出3个token序列（多token版本）
        self.frequency_branch = ImprovedFrequencyBranch(
            in_channels=in_channels,
            token_channels=freq_token_channels,
            expansion_ratio=expansion_ratio,
            pool_sizes=pool_sizes  # 新增参数
        )

        # 获取空域各path的通道数
        spatial_info = self.spatial_branch.get_channel_info()
        ch1 = spatial_info['path1_channels']  # 3×3×3
        ch2 = spatial_info['path2_channels']  # 5×5×5
        ch3 = spatial_info['path3_channels']  # 7×7×7

        # 融合模块（多token版本）
        self.fusion = FreqGuidedFusion(
            spatial_channels=[ch1, ch2, ch3],
            freq_channels=freq_token_channels,
            out_channels=out_channels,
            num_heads=num_heads,  # 新增参数
            dropout=dropout
        )

        # 下采样模块（可选）
        if downsample:
            self.downsample = DownsampleBlock(
                in_channels=out_channels,
                out_channels=out_channels * 2
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            x: 输入特征 [B, in_channels, D, H, W]

        返回:
            (skip, out):
            - skip: [B, out_channels, D, H, W] skip connection特征
            - out: [B, out_channels*2, D//2, H//2, W//2] 或 None
        """
        # 1. 空域分支
        path1, path2, path3 = self.spatial_branch(x)

        # 2. 频域分支（输出token序列）
        token0, token1, token2 = self.frequency_branch(x)
        # token0: [B, p_low, freq_ch]
        # token1: [B, p_mid, freq_ch]
        # token2: [B, p_high, freq_ch]

        # 3. 频域引导融合
        fused = self.fusion(
            spatial_paths=(path1, path2, path3),
            freq_tokens=(token0, token1, token2)
        )

        # 4. skip connection 输出
        skip = fused

        # 5. 下采样（如果需要）
        if self.downsample is not None:
            out = self.downsample(fused)
        else:
            out = None

        return skip, out

    def forward_with_details(self, x: torch.Tensor) -> dict:
        """
        前向传播，返回详细中间结果（用于可视化/分析）

        返回:
            包含所有中间特征的字典
        """
        # 空域分支
        path1, path2, path3 = self.spatial_branch(x)

        # 频域分支（带空域特征）
        (token0, token1, token2), (spatial_low, spatial_mid, spatial_high) = \
            self.frequency_branch.forward_with_spatial(x)

        # 融合（带详细输出）
        fused, fusion_details = self.fusion.forward_with_details(
            spatial_paths=(path1, path2, path3),
            freq_tokens=(token0, token1, token2)
        )

        skip = fused
        out = self.downsample(fused) if self.downsample is not None else None

        return {
            # 空域分支
            'path1': path1,
            'path2': path2,
            'path3': path3,
            # 频域分支（token序列）
            'token0': token0,  # [B, p_low, C]
            'token1': token1,  # [B, p_mid, C]
            'token2': token2,  # [B, p_high, C]
            'freq_spatial_low': spatial_low,
            'freq_spatial_mid': spatial_mid,
            'freq_spatial_high': spatial_high,
            # 融合
            'fusion_details': fusion_details,
            'fused': fused,
            # 输出
            'skip': skip,
            'out': out,
            # 频带信息
            'band_info': self.frequency_branch.get_band_info()
        }


class Encoder(nn.Module):
    """
    完整编码器 (v2.0)

    包含3层EncoderBlock + Bottleneck

    结构：
    Level 0: [B, 32, 64, 128, 128] → skip0, [B, 64, 32, 64, 64]
    Level 1: [B, 64, 32, 64, 64] → skip1, [B, 128, 16, 32, 32]
    Level 2: [B, 128, 16, 32, 32] → skip2, [B, 256, 8, 16, 16]
    Bottleneck: Conv [B, 256, 8, 16, 16]

    参数:
        in_channels: 初始输入通道数（InitConv输出，默认32）
        base_channels: 基础通道数（默认32）
        freq_token_channels: 频域token基础通道数（默认4，逐层翻倍）
        num_levels: 编码器层数（默认3）
        expansion_ratio: ComplexBlock扩展比例
        num_heads: 注意力头数（新增参数）
        dropout: dropout比例
    """

    def __init__(
            self,
            in_channels: int = 32,
            base_channels: int = 32,
            freq_token_channels: int = 4,
            num_levels: int = 3,
            expansion_ratio: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0
    ):
        super().__init__()

        self.num_levels = num_levels

        # 构建各层EncoderBlock
        self.encoder_blocks = nn.ModuleList()

        current_ch = in_channels
        for level in range(num_levels):
            # 通道配置
            out_ch = base_channels * (2 ** level)  # 32, 64, 128
            freq_ch = freq_token_channels * (2 ** level)  # 4, 8, 16

            # 是否下采样（所有层都下采样）
            do_downsample = True

            block = EncoderBlock(
                in_channels=current_ch,
                out_channels=out_ch,
                freq_token_channels=freq_ch,
                expansion_ratio=expansion_ratio,
                num_heads=num_heads,
                dropout=dropout,
                downsample=do_downsample
            )
            self.encoder_blocks.append(block)

            # 更新下一层输入通道数（下采样后翻倍）
            current_ch = out_ch * 2

        # Bottleneck
        bottleneck_in = base_channels * (2 ** (num_levels - 1)) * 2  # 128 * 2 = 256
        self.bottleneck = nn.Sequential(
            nn.Conv3d(bottleneck_in, bottleneck_in, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_in, affine=True),
            nn.ReLU(inplace=True)
        )

        self.bottleneck_channels = bottleneck_in

    def forward(self, x: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入特征 [B, in_channels, D, H, W]

        返回:
            (skips, bottleneck_out):
            - skips: [skip0, skip1, skip2] 各层skip connection
            - bottleneck_out: bottleneck输出
        """
        skips = []
        current = x

        for block in self.encoder_blocks:
            skip, current = block(current)
            skips.append(skip)

        bottleneck_out = self.bottleneck(current)

        return skips, bottleneck_out