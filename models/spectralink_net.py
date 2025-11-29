# models/spectralink_net.py
"""
SpectraLink-Net 主模型
频域-空域融合的3D血管分割网络
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

from .components import InitConv
from .encoder import Encoder
from .decoder import Decoder
from .heads import SegmentationHead


class SpectraLinkNet(nn.Module):
	"""
	SpectraLink-Net: 频域引导的多尺度血管分割网络

	核心创新：
	1. 频域-空域双分支架构：并行提取互补特征
	2. 可学习频带划分：自适应分解3D频谱为3个频带
	3. 频域引导注意力融合：低频→大尺度，高频→小尺度的显式对应

	网络结构：
	```
	Input [B, 1, 64, 128, 128]
		↓ InitConv (1→32)
	[B, 32, 64, 128, 128]
		↓ Encoder (3层 EncoderBlock + Bottleneck)
			Level 0: → skip0 [32], → [64]
			Level 1: → skip1 [64], → [128]
			Level 2: → skip2 [128], → [256]
			Bottleneck: [256]
		↓ Decoder (3层 DecoderBlock)
			Level 2: [256] + skip2 → [128]
			Level 1: [128] + skip1 → [64]
			Level 0: [64] + skip0 → [32]
	[B, 32, 64, 128, 128]
		↓ SegmentationHead (32→16→1)
	Output [B, 1, 64, 128, 128] (logits)
	```

	参数:
		in_channels: 输入通道数（默认1，单模态CT）
		out_channels: 输出通道数（默认1，二分类）
		base_channels: 基础通道数（默认32）
		num_levels: 编码器/解码器层数（默认3）
		freq_token_base: 频域token基础通道数（默认4）
		expansion_ratio: ComplexBlock扩展比例（默认2）
		dropout: dropout比例（默认0.0）

	输入:
		x: [B, in_channels, D, H, W]

	输出:
		logits: [B, out_channels, D, H, W]
	"""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: int = 1,
			base_channels: int = 32,
			num_levels: int = 3,
			freq_token_base: int = 4,
			expansion_ratio: int = 2,
			dropout: float = 0.0
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.base_channels = base_channels
		self.num_levels = num_levels
		
		# 1. 初始卷积
		self.init_conv = InitConv(
			in_channels=in_channels,
			out_channels=base_channels
		)
		
		# 2. 编码器
		self.encoder = Encoder(
			in_channels=base_channels,
			base_channels=base_channels,
			freq_token_channels=freq_token_base,
			num_levels=num_levels,
			expansion_ratio=expansion_ratio,
			dropout=dropout
		)
		
		# 3. 解码器
		# skip通道数: [32, 64, 128] for num_levels=3
		skip_channels = [base_channels * (2 ** i) for i in range(num_levels)]
		
		self.decoder = Decoder(
			bottleneck_channels=self.encoder.bottleneck_channels,
			skip_channels=skip_channels,
			out_channels=base_channels,
			num_levels=num_levels
		)
		
		# 4. 分割头
		self.seg_head = SegmentationHead(
			in_channels=base_channels,
			mid_channels=base_channels // 2,
			out_channels=out_channels
		)
		
		# 初始化权重
		self._init_weights()
	
	def _init_weights(self):
		"""权重初始化"""
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
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播

		参数:
			x: 输入 [B, in_channels, D, H, W]

		返回:
			logits: [B, out_channels, D, H, W]
		"""
		# 1. 初始卷积
		x = self.init_conv(x)
		
		# 2. 编码器
		skips, bottleneck = self.encoder(x)
		
		# 3. 解码器
		x = self.decoder(bottleneck, skips)
		
		# 4. 分割头
		logits = self.seg_head(x)
		
		return logits
	
	def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""
		前向传播，返回中间特征（用于可视化/分析）

		返回:
			包含各阶段特征的字典
		"""
		features = {}
		
		# 初始卷积
		x = self.init_conv(x)
		features['init_conv'] = x
		
		# 编码器
		skips, bottleneck = self.encoder(x)
		features['skips'] = skips
		features['bottleneck'] = bottleneck
		
		# 解码器
		x = self.decoder(bottleneck, skips)
		features['decoder_out'] = x
		
		# 分割头
		logits = self.seg_head(x)
		features['logits'] = logits
		
		return features
	
	def get_model_info(self) -> Dict:
		"""获取模型信息"""
		total_params = sum(p.numel() for p in self.parameters())
		trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		
		return {
			'model_name': 'SpectraLinkNet',
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'base_channels': self.base_channels,
			'num_levels': self.num_levels,
			'total_parameters': total_params,
			'trainable_parameters': trainable_params,
			'parameters_mb': total_params * 4 / (1024 * 1024),  # float32
		}
	
	def save_checkpoint(
			self,
			filepath: Union[str, Path],
			optimizer: Optional[torch.optim.Optimizer] = None,
			scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
			epoch: Optional[int] = None,
			best_dice: Optional[float] = None,
			**kwargs
	):
		"""
		保存检查点

		参数:
			filepath: 保存路径
			optimizer: 优化器（可选）
			scheduler: 学习率调度器（可选）
			epoch: 当前epoch（可选）
			best_dice: 最佳Dice分数（可选）
			**kwargs: 其他要保存的信息
		"""
		checkpoint = {
			'model_state_dict': self.state_dict(),
			'model_config': {
				'in_channels': self.in_channels,
				'out_channels': self.out_channels,
				'base_channels': self.base_channels,
				'num_levels': self.num_levels,
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
		"""
		从检查点加载模型

		参数:
			filepath: 检查点路径
			device: 目标设备
			strict: 是否严格匹配参数

		返回:
			(model, checkpoint_info)
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		checkpoint = torch.load(filepath, map_location=device, weights_only=False)
		
		# 从配置创建模型
		config = checkpoint.get('model_config', {})
		model = cls(
			in_channels=config.get('in_channels', 1),
			out_channels=config.get('out_channels', 1),
			base_channels=config.get('base_channels', 32),
			num_levels=config.get('num_levels', 3),
		)
		
		# 加载权重
		model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
		model.to(device)
		
		# 返回检查点信息
		checkpoint_info = {
			'epoch': checkpoint.get('epoch'),
			'best_dice': checkpoint.get('best_dice'),
			'model_info': checkpoint.get('model_info'),
		}
		
		return model, checkpoint_info


def build_spectralink_net(config: Dict = None) -> SpectraLinkNet:
	"""
	根据配置构建SpectraLinkNet

	参数:
		config: 配置字典，包含模型参数

	返回:
		SpectraLinkNet实例
	"""
	if config is None:
		config = {}
	
	model_config = config.get('model', {})
	
	return SpectraLinkNet(
		in_channels=model_config.get('in_channels', 1),
		out_channels=model_config.get('out_channels', 1),
		base_channels=model_config.get('base_channels', 32),
		num_levels=model_config.get('num_levels', 3),
		freq_token_base=model_config.get('freq_token_base', 4),
		expansion_ratio=model_config.get('expansion_ratio', 2),
		dropout=model_config.get('dropout', 0.0),
	)