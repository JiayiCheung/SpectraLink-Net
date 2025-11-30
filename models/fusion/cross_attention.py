# models/fusion/cross_attention.py
"""
交叉注意力模块
用于频域-空域特征融合

修改说明 (v2.0):
- 原版：freq_token [B, C_fd] 只有1个token，softmax无选择性
- 新版：freq_tokens [B, p, C_fd] 有p个token，实现真正的注意力选择
- 使用 F.scaled_dot_product_attention 支持Flash Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossAttention(nn.Module):
	"""
	交叉注意力模块 (v2.0)

	实现频域引导空域的注意力机制：
	- Q (Query): 来自空域特征 [B, N, C_sp]
	- K (Key): 来自频域tokens [B, p, C_fd]
	- V (Value): 来自频域tokens [B, p, C_fd]

	注意力计算：
	Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
	复杂度: O(N * p)，当 p << N 时非常高效

	参数:
		spatial_channels: 空域特征通道数（Q的维度）
		freq_channels: 频域token通道数（K/V的维度）
		num_heads: 注意力头数
		dropout: dropout比例
		use_flash_attn: 是否使用Flash Attention（PyTorch 2.0+自动启用）

	输入:
		spatial_tokens: 空域token序列 [B, N, C_sp]
		freq_tokens: 频域token序列 [B, p, C_fd]

	输出:
		attended: 注意力输出 [B, N, C_sp]
	"""
	
	def __init__(
			self,
			spatial_channels: int,
			freq_channels: int,
			num_heads: int = 4,
			dropout: float = 0.0,
			use_flash_attn: bool = True
	):
		super().__init__()
		
		self.spatial_channels = spatial_channels
		self.freq_channels = freq_channels
		self.num_heads = num_heads
		self.use_flash_attn = use_flash_attn
		
		# 使用空域通道数作为d_model
		self.d_model = spatial_channels
		self.head_dim = self.d_model // num_heads
		
		assert self.d_model % num_heads == 0, \
			f"d_model({self.d_model})必须能被num_heads({num_heads})整除"
		
		# Q投影：空域特征 → d_model
		self.q_proj = nn.Linear(spatial_channels, self.d_model)
		
		# K/V投影：频域tokens → d_model
		self.k_proj = nn.Linear(freq_channels, self.d_model)
		self.v_proj = nn.Linear(freq_channels, self.d_model)
		
		# 输出投影：d_model → 空域通道数
		self.out_proj = nn.Linear(self.d_model, spatial_channels)
		
		# Dropout
		self.dropout_p = dropout
		self.dropout = nn.Dropout(dropout)
		
		# 缩放因子
		self.scale = math.sqrt(self.head_dim)
	
	def forward(
			self,
			spatial_tokens: torch.Tensor,
			freq_tokens: torch.Tensor
	) -> torch.Tensor:
		"""
		前向传播

		参数:
			spatial_tokens: 空域token序列 [B, N, C_sp]
			freq_tokens: 频域token序列 [B, p, C_fd]

		返回:
			attended: 注意力输出 [B, N, C_sp]
		"""
		B, N, _ = spatial_tokens.shape
		_, p, _ = freq_tokens.shape
		
		# 投影
		Q = self.q_proj(spatial_tokens)  # [B, N, d_model]
		K = self.k_proj(freq_tokens)  # [B, p, d_model]
		V = self.v_proj(freq_tokens)  # [B, p, d_model]
		
		# 多头reshape: [B, N, d_model] → [B, num_heads, N, head_dim]
		Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		K = K.view(B, p, self.num_heads, self.head_dim).transpose(1, 2)
		V = V.view(B, p, self.num_heads, self.head_dim).transpose(1, 2)
		
		# 使用 scaled_dot_product_attention（自动选择Flash Attention）
		if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
			# PyTorch 2.0+ 自动使用Flash Attention（如果硬件支持）
			attn_output = F.scaled_dot_product_attention(
				Q, K, V,
				attn_mask=None,
				dropout_p=self.dropout_p if self.training else 0.0,
				is_causal=False
			)
		else:
			# 标准实现（fallback）
			attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, heads, N, p]
			attn_weights = F.softmax(attn_scores, dim=-1)
			attn_weights = self.dropout(attn_weights)
			attn_output = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]
		
		# 合并多头: [B, heads, N, head_dim] → [B, N, d_model]
		attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)
		
		# 输出投影
		output = self.out_proj(attn_output)  # [B, N, C_sp]
		
		return output


class FreqGuidedAttentionPair(nn.Module):
	"""
	频域引导注意力对 (v2.0)

	单个频带的token序列引导单个空域path的注意力模块，包含：
	1. CrossAttention（多token版本）
	2. 残差连接 + LayerNorm

	参数:
		spatial_channels: 空域path的通道数
		freq_channels: 频域token的通道数
		num_heads: 注意力头数
		dropout: dropout比例

	输入:
		spatial_feat: 空域特征 [B, C_sp, D, H, W]
		freq_tokens: 频域token序列 [B, p, C_fd]

	输出:
		output: 增强后的空域特征 [B, C_sp, D, H, W]
	"""
	
	def __init__(
			self,
			spatial_channels: int,
			freq_channels: int,
			num_heads: int = 4,
			dropout: float = 0.0
	):
		super().__init__()
		
		self.spatial_channels = spatial_channels
		self.freq_channels = freq_channels
		
		# 交叉注意力（多头版本）
		self.cross_attn = CrossAttention(
			spatial_channels=spatial_channels,
			freq_channels=freq_channels,
			num_heads=num_heads,
			dropout=dropout
		)
		
		# Pre-norm（在attention之前）
		self.norm_spatial = nn.LayerNorm(spatial_channels)
		self.norm_freq = nn.LayerNorm(freq_channels)
		
		# 可学习的残差缩放因子
		self.gamma = nn.Parameter(torch.ones(1) * 0.1)
	
	def forward(
			self,
			spatial_feat: torch.Tensor,
			freq_tokens: torch.Tensor
	) -> torch.Tensor:
		"""
		前向传播

		参数:
			spatial_feat: 空域特征 [B, C_sp, D, H, W]
			freq_tokens: 频域token序列 [B, p, C_fd]

		返回:
			output: 增强后的空域特征 [B, C_sp, D, H, W]
		"""
		B, C, D, H, W = spatial_feat.shape
		
		# 1. 空域特征展平为token序列
		# [B, C, D, H, W] → [B, C, N] → [B, N, C]
		spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, N, C]
		
		# 2. 保存残差
		identity = spatial_tokens
		
		# 3. Pre-LayerNorm
		spatial_tokens_normed = self.norm_spatial(spatial_tokens)
		freq_tokens_normed = self.norm_freq(freq_tokens)
		
		# 4. 交叉注意力
		attn_out = self.cross_attn(spatial_tokens_normed, freq_tokens_normed)  # [B, N, C]
		
		# 5. 残差连接（带可学习缩放）
		output = identity + self.gamma * attn_out  # [B, N, C]
		
		# 6. 恢复空间形状
		# [B, N, C] → [B, C, N] → [B, C, D, H, W]
		output = output.transpose(1, 2).view(B, C, D, H, W)
		
		return output


# ============================================================================
# 兼容性接口：保留原有的单token版本（用于渐进迁移）
# ============================================================================

class CrossAttentionLegacy(nn.Module):
	"""
	原版CrossAttention（单token版本）

	保留用于向后兼容，新代码请使用 CrossAttention
	"""
	
	def __init__(
			self,
			spatial_channels: int,
			freq_channels: int,
			num_heads: int = 1,
			dropout: float = 0.0
	):
		super().__init__()
		
		self.spatial_channels = spatial_channels
		self.freq_channels = freq_channels
		self.num_heads = num_heads
		self.d_model = spatial_channels
		self.head_dim = self.d_model // num_heads
		
		assert self.d_model % num_heads == 0
		
		self.q_proj = nn.Linear(spatial_channels, self.d_model)
		self.k_proj = nn.Linear(freq_channels, self.d_model)
		self.v_proj = nn.Linear(freq_channels, self.d_model)
		self.out_proj = nn.Linear(self.d_model, spatial_channels)
		self.dropout = nn.Dropout(dropout)
		self.scale = math.sqrt(self.head_dim)
	
	def forward(
			self,
			spatial_tokens: torch.Tensor,
			freq_token: torch.Tensor
	) -> torch.Tensor:
		"""
		参数:
			spatial_tokens: [B, N, C_sp]
			freq_token: [B, C_fd] (单个token，非序列)
		"""
		B, N, _ = spatial_tokens.shape
		
		# 扩展维度
		freq_token = freq_token.unsqueeze(1)  # [B, 1, C_fd]
		
		Q = self.q_proj(spatial_tokens)  # [B, N, d_model]
		K = self.k_proj(freq_token)  # [B, 1, d_model]
		V = self.v_proj(freq_token)  # [B, 1, d_model]
		
		# 单头注意力
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, N, 1]
		attn_weights = F.softmax(attn_scores, dim=-1)  # 全是1.0
		attn_weights = self.dropout(attn_weights)
		attn_output = torch.matmul(attn_weights, V)  # [B, N, d_model]
		
		output = self.out_proj(attn_output)
		return output