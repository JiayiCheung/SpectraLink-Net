# models/fusion/cross_attention.py
"""
交叉注意力模块
用于频域-空域特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossAttention(nn.Module):
	"""
	交叉注意力模块

	实现频域引导空域的注意力机制：
	- Q (Query): 来自空域特征，展平为token序列 [B, N, C_sp]
	- K (Key): 来自频域token [B, 1, C_fd]
	- V (Value): 来自频域token [B, 1, C_fd]

	由于K/V只有1个token，计算复杂度是O(N)而非O(N²)，非常高效。

	注意力计算：
	Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

	参数:
		spatial_channels: 空域特征通道数（Q的维度）
		freq_channels: 频域token通道数（K/V的维度）
		num_heads: 注意力头数，默认1（因为K/V只有1个token）
		dropout: dropout比例

	输入:
		spatial_tokens: 空域token序列 [B, N, C_sp]
		freq_token: 频域token [B, C_fd]

	输出:
		attended: 注意力输出 [B, N, C_sp]
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
		self.d_model = spatial_channels  # 使用空域通道数作为d_model
		self.head_dim = self.d_model // num_heads
		
		assert self.d_model % num_heads == 0, \
			f"d_model({self.d_model})必须能被num_heads({num_heads})整除"
		
		# Q投影：空域特征 → d_model
		self.q_proj = nn.Linear(spatial_channels, self.d_model)
		
		# K/V投影：频域token → d_model
		self.k_proj = nn.Linear(freq_channels, self.d_model)
		self.v_proj = nn.Linear(freq_channels, self.d_model)
		
		# 输出投影：d_model → 空域通道数
		self.out_proj = nn.Linear(self.d_model, spatial_channels)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
		
		# 缩放因子
		self.scale = math.sqrt(self.head_dim)
	
	def forward(
			self,
			spatial_tokens: torch.Tensor,
			freq_token: torch.Tensor
	) -> torch.Tensor:
		"""
		前向传播

		参数:
			spatial_tokens: 空域token序列 [B, N, C_sp]
			freq_token: 频域token [B, C_fd]

		返回:
			attended: 注意力输出 [B, N, C_sp]
		"""
		B, N, _ = spatial_tokens.shape
		
		# 频域token扩展维度 [B, C_fd] → [B, 1, C_fd]
		freq_token = freq_token.unsqueeze(1)
		
		# 投影
		Q = self.q_proj(spatial_tokens)  # [B, N, d_model]
		K = self.k_proj(freq_token)  # [B, 1, d_model]
		V = self.v_proj(freq_token)  # [B, 1, d_model]
		
		# 多头注意力reshape（如果num_heads > 1）
		if self.num_heads > 1:
			# [B, N, d_model] → [B, num_heads, N, head_dim]
			Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
			K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
			V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
			
			# 注意力分数 [B, num_heads, N, 1]
			attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
			attn_weights = F.softmax(attn_scores, dim=-1)
			attn_weights = self.dropout(attn_weights)
			
			# 注意力输出 [B, num_heads, N, head_dim]
			attn_output = torch.matmul(attn_weights, V)
			
			# 合并多头 [B, N, d_model]
			attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)
		else:
			# 单头注意力，更简单
			# 注意力分数 [B, N, 1]
			attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
			attn_weights = F.softmax(attn_scores, dim=-1)
			attn_weights = self.dropout(attn_weights)
			
			# 注意力输出 [B, N, d_model]
			attn_output = torch.matmul(attn_weights, V)
		
		# 输出投影
		output = self.out_proj(attn_output)  # [B, N, C_sp]
		
		return output


class FreqGuidedAttentionPair(nn.Module):
	"""
	频域引导注意力对

	单个频带token引导单个空域path的注意力模块，包含：
	1. CrossAttention
	2. 残差连接

	参数:
		spatial_channels: 空域path的通道数
		freq_channels: 频域token的通道数
		dropout: dropout比例

	输入:
		spatial_feat: 空域特征 [B, C_sp, D, H, W]
		freq_token: 频域token [B, C_fd]

	输出:
		output: 增强后的空域特征 [B, C_sp, D, H, W]
	"""
	
	def __init__(
			self,
			spatial_channels: int,
			freq_channels: int,
			dropout: float = 0.0
	):
		super().__init__()
		
		self.spatial_channels = spatial_channels
		self.freq_channels = freq_channels
		
		# 交叉注意力
		self.cross_attn = CrossAttention(
			spatial_channels=spatial_channels,
			freq_channels=freq_channels,
			num_heads=1,  # 单头，因为K/V只有1个token
			dropout=dropout
		)
		
		# LayerNorm（在token维度上）
		self.norm = nn.LayerNorm(spatial_channels)
	
	def forward(
			self,
			spatial_feat: torch.Tensor,
			freq_token: torch.Tensor
	) -> torch.Tensor:
		"""
		前向传播

		参数:
			spatial_feat: 空域特征 [B, C_sp, D, H, W]
			freq_token: 频域token [B, C_fd]

		返回:
			output: 增强后的空域特征 [B, C_sp, D, H, W]
		"""
		B, C, D, H, W = spatial_feat.shape
		
		# 1. 空域特征展平为token序列
		# [B, C, D, H, W] → [B, C, N] → [B, N, C]
		spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, N, C]
		
		# 2. 保存残差
		identity = spatial_tokens
		
		# 3. LayerNorm
		spatial_tokens = self.norm(spatial_tokens)
		
		# 4. 交叉注意力
		attn_out = self.cross_attn(spatial_tokens, freq_token)  # [B, N, C]
		
		# 5. 残差连接
		output = identity + attn_out  # [B, N, C]
		
		# 6. 恢复空间形状
		# [B, N, C] → [B, C, N] → [B, C, D, H, W]
		output = output.transpose(1, 2).view(B, C, D, H, W)
		
		return output