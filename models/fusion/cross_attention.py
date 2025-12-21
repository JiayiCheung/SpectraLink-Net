# models/fusion/cross_attention.py
"""Cross-attention module for frequency-spatial feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossAttention(nn.Module):
    """
    Cross-attention for frequency-guided spatial feature enhancement.
    
    Q from spatial features [B, N, C_sp], K/V from frequency tokens [B, p, C_fd].
    Complexity: O(N * p), efficient when p << N.
    
    Args:
        spatial_channels: Spatial feature channels (Q dimension)
        freq_channels: Frequency token channels (K/V dimension)
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_flash_attn: Enable Flash Attention (PyTorch 2.0+)
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
        
        self.d_model = spatial_channels
        self.head_dim = self.d_model // num_heads
        
        assert self.d_model % num_heads == 0
        
        self.q_proj = nn.Linear(spatial_channels, self.d_model)
        self.k_proj = nn.Linear(freq_channels, self.d_model)
        self.v_proj = nn.Linear(freq_channels, self.d_model)
        self.out_proj = nn.Linear(self.d_model, spatial_channels)
        
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
            self,
            spatial_tokens: torch.Tensor,
            freq_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N, C_sp]
            freq_tokens: [B, p, C_fd]
        Returns:
            attended: [B, N, C_sp]
        """
        B, N, _ = spatial_tokens.shape
        _, p, _ = freq_tokens.shape
        
        Q = self.q_proj(spatial_tokens)
        K = self.k_proj(freq_tokens)
        V = self.v_proj(freq_tokens)
        
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, p, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, p, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(attn_output)


class FreqGuidedAttentionPair(nn.Module):
    """
    Frequency-guided attention pair with residual connection.
    
    Args:
        spatial_channels: Spatial path channels
        freq_channels: Frequency token channels
        num_heads: Number of attention heads
        dropout: Dropout rate
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
        
        self.cross_attn = CrossAttention(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm_spatial = nn.LayerNorm(spatial_channels)
        self.norm_freq = nn.LayerNorm(freq_channels)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-5)
    
    def forward(
            self,
            spatial_feat: torch.Tensor,
            freq_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat: [B, C_sp, D, H, W]
            freq_tokens: [B, p, C_fd]
        Returns:
            output: [B, C_sp, D, H, W]
        """
        B, C, D, H, W = spatial_feat.shape
        
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)
        identity = spatial_tokens
        
        spatial_tokens_normed = self.norm_spatial(spatial_tokens)
        freq_tokens_normed = self.norm_freq(freq_tokens)
        
        attn_out = self.cross_attn(spatial_tokens_normed, freq_tokens_normed)
        output = identity + self.gamma * attn_out
        
        return output.transpose(1, 2).view(B, C, D, H, W)


class CrossAttentionLegacy(nn.Module):
    """Legacy single-token cross-attention for backward compatibility."""
    
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
        B, N, _ = spatial_tokens.shape
        
        freq_token = freq_token.unsqueeze(1)
        
        Q = self.q_proj(spatial_tokens)
        K = self.k_proj(freq_token)
        V = self.v_proj(freq_token)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        return self.out_proj(attn_output)
