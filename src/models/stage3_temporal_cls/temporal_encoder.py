"""
Temporal Encoder and Aggregator
时序编码器与聚合器

Transformer-based temporal encoding for multi-slice feature aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 64,
        encoding_type: str = 'learnable',
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            encoding_type: 'learnable', 'sinusoidal', or 'relative'
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'learnable':
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            
        elif encoding_type == 'sinusoidal':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        elif encoding_type == 'relative':
            # Relative positional encoding (simplified)
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TemporalTransformerLayer(nn.Module):
    """Single Transformer layer for temporal encoding."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, L, D)
            attn_mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention
        if self.pre_norm:
            x_norm = self.norm1(x)
            attn_out, attn_weights = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=return_attention,
            )
            x = x + self.dropout(attn_out)
        else:
            attn_out, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                need_weights=return_attention,
            )
            x = self.norm1(x + self.dropout(attn_out))
            
        # FFN
        if self.pre_norm:
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm2(x + self.ffn(x))
            
        if return_attention:
            return x, attn_weights
        return x, None


class TemporalEncoder(nn.Module):
    """Temporal Encoder using Transformer architecture.
    
    Encodes temporal relationships between multi-slice features.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 32,
        position_encoding: str = 'learnable',
        pre_norm: bool = True,
    ):
        """Initialize temporal encoder.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            position_encoding: Type of positional encoding
            pre_norm: Whether to use pre-normalization
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            encoding_type=position_encoding,
            dropout=dropout,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pre_norm=pre_norm,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization (for pre-norm)
        self.norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass.
        
        Args:
            x: Input features (B, K, D) where K is number of slices
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Encoded features (B, K, D) and optionally attention weights
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask, return_attention)
            if return_attention:
                attention_weights.append(attn)
                
        # Final normalization
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights
        return x, None


class TemporalAggregator(nn.Module):
    """Aggregates temporal features into a global representation.
    
    Supports multiple aggregation methods:
    - attention_pooling: Learned attention-based pooling
    - mean: Simple mean pooling
    - max: Max pooling
    - cls_token: Use a CLS token
    """
    
    def __init__(
        self,
        d_model: int = 512,
        aggregation_type: str = 'attention_pooling',
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize temporal aggregator.
        
        Args:
            d_model: Feature dimension
            aggregation_type: Type of aggregation
            num_heads: Number of attention heads (for attention pooling)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention_pooling':
            # Multi-head attention pooling
            self.query = nn.Parameter(torch.randn(1, num_heads, d_model // num_heads) * 0.02)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.scale = self.head_dim ** -0.5
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
        elif aggregation_type == 'cls_token':
            # CLS token (prepended during encoding)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            
    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input features (B, K, D)
            return_weights: Whether to return attention/importance weights
            
        Returns:
            Aggregated features (B, D) and optionally weights
        """
        B, K, D = x.shape
        
        if self.aggregation_type == 'attention_pooling':
            # Project keys and values
            keys = self.key_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
            values = self.value_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Expand query for batch
            query = self.query.expand(B, -1, -1).unsqueeze(2)  # (B, num_heads, 1, head_dim)
            
            # Attention
            attn = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, K)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Aggregate
            out = torch.matmul(attn, values)  # (B, num_heads, 1, head_dim)
            out = out.squeeze(2).reshape(B, D)
            out = self.out_proj(out)
            
            # Return slice importance weights (average over heads)
            weights = attn.mean(dim=1).squeeze(1) if return_weights else None  # (B, K)
            
        elif self.aggregation_type == 'mean':
            out = x.mean(dim=1)
            weights = torch.ones(B, K, device=x.device) / K if return_weights else None
            
        elif self.aggregation_type == 'max':
            out, indices = x.max(dim=1)
            weights = F.one_hot(indices, K).float() if return_weights else None
            
        elif self.aggregation_type == 'cls_token':
            # Assume CLS token is the first position
            out = x[:, 0]
            weights = None
            
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
            
        return out, weights


if __name__ == "__main__":
    # Test temporal encoder
    print("Testing Temporal Encoder:")
    encoder = TemporalEncoder(
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )
    
    batch_size = 4
    num_slices = 16
    feature_dim = 512
    
    x = torch.randn(batch_size, num_slices, feature_dim)
    encoded, attn = encoder(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Number of attention weight tensors: {len(attn)}")
    
    # Test temporal aggregator
    print("\nTesting Temporal Aggregator:")
    aggregator = TemporalAggregator(
        d_model=512,
        aggregation_type='attention_pooling',
        num_heads=4,
    )
    
    aggregated, weights = aggregator(encoded, return_weights=True)
    
    print(f"Aggregated shape: {aggregated.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=1)}")  # Should be 1.0
    
    # Model parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    aggregator_params = sum(p.numel() for p in aggregator.parameters())
    print(f"\nEncoder parameters: {encoder_params / 1e6:.2f}M")
    print(f"Aggregator parameters: {aggregator_params / 1e6:.4f}M")
