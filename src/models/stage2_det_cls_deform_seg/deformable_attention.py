"""
Deformable Attention Module
可变形注意力模块

Multi-scale deformable attention for efficient feature aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class DeformableAttention3d(nn.Module):
    """Multi-Scale Deformable Attention for 3D features.
    
    Each query learns K sampling offsets and attention weights
    to aggregate features from multiple scales.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize deformable attention.
        
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_levels: Number of feature levels
            n_points: Number of sampling points per query
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        
        # Sampling offsets (learn 3D offsets for each point)
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 3
        )
        
        # Attention weights
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        
        # Initialize attention weights uniformly
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: List[Tuple[int, int, int]],
        input_level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: Query features (B, N_q, C)
            reference_points: Reference points for queries (B, N_q, n_levels, 3)
            input_flatten: Flattened multi-scale features (B, sum(D*H*W), C)
            input_spatial_shapes: Spatial shapes at each level
            input_level_start_index: Start indices for each level
            
        Returns:
            Output features (B, N_q, C)
        """
        B, N_q, _ = query.shape
        B, N_k, _ = input_flatten.shape
        
        # Project values
        value = self.value_proj(input_flatten)
        value = value.view(B, N_k, self.n_heads, self.d_model // self.n_heads)
        
        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points, 3
        )
        
        # Predict attention weights
        attention_weights = self.attention_weights(query).view(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points
        )
        
        # Add offsets to reference points
        # reference_points: (B, N_q, n_levels, 3) -> (B, N_q, n_heads, n_levels, n_points, 3)
        reference_points = reference_points[:, :, None, :, None, :]
        sampling_locations = reference_points + sampling_offsets
        
        # Sample features (simplified version using grid_sample)
        output = self._sample_features(
            value, sampling_locations, attention_weights,
            input_spatial_shapes, input_level_start_index
        )
        
        return self.output_proj(output)
        
    def _sample_features(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        spatial_shapes: List[Tuple[int, int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sample features at given locations.
        
        Args:
            value: Value features (B, N_k, n_heads, C_v)
            sampling_locations: Sampling locations (B, N_q, n_heads, n_levels, n_points, 3)
            attention_weights: Attention weights (B, N_q, n_heads, n_levels, n_points)
            spatial_shapes: Spatial shapes at each level
            level_start_index: Start indices for each level
            
        Returns:
            Sampled features (B, N_q, C)
        """
        B, N_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
        
        sampled_values = []
        
        for level_idx, (D, H, W) in enumerate(spatial_shapes):
            start_idx = level_start_index[level_idx]
            end_idx = start_idx + D * H * W
            
            # Get value for this level
            value_level = value[:, start_idx:end_idx, :, :]  # (B, D*H*W, n_heads, C_v)
            value_level = value_level.view(B, D, H, W, n_heads, -1)
            value_level = value_level.permute(0, 4, 5, 1, 2, 3)  # (B, n_heads, C_v, D, H, W)
            
            # Get sampling locations for this level
            locations = sampling_locations[:, :, :, level_idx, :, :]  # (B, N_q, n_heads, n_points, 3)
            
            # Normalize to [-1, 1]
            locations = 2 * locations - 1
            
            # Sample using grid_sample (for each head)
            sampled = []
            for head_idx in range(n_heads):
                value_head = value_level[:, head_idx]  # (B, C_v, D, H, W)
                loc_head = locations[:, :, head_idx]  # (B, N_q, n_points, 3)
                
                # Reshape for grid_sample
                loc_head = loc_head.view(B, N_q * n_points, 1, 1, 3)
                
                # Sample
                sampled_head = F.grid_sample(
                    value_head, loc_head,
                    mode='bilinear', padding_mode='zeros', align_corners=False
                )  # (B, C_v, N_q*n_points, 1, 1)
                
                sampled_head = sampled_head.view(B, -1, N_q, n_points)  # (B, C_v, N_q, n_points)
                sampled.append(sampled_head)
                
            sampled = torch.stack(sampled, dim=1)  # (B, n_heads, C_v, N_q, n_points)
            sampled_values.append(sampled)
            
        # Stack and weight by attention
        sampled_values = torch.stack(sampled_values, dim=-1)  # (B, n_heads, C_v, N_q, n_points, n_levels)
        
        # Apply attention weights
        attention_weights = attention_weights.permute(0, 2, 4, 1, 3)  # (B, n_heads, n_points, N_q, n_levels)
        attention_weights = attention_weights.unsqueeze(2)  # (B, n_heads, 1, n_points, N_q, n_levels)
        
        output = (sampled_values * attention_weights.permute(0, 1, 2, 4, 3, 5)).sum(dim=(-1, -2))
        # (B, n_heads, C_v, N_q)
        
        output = output.permute(0, 3, 1, 2).reshape(B, N_q, -1)
        
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention (deformable)
        self.cross_attn = DeformableAttention3d(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: List[Tuple[int, int, int]],
        src_level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tgt: Query features (B, N_q, C)
            reference_points: Reference points
            src: Multi-scale source features (flattened)
            src_spatial_shapes: Spatial shapes at each level
            src_level_start_index: Start indices for each level
            
        Returns:
            Updated query features
        """
        # Self attention
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention (deformable)
        tgt2 = self.cross_attn(
            tgt, reference_points, src,
            src_spatial_shapes, src_level_start_index
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


if __name__ == "__main__":
    # Test deformable attention
    B, N_q, N_k, C = 2, 100, 1000, 256
    n_levels = 4
    n_points = 4
    
    deform_attn = DeformableAttention3d(
        d_model=C,
        n_heads=8,
        n_levels=n_levels,
        n_points=n_points,
    )
    
    query = torch.randn(B, N_q, C)
    reference_points = torch.rand(B, N_q, n_levels, 3)  # Normalized [0, 1]
    input_flatten = torch.randn(B, N_k, C)
    spatial_shapes = [(8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]
    level_start_index = torch.tensor([0, 512, 576, 584])
    
    output = deform_attn(query, reference_points, input_flatten, spatial_shapes, level_start_index)
    print(f"Output shape: {output.shape}")
    
    # Test decoder layer
    decoder_layer = DeformableTransformerDecoderLayer(d_model=C, n_levels=n_levels)
    output = decoder_layer(query, reference_points, input_flatten, spatial_shapes, level_start_index)
    print(f"Decoder output shape: {output.shape}")


# Alias for backward compatibility
DeformableAttention3D = DeformableAttention3d

