"""
Stage 2: Lesion Detection, Classification, and Deformable Segmentation
病灶检测、分类和可变形分割网络

基于Deformable DETR的3D病灶检测分割网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from .deformable_attention import DeformableAttention3D


class PositionalEncoding3D(nn.Module):
    """3D位置编码"""
    
    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        z_embed = torch.arange(D, device=x.device, dtype=torch.float32) / D
        y_embed = torch.arange(H, device=x.device, dtype=torch.float32) / H
        x_embed = torch.arange(W, device=x.device, dtype=torch.float32) / W
        
        dim_t = torch.arange(self.hidden_dim // 6, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 6))
        
        pos_z = z_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = x_embed[:, None] / dim_t
        
        pos_z = torch.stack([pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()], dim=-1).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=-1).flatten(1)
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=-1).flatten(1)
        
        pos = torch.cat([
            pos_z[None, :, None, None, :].expand(1, D, H, W, -1),
            pos_y[None, None, :, None, :].expand(1, D, H, W, -1),
            pos_x[None, None, None, :, :].expand(1, D, H, W, -1),
        ], dim=-1)
        
        pos = pos.permute(0, 4, 1, 2, 3).expand(B, -1, -1, -1, -1)
        return pos


class DeformableTransformerEncoderLayer(nn.Module):
    """可变形Transformer编码器层"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DeformableAttention3D(hidden_dim, num_heads, num_points, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, src, spatial_shapes, level_start_index, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.ffn(src)
        src = self.norm2(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    """可变形Transformer编码器"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_points: int = 4, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(hidden_dim, num_heads, num_points, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, spatial_shapes, level_start_index, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, spatial_shapes, level_start_index, pos)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """可变形Transformer解码器层"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = DeformableAttention3D(hidden_dim, num_heads, num_points, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, tgt, memory, spatial_shapes, level_start_index, query_pos=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt + query_pos if query_pos is not None else tgt, memory, spatial_shapes, level_start_index)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    """可变形Transformer解码器"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_points: int = 4,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(hidden_dim, num_heads, num_points, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt, memory, spatial_shapes, level_start_index, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, spatial_shapes, level_start_index, query_pos)
        return output


class LesionDetClsDeformSegNet(nn.Module):
    """病灶检测、分类和可变形分割网络"""
    
    def __init__(
        self,
        in_channels: int = 1,
        backbone_variant: str = 'small',
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # 简化backbone
        backbone_channels = [96, 192, 384, 768]
        self.backbone = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels if i == 0 else backbone_channels[i-1], backbone_channels[i], 3, stride=2, padding=1),
                nn.BatchNorm3d(backbone_channels[i]),
                nn.GELU()
            )
            for i in range(4)
        ])
        
        self.input_projs = nn.ModuleList([nn.Conv3d(c, hidden_dim, 1) for c in backbone_channels])
        self.level_embed = nn.Parameter(torch.zeros(4, hidden_dim))
        nn.init.normal_(self.level_embed)
        
        self.pos_encoding = PositionalEncoding3D(hidden_dim)
        self.encoder = DeformableTransformerEncoder(hidden_dim, num_heads, num_points, num_encoder_layers, dropout)
        self.decoder = DeformableTransformerDecoder(hidden_dim, num_heads, num_points, num_decoder_layers, dropout)
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 6))
        
        self.mask_head = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 64, 2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # Extract features
        features = []
        feat = x
        for layer in self.backbone:
            feat = layer(feat)
            features.append(feat)
        
        # Project features
        srcs = [proj(feat) for feat, proj in zip(features, self.input_projs)]
        pos_embeds = [self.pos_encoding(src) for src in srcs]
        
        # Flatten
        src_flatten, spatial_shapes, level_start_index = self._flatten_features(srcs)
        pos_flatten = torch.cat([p.flatten(2).transpose(1, 2) for p in pos_embeds], dim=1)
        
        # Encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, pos_flatten)
        
        # Decoder
        query_embed = self.query_embed.weight
        query_pos, query = query_embed.split(self.hidden_dim, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        query = query.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(query, memory, spatial_shapes, level_start_index, query_pos)
        
        # Predictions
        outputs_class = self.class_head(hs)
        outputs_box = self.box_head(hs).sigmoid()
        
        # Masks
        hs_reshape = hs.transpose(1, 2).reshape(B * self.num_queries, self.hidden_dim, 1, 1, 1)
        hs_expand = hs_reshape.expand(-1, -1, 8, 8, 8)
        masks = self.mask_head(hs_expand).view(B, self.num_queries, 32, 32, 32)
        masks = F.interpolate(masks, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        has_lesion = (outputs_class[:, :, 1] > 0).any(dim=1).float()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_box,
            'pred_masks': masks,
            'has_lesion': has_lesion,
            'query_features': hs,
        }
        
    def _flatten_features(self, features):
        spatial_shapes = []
        src_flatten = []
        for lvl, feat in enumerate(features):
            B, C, D, H, W = feat.shape
            spatial_shapes.append((D, H, W))
            feat_flat = feat.flatten(2).transpose(1, 2)
            feat_flat = feat_flat + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(feat_flat)
        src_flatten = torch.cat(src_flatten, dim=1)
        level_start_index = torch.zeros(len(spatial_shapes), dtype=torch.long, device=src_flatten.device)
        for i in range(1, len(spatial_shapes)):
            D, H, W = spatial_shapes[i - 1]
            level_start_index[i] = level_start_index[i - 1] + D * H * W
        return src_flatten, spatial_shapes, level_start_index
        
    def get_detections(self, outputs, score_threshold=0.5, nms_threshold=0.3):
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_masks = outputs['pred_masks']
        B = pred_logits.shape[0]
        results = []
        for b in range(B):
            scores = pred_logits[b].softmax(dim=-1)[:, 1]
            keep = scores > score_threshold
            results.append({'scores': scores[keep], 'boxes': pred_boxes[b][keep], 'masks': pred_masks[b][keep]})
        return results


LesionDetector = LesionDetClsDeformSegNet


if __name__ == "__main__":
    model = LesionDetClsDeformSegNet(in_channels=1, num_queries=50, hidden_dim=256)
    x = torch.randn(2, 1, 64, 64, 64)
    outputs = model(x)
    print(f"Pred logits shape: {outputs['pred_logits'].shape}")
    print(f"Pred boxes shape: {outputs['pred_boxes'].shape}")
    print(f"Pred masks shape: {outputs['pred_masks'].shape}")
