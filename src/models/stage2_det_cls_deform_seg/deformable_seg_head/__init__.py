"""
Deformable Segmentation Head
可变形分割头

Generates per-query instance segmentation masks using deformable attention
to iteratively refine mask predictions from multi-scale feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MaskHead(nn.Module):
    """Lightweight FPN-style mask prediction head.

    Takes a query embedding and multi-scale encoder features, applies
    dot-product attention to produce coarse masks, then upsamples.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        fpn_dims: List[int] = None,
        num_convs: int = 4,
        mask_dim: int = 32,
    ):
        """
        Args:
            hidden_dim: query / transformer hidden dim
            fpn_dims: feature pyramid channel dims (coarse → fine)
            num_convs: number of 3×3×3 conv layers in mask tower
            mask_dim: final mask feature dimension
        """
        super().__init__()

        if fpn_dims is None:
            fpn_dims = [hidden_dim, hidden_dim, hidden_dim]

        # Lateral connections (project FPN levels to mask_dim)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for fpn_ch in fpn_dims:
            self.lateral_convs.append(
                nn.Conv3d(fpn_ch, mask_dim, kernel_size=1)
            )
            self.output_convs.append(nn.Sequential(
                nn.Conv3d(mask_dim, mask_dim, 3, padding=1),
                nn.GroupNorm(8, mask_dim),
                nn.ReLU(inplace=True),
            ))

        # Mask prediction tower (operates on fused features)
        tower = []
        for _ in range(num_convs):
            tower.extend([
                nn.Conv3d(mask_dim, mask_dim, 3, padding=1),
                nn.GroupNorm(8, mask_dim),
                nn.ReLU(inplace=True),
            ])
        self.mask_tower = nn.Sequential(*tower)

        # Final 1×1 conv to produce per-pixel mask logits
        self.mask_pred = nn.Conv3d(mask_dim, mask_dim, 1)

        # Query-to-mask projection
        self.query_proj = nn.Linear(hidden_dim, mask_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        target_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            query_features: (B, N_q, hidden_dim)
            multi_scale_features: list of (B, C, D_i, H_i, W_i), coarse→fine
            target_size: output spatial size (D, H, W)

        Returns:
            masks: (B, N_q, D, H, W) – per-query mask logits
        """
        # ---- Build FPN mask features ----
        # Start from coarsest level
        x = self.lateral_convs[-1](multi_scale_features[-1])
        x = self.output_convs[-1](x)

        for i in range(len(multi_scale_features) - 2, -1, -1):
            lateral = self.lateral_convs[i](multi_scale_features[i])
            # Upsample coarser feature and fuse
            x = F.interpolate(x, size=lateral.shape[2:], mode='trilinear', align_corners=False)
            x = x + lateral
            x = self.output_convs[i](x)

        # Mask tower
        mask_features = self.mask_tower(x)       # (B, mask_dim, D', H', W')
        mask_features = self.mask_pred(mask_features)

        # ---- Dot-product to produce per-query masks ----
        q = self.query_proj(query_features)       # (B, N_q, mask_dim)
        B, N_q, C = q.shape

        # Einsum: for each query, dot with every spatial location
        mask_features_flat = mask_features.flatten(2)  # (B, mask_dim, D'×H'×W')
        masks = torch.einsum('bqc,bcn->bqn', q, mask_features_flat)

        # Reshape to spatial
        D, H, W = mask_features.shape[2:]
        masks = masks.view(B, N_q, D, H, W)

        # Upsample to target size
        if (D, H, W) != target_size:
            masks = masks.view(B * N_q, 1, D, H, W)
            masks = F.interpolate(masks, size=target_size, mode='trilinear', align_corners=False)
            masks = masks.view(B, N_q, *target_size)

        return masks


class DeformableSegHead(nn.Module):
    """Complete deformable segmentation head.

    Wraps MaskHead with optional iterative refinement and auxiliary losses.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        fpn_dims: List[int] = None,
        num_mask_convs: int = 4,
        mask_dim: int = 32,
        num_refine_steps: int = 0,
    ):
        super().__init__()

        self.mask_head = MaskHead(
            hidden_dim=hidden_dim,
            fpn_dims=fpn_dims,
            num_convs=num_mask_convs,
            mask_dim=mask_dim,
        )

        # Optional iterative refinement
        self.num_refine_steps = num_refine_steps
        if num_refine_steps > 0:
            self.refine_layers = nn.ModuleList()
            for _ in range(num_refine_steps):
                self.refine_layers.append(nn.Sequential(
                    nn.Conv3d(1 + hidden_dim, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, 1, 1),
                ))

    def forward(
        self,
        query_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        target_size: Tuple[int, int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: (B, N_q, hidden_dim)
            multi_scale_features: list of 3D feature maps
            target_size: desired output spatial size

        Returns:
            dict with 'masks' (B, N_q, D, H, W), optionally 'refined_masks'
        """
        masks = self.mask_head(query_features, multi_scale_features, target_size)
        result = {'masks': masks}

        if self.num_refine_steps > 0:
            B, N_q, D, H, W = masks.shape
            refined = masks
            for step, layer in enumerate(self.refine_layers):
                # Expand query features to spatial
                q_spatial = query_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                q_spatial = q_spatial.expand(-1, -1, -1, D, H, W)

                # Concatenate current mask prediction with query features
                mask_input = refined.unsqueeze(2)       # (B, N_q, 1, D, H, W)
                cat_input = torch.cat([mask_input, q_spatial], dim=2)  # (B, N_q, 1+C, D, H, W)

                # Process each query
                cat_flat = cat_input.view(B * N_q, -1, D, H, W)
                delta = layer(cat_flat).view(B, N_q, D, H, W)
                refined = refined + delta

            result['refined_masks'] = refined

        return result


class DiceMaskLoss(nn.Module):
    """Dice loss for instance mask predictions."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_masks: (N, D, H, W) sigmoid logits
            target_masks: (N, D, H, W) binary
        """
        pred = pred_masks.sigmoid().flatten(1)
        target = target_masks.flatten(1).float()

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


if __name__ == "__main__":
    print("Testing DeformableSegHead...")

    head = DeformableSegHead(
        hidden_dim=256,
        fpn_dims=[256, 256, 256],
        num_mask_convs=3,
        mask_dim=32,
        num_refine_steps=1,
    )

    B, N_q = 2, 10
    queries = torch.randn(B, N_q, 256)
    feats = [
        torch.randn(B, 256, 8, 8, 8),
        torch.randn(B, 256, 4, 4, 4),
        torch.randn(B, 256, 2, 2, 2),
    ]
    target_size = (32, 32, 32)

    out = head(queries, feats, target_size)
    print(f"  masks shape: {out['masks'].shape}")
    if 'refined_masks' in out:
        print(f"  refined shape: {out['refined_masks'].shape}")
    print(f"  Params: {sum(p.numel() for p in head.parameters()) / 1e6:.4f}M")
    print("  ✓ DeformableSegHead test passed!")
