"""
Hausdorff Distance Loss
Hausdorff距离损失函数

Directly optimizes boundary distance for better edge quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scipy import ndimage
import numpy as np


class HDLoss(nn.Module):
    """Hausdorff Distance Loss for boundary optimization.
    
    Uses distance transform to create differentiable approximation.
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        reduction: str = 'mean',
        sigmoid: bool = True,
    ):
        """Initialize HD Loss.
        
        Args:
            alpha: Power for distance weighting (higher = more boundary focus)
            reduction: 'mean', 'sum', or 'none'
            sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dist_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute HD Loss.
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, C, D, H, W)
            dist_map: Pre-computed distance map (optional)
            
        Returns:
            HD loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        if dist_map is None:
            # Compute distance transform on CPU
            dist_map = self._compute_distance_map(target)
            dist_map = dist_map.to(pred.device)
            
        # Weight predictions by distance
        dist_weighted_pred = pred * (dist_map ** self.alpha)
        dist_weighted_target = target.float() * (dist_map ** self.alpha)
        
        # Compute weighted difference
        loss = torch.abs(dist_weighted_pred - dist_weighted_target)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
        
    def _compute_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for target.
        
        Args:
            target: Binary target mask
            
        Returns:
            Distance transform tensor
        """
        target_np = target.cpu().numpy()
        dist_maps = []
        
        for b in range(target_np.shape[0]):
            for c in range(target_np.shape[1]):
                # Compute distance transform
                pos_dist = ndimage.distance_transform_edt(target_np[b, c])
                neg_dist = ndimage.distance_transform_edt(1 - target_np[b, c])
                
                # Signed distance field
                dist = pos_dist - neg_dist
                dist_maps.append(dist)
                
        dist_tensor = torch.from_numpy(
            np.array(dist_maps).reshape(target.shape)
        ).float()
        
        return dist_tensor


class BoundaryLoss(nn.Module):
    """Boundary Loss using distance maps.
    
    Directly minimizes distance to ground truth boundary.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        sigmoid: bool = True,
    ):
        super().__init__()
        self.reduction = reduction
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        dist_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Boundary Loss.
        
        Args:
            pred: Predictions (B, C, D, H, W)
            dist_map: Signed distance map (positive inside, negative outside)
            
        Returns:
            Boundary loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Multiply predictions by distance map
        # Positive distance = inside object, negative = outside
        loss = pred * dist_map
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_sdf(mask: torch.Tensor) -> torch.Tensor:
    """Compute signed distance field for a binary mask.
    
    Args:
        mask: Binary mask (B, C, D, H, W) or (B, D, H, W)
        
    Returns:
        Signed distance field (positive inside, negative outside)
    """
    mask_np = mask.cpu().numpy().astype(np.float64)
    
    if mask_np.ndim == 4:
        mask_np = mask_np[:, np.newaxis, ...]
        
    sdf_list = []
    for b in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            pos_sdf = ndimage.distance_transform_edt(mask_np[b, c])
            neg_sdf = ndimage.distance_transform_edt(1 - mask_np[b, c])
            sdf = pos_sdf - neg_sdf
            sdf_list.append(sdf)
            
    sdf_tensor = torch.from_numpy(
        np.array(sdf_list).reshape(mask_np.shape)
    ).float()
    
    if mask.ndim == 4:
        sdf_tensor = sdf_tensor.squeeze(1)
        
    return sdf_tensor


if __name__ == "__main__":
    # Test
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    
    hd_loss = HDLoss(alpha=2.0)
    loss = hd_loss(pred, target)
    print(f"HD Loss: {loss.item():.4f}")
    
    dist_map = compute_sdf(target)
    boundary_loss = BoundaryLoss()
    loss = boundary_loss(pred, dist_map.to(pred.device))
    print(f"Boundary Loss: {loss.item():.4f}")
