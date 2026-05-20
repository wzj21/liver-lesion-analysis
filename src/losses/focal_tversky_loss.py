"""
Focal Tversky Loss
Focal Tversky损失函数

Better handles class imbalance in medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TverskyLoss(nn.Module):
    """Tversky Loss for imbalanced segmentation.
    
    Tversky index = TP / (TP + α*FN + β*FP)
    
    α > β penalizes false negatives more (useful when missing lesions is costly)
    β > α penalizes false positives more
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # FN weight
        beta: float = 0.3,   # FP weight
        smooth: float = 1.0,
        reduction: str = 'mean',
        sigmoid: bool = True,
    ):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false negatives (typically > 0.5 for medical imaging)
            beta: Weight for false positives (alpha + beta = 1)
            smooth: Smoothing factor
            reduction: 'mean', 'sum', or 'none'
            sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tversky Loss.
        
        Args:
            pred: Predictions (B, C, ...) or (B, ...)
            target: Ground truth (same shape)
            
        Returns:
            Tversky loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Flatten
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1).float()
        
        # Compute TP, FN, FP
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for highly imbalanced segmentation.
    
    Combines Tversky loss with focal mechanism to focus on hard examples.
    Loss = (1 - Tversky)^γ
    
    Higher γ focuses more on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,  # Focal parameter
        smooth: float = 1.0,
        reduction: str = 'mean',
        sigmoid: bool = True,
    ):
        """Initialize Focal Tversky Loss.
        
        Args:
            alpha: Weight for false negatives
            beta: Weight for false positives
            gamma: Focal parameter (higher = more focus on hard examples)
            smooth: Smoothing factor
            reduction: 'mean', 'sum', or 'none'
            sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal Tversky Loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            
        Returns:
            Focal Tversky loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Flatten
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1).float()
        
        # Compute per-sample Tversky
        tp = (pred_flat * target_flat).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal modulation
        focal_tversky = (1 - tversky) ** self.gamma
        
        if self.reduction == 'mean':
            return focal_tversky.mean()
        elif self.reduction == 'sum':
            return focal_tversky.sum()
        return focal_tversky


if __name__ == "__main__":
    # Test
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    
    tversky = TverskyLoss(alpha=0.7, beta=0.3)
    loss = tversky(pred, target)
    print(f"Tversky Loss: {loss.item():.4f}")
    
    focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    loss = focal_tversky(pred, target)
    print(f"Focal Tversky Loss: {loss.item():.4f}")
