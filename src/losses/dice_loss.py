"""
Dice Loss and Variants
Dice损失函数及其变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Standard Dice Loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        softmax: bool = False,
        sigmoid: bool = True,
    ):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
            softmax: Apply softmax to predictions
            sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.softmax = softmax
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Dice Loss.
        
        Args:
            pred: Predictions (B, C, ...) or (B, ...)
            target: Ground truth (B, C, ...) or (B, ...)
            weight: Optional class weights
            
        Returns:
            Dice loss
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        elif self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Ensure same shape
        if pred.dim() != target.dim():
            if target.dim() == pred.dim() - 1:
                # target is (B, D, H, W), pred is (B, C, D, H, W)
                # Convert target to one-hot
                num_classes = pred.shape[1]
                target = F.one_hot(target.long(), num_classes).permute(0, -1, *range(1, target.dim())).float()
                
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        # Compute Dice
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        
        if weight is not None:
            loss = loss * weight
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss for class-imbalanced segmentation.
    
    Weights each class by the inverse of its volume.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        softmax: bool = True,
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.softmax = softmax
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Dice Loss.
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, D, H, W) - class indices
            
        Returns:
            Generalized Dice loss
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)
            
        num_classes = pred.shape[1]
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, -1, *range(1, target.dim())).float()
        
        # Flatten
        pred_flat = pred.reshape(pred.shape[0], num_classes, -1)
        target_flat = target_one_hot.reshape(target_one_hot.shape[0], num_classes, -1)
        
        # Compute weights (inverse of volume)
        weights = 1.0 / (target_flat.sum(dim=2) ** 2 + self.smooth)
        
        # Compute weighted Dice
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        numerator = (weights * intersection).sum(dim=1)
        denominator = (weights * union).sum(dim=1)
        
        dice = 2.0 * numerator / (denominator + self.smooth)
        loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Test Dice Loss
    pred = torch.randn(2, 2, 32, 32, 32)
    target = torch.randint(0, 2, (2, 32, 32, 32))
    
    dice_loss = DiceLoss(sigmoid=False, softmax=True)
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    gdl = GeneralizedDiceLoss()
    loss = gdl(pred, target)
    print(f"Generalized Dice Loss: {loss.item():.4f}")
