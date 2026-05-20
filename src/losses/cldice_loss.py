"""
clDice Loss (Centerline Dice)
clDice损失函数

Preserves topological connectivity in segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """Soft skeletonization using iterative erosion.
    
    Differentiable approximation of morphological skeletonization.
    """
    
    def __init__(self, num_iters: int = 10):
        """Initialize soft skeletonization.
        
        Args:
            num_iters: Number of erosion iterations
        """
        super().__init__()
        self.num_iters = num_iters
        
        # 3D erosion kernel
        kernel = torch.ones(1, 1, 3, 3, 3)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft skeleton.
        
        Args:
            x: Input mask (B, 1, D, H, W)
            
        Returns:
            Soft skeleton
        """
        skeleton = x.clone()
        
        for _ in range(self.num_iters):
            # Soft erosion
            eroded = F.conv3d(skeleton, self.kernel, padding=1)
            
            # Keep only skeleton points
            skeleton = skeleton - (skeleton - eroded).clamp(min=0)
            skeleton = skeleton.clamp(0, 1)
            
        return skeleton


class clDiceLoss(nn.Module):
    """clDice Loss for topology-preserving segmentation.
    
    clDice = 2 * |Skel(P) ∩ Skel(G)| / (|Skel(P)| + |Skel(G)|)
    
    Where Skel() is the skeleton/centerline extraction.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        skel_iters: int = 10,
        include_background: bool = False,
        sigmoid: bool = True,
    ):
        """Initialize clDice Loss.
        
        Args:
            smooth: Smoothing factor
            skel_iters: Number of skeletonization iterations
            include_background: Whether to include background class
            sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.smooth = smooth
        self.skeletonize = SoftSkeletonize(num_iters=skel_iters)
        self.include_background = include_background
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clDice Loss.
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, C, D, H, W)
            
        Returns:
            clDice loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Get skeletons
        pred_skel = self.skeletonize(pred)
        target_skel = self.skeletonize(target.float())
        
        # Compute topological precision (skeleton overlap)
        # Tprec = |Skel(G) ∩ P| / |Skel(G)|
        tprec = ((pred * target_skel).sum(dim=(2, 3, 4)) + self.smooth) / \
                (target_skel.sum(dim=(2, 3, 4)) + self.smooth)
        
        # Compute topological recall
        # Tsens = |Skel(P) ∩ G| / |Skel(P)|
        tsens = ((target.float() * pred_skel).sum(dim=(2, 3, 4)) + self.smooth) / \
                (pred_skel.sum(dim=(2, 3, 4)) + self.smooth)
        
        # clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        
        if not self.include_background:
            # Skip background channel
            cl_dice = cl_dice[:, 1:]
            
        loss = 1 - cl_dice.mean()
        return loss


class SoftClDiceLoss(nn.Module):
    """Combined Soft Dice + clDice Loss.
    
    Balances volumetric accuracy with topological preservation.
    """
    
    def __init__(
        self,
        soft_dice_weight: float = 0.5,
        cl_dice_weight: float = 0.5,
        smooth: float = 1.0,
        skel_iters: int = 10,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.soft_dice_weight = soft_dice_weight
        self.cl_dice_weight = cl_dice_weight
        self.smooth = smooth
        self.cl_dice = clDiceLoss(smooth=smooth, skel_iters=skel_iters, sigmoid=False)
        self.sigmoid = sigmoid
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            
        Returns:
            Combined loss
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Soft Dice
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.float().reshape(target.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        soft_dice = (2 * intersection + self.smooth) / (union + self.smooth)
        soft_dice_loss = 1 - soft_dice.mean()
        
        # clDice
        cl_dice_loss = self.cl_dice(pred, target)
        
        return self.soft_dice_weight * soft_dice_loss + self.cl_dice_weight * cl_dice_loss


if __name__ == "__main__":
    # Test
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    
    cldice = clDiceLoss()
    loss = cldice(pred, target)
    print(f"clDice Loss: {loss.item():.4f}")
    
    combined = SoftClDiceLoss()
    loss = combined(pred, target)
    print(f"Combined Loss: {loss.item():.4f}")
