"""
Evidential Loss Functions
Evidential损失函数

For uncertainty estimation using evidential deep learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EvidentialLoss(nn.Module):
    """Evidential Loss for segmentation with uncertainty.
    
    Uses Dirichlet distribution for uncertainty estimation.
    Output: α (Dirichlet parameters) where α = evidence + 1
    
    Prediction: p = α / S, where S = sum(α)
    Uncertainty: u = K / S, where K = number of classes
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
    ):
        """Initialize Evidential Loss.
        
        Args:
            num_classes: Number of classes
            kl_weight: Weight for KL divergence term
            annealing_epochs: Epochs for KL weight annealing
        """
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
        
    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Evidential Loss.
        
        Args:
            alpha: Dirichlet parameters (B, K, ...) where K is num_classes
            target: Ground truth class indices (B, ...) or one-hot (B, K, ...)
            epoch: Current epoch for annealing
            
        Returns:
            Tuple of (loss, info_dict)
        """
        if epoch is not None:
            self.current_epoch = epoch
            
        # Ensure alpha > 0
        alpha = alpha + 1e-6
        
        # Compute S (Dirichlet strength)
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot encode target if needed
        if target.dim() == alpha.dim() - 1:
            target_one_hot = F.one_hot(target.long(), self.num_classes)
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target.dim())).float()
        else:
            target_one_hot = target.float()
            
        # Expected probability
        p = alpha / S
        
        # Mean squared error loss (Type II Maximum Likelihood)
        err = (target_one_hot - p) ** 2
        var = p * (1 - p) / (S + 1)
        mse_loss = (err + var).sum(dim=1).mean()
        
        # KL divergence for uncertainty regularization
        kl_div = self._kl_divergence(alpha, target_one_hot)
        
        # Annealing coefficient
        annealing = min(1.0, self.current_epoch / self.annealing_epochs)
        kl_weight = self.kl_weight * annealing
        
        total_loss = mse_loss + kl_weight * kl_div
        
        # Compute uncertainty for info
        uncertainty = self.num_classes / S.squeeze(1)
        
        info = {
            'mse_loss': mse_loss.item(),
            'kl_div': kl_div.item(),
            'uncertainty_mean': uncertainty.mean().item(),
            'annealing': annealing,
        }
        
        return total_loss, info
        
    def _kl_divergence(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from uniform Dirichlet.
        
        Args:
            alpha: Dirichlet parameters
            target: One-hot target
            
        Returns:
            KL divergence
        """
        # Remove evidence from non-target classes
        alpha_tilde = target + (1 - target) * alpha
        
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        # KL from uniform Dirichlet (all alphas = 1)
        kl = torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(torch.tensor(self.num_classes, dtype=alpha.dtype, device=alpha.device))
        kl -= (torch.lgamma(alpha_tilde)).sum(dim=1)
        kl += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1)
        
        return kl.mean()


class EvidentialClassificationLoss(nn.Module):
    """Evidential Loss for classification with uncertainty.
    
    For standard classification tasks (not dense prediction).
    """
    
    def __init__(
        self,
        num_classes: int,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
        
    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Evidential Classification Loss.
        
        Args:
            alpha: Dirichlet parameters (B, K)
            target: Class indices (B,)
            epoch: Current epoch
            
        Returns:
            Tuple of (loss, info_dict)
        """
        if epoch is not None:
            self.current_epoch = epoch
            
        alpha = alpha + 1e-6
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot encode
        target_one_hot = F.one_hot(target.long(), self.num_classes).float()
        
        # Expected probability
        p = alpha / S
        
        # Digamma-based loss
        digamma_term = torch.digamma(S) - torch.digamma(alpha)
        loss = (target_one_hot * digamma_term).sum(dim=1).mean()
        
        # KL divergence
        kl_div = self._kl_divergence(alpha, target_one_hot)
        
        annealing = min(1.0, self.current_epoch / self.annealing_epochs)
        total_loss = loss + self.kl_weight * annealing * kl_div
        
        uncertainty = self.num_classes / S.squeeze(1)
        
        info = {
            'ce_loss': loss.item(),
            'kl_div': kl_div.item(),
            'uncertainty_mean': uncertainty.mean().item(),
        }
        
        return total_loss, info
        
    def _kl_divergence(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        alpha_tilde = target + (1 - target) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl = torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(
            torch.tensor(self.num_classes, dtype=alpha.dtype, device=alpha.device))
        kl -= torch.lgamma(alpha_tilde).sum(dim=1)
        kl += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1)
        
        return kl.mean()


def evidential_prediction(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get prediction and uncertainty from Dirichlet parameters.
    
    Args:
        alpha: Dirichlet parameters (B, K, ...)
        
    Returns:
        Tuple of (predicted_probability, uncertainty)
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    K = alpha.shape[1]
    u = K / S.squeeze(1)
    return p, u


if __name__ == "__main__":
    # Test segmentation loss
    alpha = torch.rand(2, 2, 32, 32, 32) * 10 + 1
    target = torch.randint(0, 2, (2, 32, 32, 32))
    
    loss_fn = EvidentialLoss(num_classes=2)
    loss, info = loss_fn(alpha, target, epoch=5)
    print(f"Evidential Seg Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # Test classification loss
    alpha_cls = torch.rand(8, 4) * 10 + 1
    target_cls = torch.randint(0, 4, (8,))
    
    loss_fn_cls = EvidentialClassificationLoss(num_classes=4)
    loss, info = loss_fn_cls(alpha_cls, target_cls, epoch=5)
    print(f"Evidential Cls Loss: {loss.item():.4f}")
    print(f"Info: {info}")
