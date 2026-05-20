"""
Evidential Deep Learning Layers
Evidential深度学习层

For uncertainty estimation in segmentation and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EvidentialSegmentationHead(nn.Module):
    """Evidential segmentation head for uncertainty estimation.
    
    Outputs Dirichlet distribution parameters (alpha) for each pixel.
    
    Prediction: p = α / S, where S = sum(α)
    Uncertainty: u = K / S, where K = number of classes
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_channels: Optional[int] = None,
        prior_scale: float = 1.0,
    ):
        """Initialize evidential segmentation head.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            hidden_channels: Hidden layer channels (default: in_channels // 2)
            prior_scale: Scale for the Dirichlet prior
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.prior_scale = prior_scale
        
        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)
            
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)
        
        # Initialize to produce alpha ≈ 1 (uniform Dirichlet)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features (B, C, D, H, W)
            
        Returns:
            Tuple of:
            - alpha: Dirichlet parameters (B, K, D, H, W)
            - prediction: Class probabilities (B, K, D, H, W)
            - uncertainty: Per-voxel uncertainty (B, D, H, W)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        evidence = F.softplus(self.conv2(x))  # Ensure positive
        
        # Alpha = evidence + prior
        alpha = evidence + self.prior_scale
        
        # Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Prediction (expected probability)
        prediction = alpha / S
        
        # Uncertainty (inverse of Dirichlet strength)
        uncertainty = self.num_classes / S.squeeze(1)
        
        return alpha, prediction, uncertainty
        
    def get_prediction(self, alpha: torch.Tensor) -> torch.Tensor:
        """Get class prediction from alpha."""
        return alpha.argmax(dim=1)


class EvidentialClassificationHead(nn.Module):
    """Evidential classification head for uncertainty estimation.
    
    For standard classification tasks.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        prior_scale: float = 1.0,
    ):
        """Initialize evidential classification head.
        
        Args:
            in_features: Number of input features
            num_classes: Number of classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            prior_scale: Scale for the Dirichlet prior
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.prior_scale = prior_scale
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        layers = []
        prev_dim = in_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize final layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features (B, D)
            
        Returns:
            Tuple of:
            - alpha: Dirichlet parameters (B, K)
            - prediction: Class probabilities (B, K)
            - uncertainty: Per-sample uncertainty (B,)
        """
        evidence = F.softplus(self.mlp(x))
        alpha = evidence + self.prior_scale
        
        S = alpha.sum(dim=1, keepdim=True)
        prediction = alpha / S
        uncertainty = self.num_classes / S.squeeze(1)
        
        return alpha, prediction, uncertainty
        
    def get_prediction(self, alpha: torch.Tensor) -> torch.Tensor:
        """Get class prediction from alpha."""
        return alpha.argmax(dim=1)


class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Keeps dropout active during inference for uncertainty estimation.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class MCDropoutWrapper(nn.Module):
    """Wrapper for MC Dropout inference.
    
    Performs multiple forward passes with dropout to estimate uncertainty.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """Initialize MC Dropout wrapper.
        
        Args:
            model: Base model with dropout layers
            num_samples: Number of MC samples
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with MC Dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
                
        predictions = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty as prediction variance
        uncertainty = predictions.var(dim=0).mean(dim=1)  # Average over classes
        
        return mean_pred, uncertainty


def compute_evidential_uncertainty(
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute different types of uncertainty from Dirichlet parameters.
    
    Args:
        alpha: Dirichlet parameters
        
    Returns:
        Tuple of:
        - vacuity: Lack of evidence
        - dissonance: Conflicting evidence
        - entropy: Predictive entropy
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    K = alpha.shape[1]
    
    # Vacuity (lack of evidence)
    vacuity = K / S.squeeze(1)
    
    # Dissonance (conflicting evidence)
    # Higher when multiple classes have high probability
    log_p = torch.log(p + 1e-10)
    entropy_component = -(p * log_p).sum(dim=1)
    max_entropy = torch.log(torch.tensor(K, dtype=p.dtype, device=p.device))
    dissonance = entropy_component / max_entropy
    
    # Predictive entropy
    entropy = entropy_component
    
    return vacuity, dissonance, entropy


if __name__ == "__main__":
    # Test evidential segmentation head
    seg_head = EvidentialSegmentationHead(in_channels=256, num_classes=2)
    x = torch.randn(2, 256, 32, 32, 32)
    alpha, pred, uncertainty = seg_head(x)
    print(f"Segmentation - Alpha: {alpha.shape}, Pred: {pred.shape}, Uncertainty: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    
    # Test evidential classification head
    cls_head = EvidentialClassificationHead(in_features=512, num_classes=4)
    x = torch.randn(8, 512)
    alpha, pred, uncertainty = cls_head(x)
    print(f"Classification - Alpha: {alpha.shape}, Pred: {pred.shape}, Uncertainty: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
