"""
Evidential Classifier
Evidential分类器

Evidential deep learning based classifier for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EvidentialClassifier(nn.Module):
    """Evidential Classification Head with uncertainty estimation.
    
    Uses Dirichlet distribution for uncertainty estimation:
    - α (alpha): Dirichlet parameters = evidence + 1
    - Prediction: p = α / S, where S = sum(α)
    - Uncertainty: u = K / S, where K = number of classes
    
    Output classes:
    0: 良性 (Benign)
    1: 恶性 (Malignant)
    2: 肝囊型包虫病 (Cystic Echinococcosis, CE)
    3: 肝泡型包虫病 (Alveolar Echinococcosis, AE)
    """
    
    CLASS_NAMES = ['benign', 'malignant', 'cystic_echinococcosis', 'alveolar_echinococcosis']
    CLASS_NAMES_CN = ['良性', '恶性', '肝囊型包虫病', '肝泡型包虫病']
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 4,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2,
        prior_scale: float = 1.0,
    ):
        """Initialize evidential classifier.
        
        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            prior_scale: Scale for Dirichlet prior
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.prior_scale = prior_scale
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        # Build MLP
        layers = []
        prev_dim = in_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
            
        # Output layer (produces evidence)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize output layer to produce near-uniform predictions
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features (B, D)
            
        Returns:
            Dictionary with:
            - alpha: Dirichlet parameters (B, K)
            - probs: Class probabilities (B, K)
            - uncertainty: Per-sample uncertainty (B,)
            - pred: Predicted class indices (B,)
        """
        # Compute evidence (must be non-negative)
        evidence = F.softplus(self.mlp(x))
        
        # Dirichlet parameters (alpha = evidence + prior)
        alpha = evidence + self.prior_scale
        
        # Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Class probabilities (expected value of Dirichlet)
        probs = alpha / S
        
        # Uncertainty (inverse of total evidence)
        uncertainty = self.num_classes / S.squeeze(1)
        
        # Predicted class
        pred = alpha.argmax(dim=1)
        
        return {
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty,
            'pred': pred,
            'evidence': evidence,
        }
        
    def get_detailed_uncertainty(
        self,
        alpha: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed uncertainty metrics.
        
        Args:
            alpha: Dirichlet parameters (B, K)
            
        Returns:
            Dictionary with various uncertainty metrics
        """
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        
        # Vacuity (lack of evidence) - higher when total evidence is low
        vacuity = self.num_classes / S.squeeze(1)
        
        # Dissonance (conflicting evidence) - higher when probabilities are similar
        # Using entropy as a proxy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=probs.dtype, device=probs.device))
        dissonance = entropy / max_entropy
        
        # Aleatoric uncertainty (data uncertainty) - from Dirichlet variance
        aleatoric = (probs * (1 - probs) / (S + 1)).mean(dim=1)
        
        # Epistemic uncertainty (model uncertainty) - vacuity
        epistemic = vacuity
        
        return {
            'vacuity': vacuity,
            'dissonance': dissonance,
            'entropy': entropy,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_uncertainty': vacuity + dissonance,
        }


class EvidentialLoss(nn.Module):
    """Loss function for evidential classification.
    
    Combines:
    - Type II Maximum Likelihood loss (MSE-based)
    - KL divergence regularization
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
    ):
        """Initialize loss function.
        
        Args:
            num_classes: Number of classes
            kl_weight: Weight for KL divergence term
            annealing_epochs: Epochs for KL weight annealing
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        
    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss.
        
        Args:
            alpha: Dirichlet parameters (B, K)
            target: Class indices (B,)
            epoch: Current epoch for annealing
            
        Returns:
            Tuple of (loss, info_dict)
        """
        B = alpha.size(0)
        device = alpha.device
        
        # One-hot encode target
        target_onehot = F.one_hot(target, self.num_classes).float()
        
        # Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected probability
        p = alpha / S
        
        # Type II Maximum Likelihood loss
        # E[(y - p)^2] = (y - p)^2 + p(1-p)/(S+1)
        err = (target_onehot - p) ** 2
        var = p * (1 - p) / (S + 1)
        mse_loss = (err + var).sum(dim=1).mean()
        
        # KL divergence from uniform Dirichlet
        # Remove evidence from non-target classes
        alpha_tilde = target_onehot + (1 - target_onehot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        # KL(Dir(alpha_tilde) || Dir(1))
        kl = torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(
            torch.tensor(self.num_classes, dtype=alpha.dtype, device=device)
        )
        kl -= torch.lgamma(alpha_tilde).sum(dim=1)
        kl += ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(dim=1)
        kl_loss = kl.mean()
        
        # Annealing coefficient
        annealing = min(1.0, epoch / max(1, self.annealing_epochs))
        
        # Total loss
        total_loss = mse_loss + self.kl_weight * annealing * kl_loss
        
        # Info dict
        info = {
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item(),
            'annealing': annealing,
            'total_loss': total_loss.item(),
        }
        
        return total_loss, info


class SliceAuxiliaryLoss(nn.Module):
    """Auxiliary loss for slice-level predictions.
    
    Encourages consistency between slice predictions.
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        slice_logits: torch.Tensor,  # (B, K, num_classes)
        target: torch.Tensor,  # (B,)
        slice_weights: Optional[torch.Tensor] = None,  # (B, K)
    ) -> torch.Tensor:
        """Compute slice auxiliary loss.
        
        Args:
            slice_logits: Per-slice logits
            target: Global target
            slice_weights: Optional importance weights for slices
            
        Returns:
            Loss value
        """
        B, K, C = slice_logits.shape
        
        # Expand target for all slices
        target_expanded = target.unsqueeze(1).expand(-1, K)  # (B, K)
        
        # Reshape for cross entropy
        slice_logits_flat = slice_logits.reshape(B * K, C)
        target_flat = target_expanded.reshape(B * K)
        
        # Compute loss
        loss = self.ce(slice_logits_flat, target_flat).reshape(B, K)
        
        # Weight by slice importance
        if slice_weights is not None:
            loss = loss * slice_weights
            
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """Consistency loss between slice predictions."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        slice_logits: torch.Tensor,  # (B, K, num_classes)
    ) -> torch.Tensor:
        """Compute consistency loss.
        
        Encourages similar predictions across slices.
        
        Args:
            slice_logits: Per-slice logits
            
        Returns:
            Loss value
        """
        # Soft predictions
        probs = F.softmax(slice_logits / self.temperature, dim=-1)
        
        # Mean prediction
        mean_probs = probs.mean(dim=1, keepdim=True)
        
        # KL divergence from mean
        kl = F.kl_div(
            (probs + 1e-10).log(),
            mean_probs.expand_as(probs),
            reduction='batchmean',
        )
        
        return kl


if __name__ == "__main__":
    # Test evidential classifier
    print("Testing Evidential Classifier:")
    classifier = EvidentialClassifier(
        in_features=512,
        num_classes=4,
        hidden_dims=[256, 128, 64],
    )
    
    batch_size = 8
    features = torch.randn(batch_size, 512)
    
    output = classifier(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Alpha shape: {output['alpha'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Predictions: {output['pred']}")
    print(f"Uncertainty range: [{output['uncertainty'].min():.3f}, {output['uncertainty'].max():.3f}]")
    
    # Test detailed uncertainty
    detailed = classifier.get_detailed_uncertainty(output['alpha'])
    print("\nDetailed uncertainty metrics:")
    for key, value in detailed.items():
        print(f"  {key}: mean={value.mean():.4f}")
        
    # Test loss
    print("\nTesting Evidential Loss:")
    loss_fn = EvidentialLoss(num_classes=4, kl_weight=0.1)
    target = torch.randint(0, 4, (batch_size,))
    
    loss, info = loss_fn(output['alpha'], target, epoch=5)
    print(f"Loss: {loss.item():.4f}")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
        
    # Model parameters
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nClassifier parameters: {num_params / 1e6:.4f}M")
