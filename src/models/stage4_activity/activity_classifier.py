"""
Activity Classifier
活性分类器

Complete Stage4 classifier for echinococcosis activity assessment.
Binary classification: Active vs Inactive

WHO-IWGE Classification for CE:
- CE1, CE2: Active (requires treatment)
- CE3a, CE3b: Transitional
- CE4, CE5: Inactive (observation)

PNM Classification for AE:
- Active: Growing, infiltrating
- Inactive: Stable, calcified
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .boundary_analyzer import BoundaryFeatureExtractor
from .internal_analyzer import InternalStructureAnalyzer


class ActivityClassifier(nn.Module):
    """Evidential classifier for activity status.
    
    Binary classification with uncertainty estimation:
    - 0: Active (活动性)
    - 1: Inactive (非活动性)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        prior_scale: float = 1.0,
    ):
        """Initialize activity classifier.
        
        Args:
            in_features: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            prior_scale: Dirichlet prior scale
        """
        super().__init__()
        
        self.num_classes = 2
        self.prior_scale = prior_scale
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        # Build classifier
        layers = []
        prev_dim = in_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 2))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize for near-uniform output
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features (B, D)
            
        Returns:
            Dictionary with classification outputs
        """
        # Compute evidence
        evidence = F.softplus(self.classifier(x))
        
        # Dirichlet parameters
        alpha = evidence + self.prior_scale
        
        # Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Probabilities
        probs = alpha / S
        
        # Uncertainty
        uncertainty = self.num_classes / S.squeeze(1)
        
        # Prediction
        pred = alpha.argmax(dim=1)
        
        return {
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty,
            'pred': pred,
            'evidence': evidence,
            'active_prob': probs[:, 0],  # Probability of being active
            'inactive_prob': probs[:, 1],  # Probability of being inactive
        }


class EchinococcosisActivityNet(nn.Module):
    """Complete network for echinococcosis activity assessment.
    
    Combines:
    - Boundary feature analysis (wall characteristics)
    - Internal structure analysis (density, calcification, etc.)
    - Stage3 features (from classification)
    - Lesion type information (CE vs AE)
    
    Type-specific processing:
    - CE: Focus on wall integrity, sub-cyst morphology, calcification
    - AE: Focus on infiltration boundary, necrosis, calcification distribution
    """
    
    CLASS_NAMES = ['active', 'inactive']
    CLASS_NAMES_CN = ['活动性', '非活动性']
    
    def __init__(
        self,
        boundary_output_dim: int = 128,
        internal_output_dim: int = 128,
        stage3_feature_dim: int = 256,
        classifier_hidden_dims: List[int] = None,
        dropout: float = 0.2,
        use_3d: bool = True,
    ):
        """Initialize activity network.
        
        Args:
            boundary_output_dim: Output dim for boundary analyzer
            internal_output_dim: Output dim for internal analyzer
            stage3_feature_dim: Dimension of Stage3 features
            classifier_hidden_dims: Hidden dims for classifier
            dropout: Dropout rate
            use_3d: Whether to use 3D convolutions
        """
        super().__init__()
        
        # Boundary feature extractor
        self.boundary_analyzer = BoundaryFeatureExtractor(
            in_channels=2,
            hidden_channels=[32, 64, 128],
            output_dim=boundary_output_dim,
            use_3d=use_3d,
        )
        
        # Internal structure analyzer
        self.internal_analyzer = InternalStructureAnalyzer(
            in_channels=2,
            hidden_channels=[32, 64, 128],
            output_dim=internal_output_dim,
            use_3d=use_3d,
        )
        
        # Lesion type embedding (CE=0, AE=1)
        self.type_embedding = nn.Embedding(2, 32)
        
        # Feature fusion
        total_features = boundary_output_dim + internal_output_dim + stage3_feature_dim + 32
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        
        # Activity classifier
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [128, 64]
            
        self.classifier = ActivityClassifier(
            in_features=256,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout,
        )
        
        # Type-specific attention
        self.ce_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )
        
        self.ae_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
        stage3_features: torch.Tensor,
        lesion_type: torch.Tensor,  # 0=CE, 1=AE
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            ct: CT volume (B, 1, D, H, W)
            mask: Lesion mask (B, 1, D, H, W)
            stage3_features: Features from Stage3 (B, D)
            lesion_type: Lesion type indices (B,) - 0=CE, 1=AE
            
        Returns:
            Dictionary with classification outputs and interpretable features
        """
        B = ct.size(0)
        
        # Extract boundary features
        boundary_features, boundary_info = self.boundary_analyzer(ct, mask)
        
        # Extract internal features
        internal_features, internal_info = self.internal_analyzer(ct, mask)
        
        # Type embedding
        type_embed = self.type_embedding(lesion_type)
        
        # Concatenate all features
        combined = torch.cat([
            boundary_features,
            internal_features,
            stage3_features,
            type_embed,
        ], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Apply type-specific attention
        ce_mask = (lesion_type == 0).float().unsqueeze(1)
        ae_mask = (lesion_type == 1).float().unsqueeze(1)
        
        ce_attn = self.ce_attention(fused)
        ae_attn = self.ae_attention(fused)
        
        # Weighted combination based on lesion type
        attended_features = fused * (ce_mask * ce_attn + ae_mask * ae_attn)
        
        # Classification
        output = self.classifier(attended_features)
        
        # Add interpretable features
        output['boundary_features'] = boundary_info
        output['internal_features'] = internal_info
        output['lesion_type'] = lesion_type
        output['fused_features'] = fused
        
        return output
        
    def get_activity_assessment(
        self,
        output: Dict[str, torch.Tensor],
        lesion_type: torch.Tensor,
    ) -> Dict[str, any]:
        """Generate human-readable activity assessment.
        
        Args:
            output: Model output
            lesion_type: Lesion type (0=CE, 1=AE)
            
        Returns:
            Assessment dictionary
        """
        B = output['pred'].size(0)
        assessments = []
        
        for b in range(B):
            assessment = {
                'activity_status': 'active' if output['pred'][b] == 0 else 'inactive',
                'activity_status_cn': '活动性' if output['pred'][b] == 0 else '非活动性',
                'active_probability': output['active_prob'][b].item(),
                'inactive_probability': output['inactive_prob'][b].item(),
                'uncertainty': output['uncertainty'][b].item(),
                'lesion_type': 'CE' if lesion_type[b] == 0 else 'AE',
            }
            
            # Add type-specific interpretation
            if lesion_type[b] == 0:  # CE
                calcification = output['internal_features']['calcification_ratio'][b].item()
                
                if output['pred'][b] == 0:  # Active
                    if calcification < 0.1:
                        assessment['who_stage_likely'] = 'CE1/CE2'
                        assessment['interpretation'] = '活动性囊型包虫病，建议治疗'
                    else:
                        assessment['who_stage_likely'] = 'CE3'
                        assessment['interpretation'] = '过渡期囊型包虫病，需密切观察'
                else:  # Inactive
                    if calcification > 0.5:
                        assessment['who_stage_likely'] = 'CE5'
                        assessment['interpretation'] = '钙化型囊型包虫病，建议观察'
                    else:
                        assessment['who_stage_likely'] = 'CE4'
                        assessment['interpretation'] = '非活动性囊型包虫病，建议观察'
                        
            else:  # AE
                infiltration = output['boundary_features'].get('infiltration_score', torch.tensor([0.0]))[b].item() if 'infiltration_score' in output['boundary_features'] else 0.0
                
                if output['pred'][b] == 0:  # Active
                    assessment['interpretation'] = '活动性泡型包虫病，浸润性生长，建议积极治疗'
                    assessment['infiltration_risk'] = 'high' if infiltration > 0.5 else 'moderate'
                else:  # Inactive
                    assessment['interpretation'] = '非活动性泡型包虫病，建议定期随访'
                    assessment['infiltration_risk'] = 'low'
                    
            # Confidence level
            max_prob = max(assessment['active_probability'], assessment['inactive_probability'])
            if max_prob > 0.9 and assessment['uncertainty'] < 0.5:
                assessment['confidence'] = 'high'
            elif max_prob > 0.7:
                assessment['confidence'] = 'moderate'
            else:
                assessment['confidence'] = 'low'
                assessment['recommendation'] = '建议人工复核'
                
            assessments.append(assessment)
            
        return assessments


class Stage4Loss(nn.Module):
    """Loss function for Stage4 activity classification."""
    
    def __init__(
        self,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
        class_weights: Optional[List[float]] = None,
    ):
        """Initialize loss.
        
        Args:
            kl_weight: Weight for KL divergence
            annealing_epochs: Epochs for KL annealing
            class_weights: Optional class weights [active_weight, inactive_weight]
        """
        super().__init__()
        
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
            
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss.
        
        Args:
            output: Model output
            target: Ground truth (0=active, 1=inactive)
            epoch: Current epoch
            
        Returns:
            Tuple of (loss, info_dict)
        """
        alpha = output['alpha']
        B = alpha.size(0)
        num_classes = 2
        
        # One-hot target
        target_onehot = F.one_hot(target, num_classes).float()
        
        # Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected probability
        p = alpha / S
        
        # MSE loss (Type II MLE)
        err = (target_onehot - p) ** 2
        var = p * (1 - p) / (S + 1)
        mse_loss = (err + var).sum(dim=1)
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights[target]
            mse_loss = mse_loss * weights
            
        mse_loss = mse_loss.mean()
        
        # KL divergence
        alpha_tilde = target_onehot + (1 - target_onehot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl = torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(
            torch.tensor(num_classes, dtype=alpha.dtype, device=alpha.device)
        )
        kl -= torch.lgamma(alpha_tilde).sum(dim=1)
        kl += ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(dim=1)
        kl_loss = kl.mean()
        
        # Annealing
        annealing = min(1.0, epoch / max(1, self.annealing_epochs))
        
        # Total loss
        total_loss = mse_loss + self.kl_weight * annealing * kl_loss
        
        # Accuracy
        pred = output['pred']
        accuracy = (pred == target).float().mean()
        
        info = {
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy.item(),
            'annealing': annealing,
        }
        
        return total_loss, info


if __name__ == "__main__":
    # Test activity network
    print("Testing Echinococcosis Activity Network:")
    
    model = EchinococcosisActivityNet(
        boundary_output_dim=128,
        internal_output_dim=128,
        stage3_feature_dim=256,
        use_3d=True,
    )
    
    # Simulate inputs
    batch_size = 4
    ct = torch.randn(batch_size, 1, 32, 64, 64) * 50 + 30
    mask = (torch.rand(batch_size, 1, 32, 64, 64) > 0.7).float()
    stage3_features = torch.randn(batch_size, 256)
    lesion_type = torch.randint(0, 2, (batch_size,))  # 0=CE, 1=AE
    
    # Forward pass
    output = model(ct, mask, stage3_features, lesion_type)
    
    print(f"\nInput shapes:")
    print(f"  CT: {ct.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Stage3 features: {stage3_features.shape}")
    print(f"  Lesion type: {lesion_type}")
    
    print(f"\nOutput:")
    print(f"  Predictions: {output['pred']}")
    print(f"  Active probs: {output['active_prob']}")
    print(f"  Inactive probs: {output['inactive_prob']}")
    print(f"  Uncertainty: {output['uncertainty']}")
    
    # Get assessments
    assessments = model.get_activity_assessment(output, lesion_type)
    print("\nAssessments:")
    for i, assessment in enumerate(assessments):
        print(f"\n  Sample {i}:")
        for key, value in assessment.items():
            print(f"    {key}: {value}")
            
    # Test loss
    print("\nTesting Loss:")
    loss_fn = Stage4Loss(kl_weight=0.1)
    target = torch.randint(0, 2, (batch_size,))
    
    loss, info = loss_fn(output, target, epoch=5)
    print(f"Loss: {loss.item():.4f}")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
        
    # Model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {num_params / 1e6:.2f}M")
