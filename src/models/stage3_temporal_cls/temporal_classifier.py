"""
Temporal Lesion Classifier
时序病灶分类器

Complete Stage3 classifier combining all components:
- Mask-guided encoder for slice features
- Morphology encoder for geometric features
- Temporal encoder for sequence modeling
- Evidential classifier for uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .mask_guided_encoder import MaskGuidedEncoder, extract_mask_boundary
from .morphology_encoder import MorphologyEncoder
from .temporal_encoder import TemporalEncoder, TemporalAggregator
from .evidential_classifier import EvidentialClassifier, EvidentialLoss, SliceAuxiliaryLoss, ConsistencyLoss


class FeatureFusion(nn.Module):
    """Fuses slice features with morphology features."""
    
    def __init__(
        self,
        slice_dim: int,
        morphology_dim: int,
        output_dim: int,
        fusion_method: str = 'concat_linear',
    ):
        """Initialize feature fusion.
        
        Args:
            slice_dim: Dimension of slice features
            morphology_dim: Dimension of morphology features
            output_dim: Output dimension
            fusion_method: 'concat_linear', 'cross_attention', or 'gated'
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat_linear':
            self.fusion = nn.Sequential(
                nn.Linear(slice_dim + morphology_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(output_dim),
            )
            
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(slice_dim + morphology_dim, output_dim),
                nn.Sigmoid(),
            )
            self.transform = nn.Linear(slice_dim + morphology_dim, output_dim)
            
        elif fusion_method == 'cross_attention':
            self.query_proj = nn.Linear(slice_dim, output_dim)
            self.key_proj = nn.Linear(morphology_dim, output_dim)
            self.value_proj = nn.Linear(morphology_dim, output_dim)
            self.output_proj = nn.Linear(output_dim, output_dim)
            
    def forward(
        self,
        slice_features: torch.Tensor,  # (B, K, D_slice) or (B, D_slice)
        morphology_features: torch.Tensor,  # (B, D_morph)
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            slice_features: Slice features (can be per-slice or aggregated)
            morphology_features: Morphology features
            
        Returns:
            Fused features
        """
        # Expand morphology features if slice features are per-slice
        if slice_features.dim() == 3:
            B, K, _ = slice_features.shape
            morph_expanded = morphology_features.unsqueeze(1).expand(-1, K, -1)
        else:
            morph_expanded = morphology_features
            
        if self.fusion_method == 'concat_linear':
            combined = torch.cat([slice_features, morph_expanded], dim=-1)
            return self.fusion(combined)
            
        elif self.fusion_method == 'gated':
            combined = torch.cat([slice_features, morph_expanded], dim=-1)
            gate = self.gate(combined)
            transform = self.transform(combined)
            return gate * transform
            
        elif self.fusion_method == 'cross_attention':
            # Use morphology as key/value, slice as query
            q = self.query_proj(slice_features)
            k = self.key_proj(morphology_features)
            v = self.value_proj(morphology_features)
            
            if q.dim() == 3:
                # Per-slice attention
                attn = torch.einsum('bkd,bd->bk', q, k) / (q.size(-1) ** 0.5)
                attn = F.softmax(attn, dim=-1)
                out = torch.einsum('bk,bd->bkd', attn, v)
            else:
                out = v
                
            return self.output_proj(out + slice_features[..., :out.size(-1)])


class TemporalLesionClassifier(nn.Module):
    """Complete Temporal Lesion Classifier for Stage3.
    
    4-class classification:
    - 0: 良性 (Benign)
    - 1: 恶性 (Malignant)
    - 2: 肝囊型包虫病 (Cystic Echinococcosis)
    - 3: 肝泡型包虫病 (Alveolar Echinococcosis)
    """
    
    CLASS_NAMES = ['benign', 'malignant', 'cystic_echinococcosis', 'alveolar_echinococcosis']
    CLASS_NAMES_CN = ['良性', '恶性', '肝囊型包虫病', '肝泡型包虫病']
    
    def __init__(
        self,
        # Encoder settings
        slice_encoder_variant: str = 'tiny',
        slice_encoder_pretrained: Optional[str] = None,
        include_boundary: bool = True,
        # Morphology settings
        num_morphology_features: int = 16,
        morphology_hidden_dims: List[int] = None,
        morphology_output_dim: int = 256,
        # Temporal settings
        temporal_hidden_dim: int = 512,
        temporal_num_layers: int = 4,
        temporal_num_heads: int = 8,
        temporal_dropout: float = 0.1,
        # Classifier settings
        num_classes: int = 4,
        classifier_hidden_dims: List[int] = None,
        classifier_dropout: float = 0.2,
        # Other settings
        fusion_method: str = 'concat_linear',
        aggregation_type: str = 'attention_pooling',
    ):
        """Initialize temporal lesion classifier.
        
        Args:
            slice_encoder_variant: ConvNeXt variant for slice encoder
            slice_encoder_pretrained: Path to pretrained weights
            include_boundary: Whether to include mask boundary
            num_morphology_features: Number of morphology features
            morphology_hidden_dims: Hidden dims for morphology encoder
            morphology_output_dim: Output dim for morphology encoder
            temporal_hidden_dim: Hidden dim for temporal encoder
            temporal_num_layers: Number of transformer layers
            temporal_num_heads: Number of attention heads
            temporal_dropout: Dropout rate for temporal encoder
            num_classes: Number of output classes
            classifier_hidden_dims: Hidden dims for classifier
            classifier_dropout: Dropout rate for classifier
            fusion_method: Feature fusion method
            aggregation_type: Temporal aggregation type
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Slice encoder (mask-guided)
        self.slice_encoder = MaskGuidedEncoder(
            backbone_variant=slice_encoder_variant,
            pretrained_path=slice_encoder_pretrained,
            include_boundary=include_boundary,
        )
        slice_feature_dim = self.slice_encoder.feature_dim
        
        # Morphology encoder
        if morphology_hidden_dims is None:
            morphology_hidden_dims = [64, 128, 256]
        self.morphology_encoder = MorphologyEncoder(
            num_input_features=num_morphology_features,
            hidden_dims=morphology_hidden_dims,
            output_dim=morphology_output_dim,
        )
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(
            slice_dim=slice_feature_dim,
            morphology_dim=morphology_output_dim,
            output_dim=temporal_hidden_dim,
            fusion_method=fusion_method,
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            d_model=temporal_hidden_dim,
            nhead=temporal_num_heads,
            num_layers=temporal_num_layers,
            dim_feedforward=temporal_hidden_dim * 2,
            dropout=temporal_dropout,
        )
        
        # Temporal aggregator
        self.temporal_aggregator = TemporalAggregator(
            d_model=temporal_hidden_dim,
            aggregation_type=aggregation_type,
            num_heads=4,
        )
        
        # Evidential classifier
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128, 64]
        self.classifier = EvidentialClassifier(
            in_features=temporal_hidden_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
        )
        
        # Auxiliary slice-level classifier (for deep supervision)
        self.slice_classifier = nn.Linear(temporal_hidden_dim, num_classes)
        
    def forward(
        self,
        slices: torch.Tensor,  # (B, K, 1, H, W) - CT slices
        masks: torch.Tensor,  # (B, K, 1, H, W) - Lesion masks
        morphology_features: torch.Tensor,  # (B, num_features)
        boundaries: Optional[torch.Tensor] = None,  # (B, K, 1, H, W)
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            slices: CT slices (B, K, 1, H, W)
            masks: Lesion masks (B, K, 1, H, W)
            morphology_features: Pre-computed morphology features (B, num_features)
            boundaries: Optional mask boundaries
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with classification outputs and intermediate features
        """
        B, K, C, H, W = slices.shape
        
        # Compute boundaries if not provided
        if boundaries is None:
            boundaries = torch.stack([
                torch.stack([extract_mask_boundary(masks[b, k:k+1]) for k in range(K)])
                for b in range(B)
            ])
            
        # Encode each slice
        slice_features = []
        for k in range(K):
            feat = self.slice_encoder(
                slices[:, k],
                masks[:, k],
                boundaries[:, k] if boundaries is not None else None,
            )
            slice_features.append(feat)
        slice_features = torch.stack(slice_features, dim=1)  # (B, K, D_slice)
        
        # Encode morphology features
        morph_features = self.morphology_encoder(morphology_features)  # (B, D_morph)
        
        # Fuse features
        fused_features = self.feature_fusion(slice_features, morph_features)  # (B, K, D)
        
        # Temporal encoding
        temporal_features, temporal_attn = self.temporal_encoder(
            fused_features, return_attention=return_attention
        )  # (B, K, D)
        
        # Auxiliary slice predictions (for deep supervision)
        slice_logits = self.slice_classifier(temporal_features)  # (B, K, num_classes)
        
        # Temporal aggregation
        aggregated_features, slice_weights = self.temporal_aggregator(
            temporal_features, return_weights=True
        )  # (B, D)
        
        # Classification
        cls_output = self.classifier(aggregated_features)
        
        # Build output dictionary
        output = {
            **cls_output,
            'slice_features': slice_features,
            'temporal_features': temporal_features,
            'aggregated_features': aggregated_features,
            'slice_logits': slice_logits,
            'slice_weights': slice_weights,
        }
        
        if return_attention and temporal_attn is not None:
            output['temporal_attention'] = temporal_attn
            
        return output
    
    def get_interpretability_info(
        self,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract interpretability information.
        
        Args:
            output: Model output dictionary
            
        Returns:
            Dictionary with interpretability info
        """
        # Slice importance (from aggregator weights)
        slice_importance = output['slice_weights']  # (B, K)
        
        # Top-K important slices
        top_k = min(5, slice_importance.size(1))
        top_values, top_indices = slice_importance.topk(top_k, dim=1)
        
        # Per-class evidence
        evidence = output['evidence']
        
        # Uncertainty decomposition
        uncertainty_info = self.classifier.get_detailed_uncertainty(output['alpha'])
        
        return {
            'slice_importance': slice_importance,
            'top_slice_indices': top_indices,
            'top_slice_weights': top_values,
            'evidence_per_class': evidence,
            **uncertainty_info,
        }


class Stage3Loss(nn.Module):
    """Combined loss for Stage3 training.
    
    Combines:
    - Evidential classification loss
    - Slice auxiliary loss (deep supervision)
    - Consistency loss
    - Attention regularization
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        evidential_weight: float = 1.0,
        slice_aux_weight: float = 0.3,
        consistency_weight: float = 0.2,
        attention_reg_weight: float = 0.1,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
    ):
        super().__init__()
        
        self.evidential_loss = EvidentialLoss(
            num_classes=num_classes,
            kl_weight=kl_weight,
            annealing_epochs=annealing_epochs,
        )
        self.slice_aux_loss = SliceAuxiliaryLoss(num_classes=num_classes)
        self.consistency_loss = ConsistencyLoss()
        
        self.evidential_weight = evidential_weight
        self.slice_aux_weight = slice_aux_weight
        self.consistency_weight = consistency_weight
        self.attention_reg_weight = attention_reg_weight
        
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            output: Model output dictionary
            target: Ground truth class indices (B,)
            epoch: Current epoch
            
        Returns:
            Tuple of (total_loss, loss_info)
        """
        # Evidential loss
        evid_loss, evid_info = self.evidential_loss(output['alpha'], target, epoch)
        
        # Slice auxiliary loss
        slice_loss = self.slice_aux_loss(
            output['slice_logits'],
            target,
            output.get('slice_weights'),
        )
        
        # Consistency loss
        consist_loss = self.consistency_loss(output['slice_logits'])
        
        # Attention regularization (encourage sparse attention)
        if output.get('slice_weights') is not None:
            weights = output['slice_weights']
            entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=1).mean()
            attn_reg = -entropy  # Negative entropy to encourage sparsity
        else:
            attn_reg = torch.tensor(0.0, device=target.device)
            
        # Total loss
        total_loss = (
            self.evidential_weight * evid_loss +
            self.slice_aux_weight * slice_loss +
            self.consistency_weight * consist_loss +
            self.attention_reg_weight * attn_reg
        )
        
        info = {
            **evid_info,
            'slice_aux_loss': slice_loss.item(),
            'consistency_loss': consist_loss.item(),
            'attention_reg': attn_reg.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, info


if __name__ == "__main__":
    # Test complete classifier
    print("Testing Temporal Lesion Classifier:")
    
    classifier = TemporalLesionClassifier(
        slice_encoder_variant='tiny',
        num_morphology_features=16,
        temporal_hidden_dim=256,
        temporal_num_layers=2,
        temporal_num_heads=4,
        num_classes=4,
    )
    
    # Simulate inputs
    batch_size = 4
    num_slices = 8
    image_size = 112  # Smaller for testing
    
    slices = torch.randn(batch_size, num_slices, 1, image_size, image_size)
    masks = (torch.rand(batch_size, num_slices, 1, image_size, image_size) > 0.7).float()
    morphology_features = torch.randn(batch_size, 16)
    
    # Forward pass
    output = classifier(slices, masks, morphology_features, return_attention=True)
    
    print(f"\nInput shapes:")
    print(f"  Slices: {slices.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Morphology: {morphology_features.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Alpha: {output['alpha'].shape}")
    print(f"  Probs: {output['probs'].shape}")
    print(f"  Uncertainty: {output['uncertainty'].shape}")
    print(f"  Predictions: {output['pred']}")
    print(f"  Slice logits: {output['slice_logits'].shape}")
    print(f"  Slice weights: {output['slice_weights'].shape}")
    
    # Test interpretability
    interp_info = classifier.get_interpretability_info(output)
    print(f"\nInterpretability info:")
    print(f"  Top slice indices: {interp_info['top_slice_indices']}")
    print(f"  Top slice weights: {interp_info['top_slice_weights']}")
    print(f"  Vacuity: {interp_info['vacuity'].mean():.4f}")
    print(f"  Dissonance: {interp_info['dissonance'].mean():.4f}")
    
    # Test loss
    print("\nTesting Stage3 Loss:")
    loss_fn = Stage3Loss(num_classes=4)
    target = torch.randint(0, 4, (batch_size,))
    
    loss, info = loss_fn(output, target, epoch=5)
    print(f"Total loss: {loss.item():.4f}")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
        
    # Model parameters
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nTotal model parameters: {num_params / 1e6:.2f}M")
