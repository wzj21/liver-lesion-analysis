"""
3D Contrastive Learning for Self-Supervised Pretraining
3D对比学习自监督预训练

SimCLR / MoCo style contrastive learning adapted for 3D medical volumes.
Learns representations by maximising agreement between differently augmented
views of the same volume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
import copy
import logging

logger = logging.getLogger(__name__)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Used in SimCLR-style contrastive learning.
    For each positive pair (i, j), all other 2(N-1) samples
    in the batch serve as negatives.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z_i: projections from view 1, (B, D)
            z_j: projections from view 2, (B, D)
            
        Returns:
            Scalar loss
        """
        B = z_i.shape[0]
        device = z_i.device
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        
        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarity
        mask = torch.eye(2 * B, device=device).bool()
        sim.masked_fill_(mask, -1e9)
        
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ])
        
        loss = F.cross_entropy(sim, labels)
        return loss


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Encoder3D(nn.Module):
    """Simple 3D CNN encoder for contrastive learning.
    
    Can be replaced with ConvNeXt3d backbone for better performance.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        stages = []
        ch = in_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            stages.append(nn.Sequential(
                nn.Conv3d(ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
            
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(ch, feature_dim)
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class ContrastiveLearning3D(nn.Module):
    """SimCLR-style Contrastive Learning for 3D medical volumes.
    
    Two augmented views of the same volume are passed through a shared
    encoder and projection head.  The NT-Xent loss maximises agreement
    between the two views.
    
    Usage:
        model = ContrastiveLearning3D(encoder=my_encoder)
        loss = model(view1, view2)['loss']
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        feature_dim: int = 512,
        projection_dim: int = 128,
        projection_hidden: int = 2048,
        temperature: float = 0.07,
        use_momentum: bool = False,
        momentum: float = 0.999,
    ):
        """
        Args:
            encoder: backbone encoder (must output (B, feature_dim))
            feature_dim: encoder output dimension
            projection_dim: projection head output dim
            projection_hidden: projection head hidden dim
            temperature: NT-Xent temperature
            use_momentum: if True, use MoCo-style momentum encoder
            momentum: EMA momentum coefficient
        """
        super().__init__()
        
        if encoder is None:
            encoder = Encoder3D(feature_dim=feature_dim)
            feature_dim = encoder.feature_dim
            
        self.encoder = encoder
        self.projector = ProjectionHead(
            feature_dim, projection_hidden, projection_dim,
        )
        self.criterion = NTXentLoss(temperature)
        
        self.use_momentum = use_momentum
        if use_momentum:
            self.momentum = momentum
            self.encoder_m = copy.deepcopy(encoder)
            self.projector_m = copy.deepcopy(self.projector)
            # Freeze momentum networks
            for p in self.encoder_m.parameters():
                p.requires_grad = False
            for p in self.projector_m.parameters():
                p.requires_grad = False
                
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder."""
        for p, pm in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            pm.data = self.momentum * pm.data + (1.0 - self.momentum) * p.data
        for p, pm in zip(self.projector.parameters(), self.projector_m.parameters()):
            pm.data = self.momentum * pm.data + (1.0 - self.momentum) * p.data
            
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            view1: first augmented view  (B, C, D, H, W)
            view2: second augmented view (B, C, D, H, W)
            
        Returns:
            dict with 'loss', 'z1', 'z2'
        """
        # Online encoder
        h1 = self.encoder(view1)
        z1 = self.projector(h1)
        
        if self.use_momentum:
            self._momentum_update()
            with torch.no_grad():
                h2 = self.encoder_m(view2)
                z2 = self.projector_m(h2)
        else:
            h2 = self.encoder(view2)
            z2 = self.projector(h2)
            
        loss = self.criterion(z1, z2)
        
        return {
            'loss': loss,
            'z1': z1,
            'z2': z2,
            'h1': h1,
            'h2': h2,
        }


class ContrastiveAugmentation3D:
    """Augmentation pipeline for 3D contrastive learning.
    
    Produces two correlated views of the same volume with different
    random transformations.
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_std: float = 0.05,
        intensity_shift: float = 0.1,
        crop_ratio: Tuple[float, float] = (0.7, 1.0),
    ):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.intensity_shift = intensity_shift
        self.crop_ratio = crop_ratio
        
    def __call__(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views.
        
        Args:
            volume: (C, D, H, W) single volume
            
        Returns:
            (view1, view2) each (C, D, H, W)
        """
        view1 = self._augment(volume.clone())
        view2 = self._augment(volume.clone())
        return view1, view2
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random flip along each axis
        for axis in [1, 2, 3]:  # D, H, W
            if torch.rand(1).item() < self.flip_prob:
                x = x.flip(axis)
                
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        # Random intensity shift
        if self.intensity_shift > 0:
            shift = (torch.rand(1).item() * 2 - 1) * self.intensity_shift
            x = x + shift
            
        # Random contrast
        factor = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
        mean = x.mean()
        x = (x - mean) * factor + mean
        
        return x


if __name__ == "__main__":
    print("Testing ContrastiveLearning3D...")
    
    encoder = Encoder3D(in_channels=1, feature_dim=256)
    model = ContrastiveLearning3D(
        encoder=encoder,
        feature_dim=256,
        projection_dim=128,
        temperature=0.1,
    )
    
    v1 = torch.randn(4, 1, 32, 32, 32)
    v2 = torch.randn(4, 1, 32, 32, 32)
    
    out = model(v1, v2)
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  z1 shape: {out['z1'].shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test augmentation
    aug = ContrastiveAugmentation3D()
    vol = torch.randn(1, 64, 64, 64)
    a, b = aug(vol)
    print(f"  Aug views: {a.shape}, {b.shape}")
    print("  ✓ Contrastive test passed!")
