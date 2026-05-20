"""
Boundary Feature Extractor
边界特征提取器

Analyzes lesion boundary characteristics for activity assessment:
- Wall thickness and continuity
- Boundary sharpness and irregularity
- Infiltration patterns (for AE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class BoundaryFeatureExtractor(nn.Module):
    """Extracts boundary-related features from lesion masks and CT.
    
    Key features for activity assessment:
    - Cyst wall analysis (thickness, density, continuity)
    - Boundary sharpness and regularity
    - Infiltration score (important for AE)
    """
    
    # Feature names for interpretability
    FEATURE_NAMES = [
        'wall_thickness_mean',
        'wall_thickness_std',
        'wall_thickness_max',
        'wall_density_mean',
        'wall_density_std',
        'wall_continuity_score',
        'wall_irregularity_score',
        'boundary_sharpness',
        'infiltration_score',
        'calcification_boundary_ratio',
    ]
    
    def __init__(
        self,
        in_channels: int = 2,  # CT + mask
        hidden_channels: List[int] = None,
        output_dim: int = 128,
        use_3d: bool = True,
    ):
        """Initialize boundary feature extractor.
        
        Args:
            in_channels: Number of input channels (CT + mask)
            hidden_channels: Hidden channel dimensions
            output_dim: Output feature dimension
            use_3d: Whether to use 3D convolutions
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
            
        self.use_3d = use_3d
        Conv = nn.Conv3d if use_3d else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if use_3d else nn.BatchNorm2d
        
        # Boundary-focused CNN
        layers = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers.extend([
                Conv(prev_ch, ch, kernel_size=3, padding=1),
                BatchNorm(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch
            
        self.cnn = nn.Sequential(*layers)
        
        # Global pooling + FC
        self.global_pool = nn.AdaptiveAvgPool3d(1) if use_3d else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        
        # Boundary detection kernels (Sobel-like)
        self._init_boundary_kernels()
        
    def _init_boundary_kernels(self):
        """Initialize boundary detection kernels."""
        if self.use_3d:
            # 3D Sobel kernels for boundary detection
            sobel_x = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            sobel_y = sobel_x.permute(0, 1, 3, 2, 4)
            sobel_z = sobel_x.permute(0, 1, 4, 3, 2)
            
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)
            self.register_buffer('sobel_z', sobel_z)
        else:
            # 2D Sobel kernels
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            sobel_y = sobel_x.permute(0, 1, 3, 2)
            
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)
            
    def compute_boundary_gradient(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary gradient magnitude.
        
        Args:
            mask: Binary mask (B, 1, ...)
            
        Returns:
            Gradient magnitude at boundary
        """
        mask_float = mask.float()
        
        if self.use_3d:
            gx = F.conv3d(mask_float, self.sobel_x, padding=1)
            gy = F.conv3d(mask_float, self.sobel_y, padding=1)
            gz = F.conv3d(mask_float, self.sobel_z, padding=1)
            gradient = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        else:
            gx = F.conv2d(mask_float, self.sobel_x, padding=1)
            gy = F.conv2d(mask_float, self.sobel_y, padding=1)
            gradient = torch.sqrt(gx**2 + gy**2 + 1e-8)
            
        return gradient
        
    def extract_boundary_region(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
        width: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract boundary region from CT and mask.
        
        Args:
            ct: CT volume (B, 1, ...)
            mask: Binary mask (B, 1, ...)
            width: Boundary width in voxels
            
        Returns:
            Tuple of (boundary_ct, boundary_mask)
        """
        # Dilate mask
        if self.use_3d:
            kernel = torch.ones(1, 1, width, width, width, device=mask.device)
            dilated = F.conv3d(mask.float(), kernel, padding=width//2)
        else:
            kernel = torch.ones(1, 1, width, width, device=mask.device)
            dilated = F.conv2d(mask.float(), kernel, padding=width//2)
            
        dilated = (dilated > 0).float()
        
        # Erode mask
        if self.use_3d:
            eroded = F.conv3d(mask.float(), kernel, padding=width//2)
            eroded = (eroded >= width**3).float()
        else:
            eroded = F.conv2d(mask.float(), kernel, padding=width//2)
            eroded = (eroded >= width**2).float()
            
        # Boundary = dilated - eroded
        boundary_mask = dilated - eroded
        boundary_mask = boundary_mask.clamp(0, 1)
        
        # Extract boundary CT values
        boundary_ct = ct * boundary_mask
        
        return boundary_ct, boundary_mask
        
    def forward(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            ct: CT volume (B, 1, D, H, W) or (B, 1, H, W)
            mask: Lesion mask (B, 1, D, H, W) or (B, 1, H, W)
            
        Returns:
            Tuple of (features, feature_dict)
        """
        # Extract boundary region
        boundary_ct, boundary_mask = self.extract_boundary_region(ct, mask)
        
        # Compute gradient at boundary
        gradient = self.compute_boundary_gradient(mask)
        
        # Concatenate inputs
        x = torch.cat([boundary_ct, boundary_mask], dim=1)
        
        # CNN features
        cnn_features = self.cnn(x)
        
        # Global pooling
        pooled = self.global_pool(cnn_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # FC layers
        features = self.fc(pooled)
        
        # Compute interpretable features
        feature_dict = self._compute_interpretable_features(ct, mask, boundary_ct, boundary_mask, gradient)
        
        return features, feature_dict
        
    def _compute_interpretable_features(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
        boundary_ct: torch.Tensor,
        boundary_mask: torch.Tensor,
        gradient: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute interpretable boundary features.
        
        Args:
            ct: Full CT volume
            mask: Full mask
            boundary_ct: CT at boundary
            boundary_mask: Boundary mask
            gradient: Gradient magnitude
            
        Returns:
            Dictionary of interpretable features
        """
        B = ct.size(0)
        device = ct.device
        
        features = {}
        
        for b in range(B):
            boundary_voxels = boundary_mask[b, 0] > 0.5
            
            if boundary_voxels.sum() > 0:
                # Wall density statistics
                wall_values = boundary_ct[b, 0][boundary_voxels]
                features[f'wall_density_mean_{b}'] = wall_values.mean()
                features[f'wall_density_std_{b}'] = wall_values.std()
                
                # Boundary sharpness (gradient magnitude at boundary)
                grad_at_boundary = gradient[b, 0][boundary_voxels]
                features[f'boundary_sharpness_{b}'] = grad_at_boundary.mean()
                
                # Wall thickness approximation (boundary mask volume / surface area)
                boundary_volume = boundary_voxels.sum().float()
                grad_sum = gradient[b, 0].sum()
                if grad_sum > 0:
                    features[f'wall_thickness_approx_{b}'] = boundary_volume / grad_sum
                else:
                    features[f'wall_thickness_approx_{b}'] = torch.tensor(0.0, device=device)
                    
                # Calcification at boundary (HU > 150)
                calcified = (boundary_ct[b, 0] > 150) & boundary_voxels
                features[f'calcification_ratio_{b}'] = calcified.sum().float() / boundary_voxels.sum().float()
            else:
                features[f'wall_density_mean_{b}'] = torch.tensor(0.0, device=device)
                features[f'wall_density_std_{b}'] = torch.tensor(0.0, device=device)
                features[f'boundary_sharpness_{b}'] = torch.tensor(0.0, device=device)
                features[f'wall_thickness_approx_{b}'] = torch.tensor(0.0, device=device)
                features[f'calcification_ratio_{b}'] = torch.tensor(0.0, device=device)
                
        # Aggregate across batch
        aggregated = {
            'wall_density_mean': torch.stack([features[f'wall_density_mean_{b}'] for b in range(B)]),
            'wall_density_std': torch.stack([features[f'wall_density_std_{b}'] for b in range(B)]),
            'boundary_sharpness': torch.stack([features[f'boundary_sharpness_{b}'] for b in range(B)]),
            'wall_thickness_approx': torch.stack([features[f'wall_thickness_approx_{b}'] for b in range(B)]),
            'calcification_ratio': torch.stack([features[f'calcification_ratio_{b}'] for b in range(B)]),
        }
        
        return aggregated


def compute_wall_features(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute detailed wall features from numpy arrays.
    
    Args:
        ct_volume: CT volume in HU
        mask: Binary lesion mask
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary of wall features
    """
    from scipy import ndimage
    
    features = {}
    
    # Create boundary mask
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(mask, struct, iterations=3)
    eroded = ndimage.binary_erosion(mask, struct, iterations=3)
    boundary = dilated & ~eroded
    
    if boundary.sum() == 0:
        return {name: 0.0 for name in BoundaryFeatureExtractor.FEATURE_NAMES}
        
    # Wall CT values
    wall_values = ct_volume[boundary]
    
    features['wall_density_mean'] = float(wall_values.mean())
    features['wall_density_std'] = float(wall_values.std())
    
    # Wall thickness estimation using distance transform
    dist_inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    dist_outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    
    # Thickness at boundary points
    boundary_dist = dist_inside[boundary]
    features['wall_thickness_mean'] = float(boundary_dist.mean())
    features['wall_thickness_std'] = float(boundary_dist.std())
    features['wall_thickness_max'] = float(boundary_dist.max())
    
    # Wall continuity (how continuous is the wall)
    # Higher value = more continuous
    labeled, num_components = ndimage.label(boundary)
    if num_components > 0:
        component_sizes = ndimage.sum(boundary, labeled, range(1, num_components + 1))
        features['wall_continuity_score'] = float(component_sizes.max() / boundary.sum())
    else:
        features['wall_continuity_score'] = 0.0
        
    # Wall irregularity (surface roughness)
    # Compute using gradient
    gx = ndimage.sobel(mask.astype(float), axis=0)
    gy = ndimage.sobel(mask.astype(float), axis=1)
    gz = ndimage.sobel(mask.astype(float), axis=2)
    gradient_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    surface_gradient = gradient_mag[boundary]
    features['wall_irregularity_score'] = float(surface_gradient.std())
    
    # Boundary sharpness (gradient at boundary)
    ct_gx = ndimage.sobel(ct_volume, axis=0)
    ct_gy = ndimage.sobel(ct_volume, axis=1)
    ct_gz = ndimage.sobel(ct_volume, axis=2)
    ct_gradient = np.sqrt(ct_gx**2 + ct_gy**2 + ct_gz**2)
    
    features['boundary_sharpness'] = float(ct_gradient[boundary].mean())
    
    # Infiltration score (for AE - fuzzy boundaries indicate infiltration)
    # Lower sharpness + higher irregularity = higher infiltration
    if features['boundary_sharpness'] > 0:
        features['infiltration_score'] = features['wall_irregularity_score'] / features['boundary_sharpness']
    else:
        features['infiltration_score'] = 0.0
        
    # Calcification at boundary
    calcified = (ct_volume > 150) & boundary
    features['calcification_boundary_ratio'] = float(calcified.sum() / boundary.sum())
    
    return features


if __name__ == "__main__":
    # Test boundary feature extractor
    print("Testing Boundary Feature Extractor:")
    
    extractor = BoundaryFeatureExtractor(
        in_channels=2,
        hidden_channels=[32, 64, 128],
        output_dim=128,
        use_3d=True,
    )
    
    # Simulate input
    batch_size = 2
    ct = torch.randn(batch_size, 1, 32, 64, 64) * 100  # Simulated CT in HU range
    mask = (torch.rand(batch_size, 1, 32, 64, 64) > 0.7).float()
    
    # Forward pass
    features, feature_dict = extractor(ct, mask)
    
    print(f"Input CT shape: {ct.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Output features shape: {features.shape}")
    print("\nInterpretable features:")
    for key, value in feature_dict.items():
        print(f"  {key}: {value}")
        
    # Model parameters
    num_params = sum(p.numel() for p in extractor.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.4f}M")
