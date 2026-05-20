"""
Morphology Feature Encoder
形态特征编码器

Encodes geometric and intensity features of lesions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class MorphologyEncoder(nn.Module):
    """Encoder for lesion morphological features.
    
    Encodes geometric features (volume, shape) and intensity features (HU statistics).
    """
    
    # List of morphological features
    FEATURE_NAMES = [
        # Geometric features
        'volume',           # 体积
        'surface_area',     # 表面积
        'sphericity',       # 球形度
        'compactness',      # 紧凑度
        'eccentricity',     # 离心率
        'solidity',         # 实心度
        'extent',           # 范围
        'aspect_ratio',     # 长宽比
        # Intensity features
        'hu_mean',          # 平均HU值
        'hu_std',           # HU标准差
        'hu_min',           # 最小HU值
        'hu_max',           # 最大HU值
        'hu_median',        # 中位数HU值
        'hu_skewness',      # HU偏度
        'hu_kurtosis',      # HU峰度
        'hu_entropy',       # HU熵
    ]
    
    def __init__(
        self,
        num_input_features: int = 16,
        hidden_dims: List[int] = None,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """Initialize morphology encoder.
        
        Args:
            num_input_features: Number of input morphological features
            hidden_dims: Hidden layer dimensions
            output_dim: Output feature dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        self.num_input_features = num_input_features
        self.output_dim = output_dim
        
        # Build MLP
        layers = []
        prev_dim = num_input_features
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Feature normalization (learned)
        self.feature_norm = nn.BatchNorm1d(num_input_features)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Morphological features (B, num_features)
            
        Returns:
            Encoded features (B, output_dim)
        """
        # Normalize features
        x = self.feature_norm(features)
        
        # Encode
        x = self.mlp(x)
        
        return x


def compute_morphological_features(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute morphological features from CT volume and mask.
    
    Args:
        ct_volume: CT volume in HU
        mask: Binary lesion mask
        spacing: Voxel spacing in mm (z, y, x)
        
    Returns:
        Dictionary of morphological features
    """
    features = {}
    
    # Ensure mask is binary
    mask = mask.astype(bool)
    
    # Extract lesion voxels
    lesion_voxels = ct_volume[mask]
    
    if len(lesion_voxels) == 0:
        # Return zeros if no lesion
        return {name: 0.0 for name in MorphologyEncoder.FEATURE_NAMES}
        
    # Voxel volume
    voxel_vol = spacing[0] * spacing[1] * spacing[2]  # mm³
    
    # --- Geometric features ---
    
    # Volume
    features['volume'] = len(lesion_voxels) * voxel_vol
    
    # Surface area (approximate using marching cubes or boundary voxels)
    try:
        from scipy import ndimage
        # Count boundary voxels (6-connectivity)
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        features['surface_area'] = boundary.sum() * (spacing[0] * spacing[1])  # Approximate
    except:
        features['surface_area'] = 0.0
        
    # Sphericity = (pi^(1/3) * (6*V)^(2/3)) / A
    if features['surface_area'] > 0:
        features['sphericity'] = (
            np.pi ** (1/3) * (6 * features['volume']) ** (2/3)
        ) / features['surface_area']
    else:
        features['sphericity'] = 0.0
        
    # Compactness = V / (bbox_volume)
    coords = np.argwhere(mask)
    if len(coords) > 0:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0) + 1
        bbox_size = (bbox_max - bbox_min) * np.array(spacing)
        bbox_volume = bbox_size.prod()
        features['compactness'] = features['volume'] / bbox_volume if bbox_volume > 0 else 0.0
        
        # Extent = V / convex_hull_volume (simplified)
        features['extent'] = features['compactness']  # Approximation
        
        # Aspect ratio
        sorted_dims = np.sort(bbox_size)[::-1]
        features['aspect_ratio'] = sorted_dims[0] / sorted_dims[-1] if sorted_dims[-1] > 0 else 1.0
    else:
        features['compactness'] = 0.0
        features['extent'] = 0.0
        features['aspect_ratio'] = 1.0
        
    # Eccentricity (based on principal axes)
    try:
        # Compute covariance matrix of coordinates
        coords_centered = coords - coords.mean(axis=0)
        cov = np.cov(coords_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        if eigenvalues[0] > 0:
            features['eccentricity'] = np.sqrt(1 - eigenvalues[-1] / eigenvalues[0])
        else:
            features['eccentricity'] = 0.0
    except:
        features['eccentricity'] = 0.0
        
    # Solidity = V / convex_hull_V (simplified to extent)
    features['solidity'] = features['extent']
    
    # --- Intensity features ---
    
    features['hu_mean'] = float(lesion_voxels.mean())
    features['hu_std'] = float(lesion_voxels.std())
    features['hu_min'] = float(lesion_voxels.min())
    features['hu_max'] = float(lesion_voxels.max())
    features['hu_median'] = float(np.median(lesion_voxels))
    
    # Skewness
    if features['hu_std'] > 0:
        features['hu_skewness'] = float(
            ((lesion_voxels - features['hu_mean']) ** 3).mean() / (features['hu_std'] ** 3)
        )
    else:
        features['hu_skewness'] = 0.0
        
    # Kurtosis
    if features['hu_std'] > 0:
        features['hu_kurtosis'] = float(
            ((lesion_voxels - features['hu_mean']) ** 4).mean() / (features['hu_std'] ** 4) - 3
        )
    else:
        features['hu_kurtosis'] = 0.0
        
    # Entropy
    try:
        hist, _ = np.histogram(lesion_voxels, bins=64, density=True)
        hist = hist[hist > 0]  # Remove zeros
        features['hu_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
    except:
        features['hu_entropy'] = 0.0
        
    return features


def features_to_tensor(
    features: Dict[str, float],
    feature_names: List[str] = None,
) -> torch.Tensor:
    """Convert feature dictionary to tensor.
    
    Args:
        features: Feature dictionary
        feature_names: List of feature names to include (in order)
        
    Returns:
        Feature tensor
    """
    if feature_names is None:
        feature_names = MorphologyEncoder.FEATURE_NAMES
        
    values = [features.get(name, 0.0) for name in feature_names]
    return torch.tensor(values, dtype=torch.float32)


class MorphologyFeatureExtractor:
    """Extracts morphological features from CT and mask."""
    
    def __init__(self, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.spacing = spacing
        
    def extract(
        self,
        ct_volume: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        """Extract features and return as tensor.
        
        Args:
            ct_volume: CT volume in HU
            mask: Binary lesion mask
            
        Returns:
            Feature tensor (num_features,)
        """
        features = compute_morphological_features(ct_volume, mask, self.spacing)
        return features_to_tensor(features)
        
    def extract_batch(
        self,
        ct_volumes: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> torch.Tensor:
        """Extract features for a batch.
        
        Args:
            ct_volumes: List of CT volumes
            masks: List of masks
            
        Returns:
            Feature tensor (B, num_features)
        """
        feature_list = [
            self.extract(ct, mask)
            for ct, mask in zip(ct_volumes, masks)
        ]
        return torch.stack(feature_list, dim=0)


if __name__ == "__main__":
    # Test morphology encoder
    encoder = MorphologyEncoder(
        num_input_features=16,
        hidden_dims=[64, 128, 256],
        output_dim=256,
    )
    
    # Simulate input features
    batch_size = 8
    features = torch.randn(batch_size, 16)
    
    # Forward pass
    encoded = encoder(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {encoded.shape}")
    
    # Test feature extraction
    print("\nTesting feature extraction:")
    ct_vol = np.random.randn(64, 64, 64).astype(np.float32) * 100
    mask = np.zeros((64, 64, 64), dtype=bool)
    mask[20:40, 20:40, 20:40] = True  # Simulated lesion
    
    extracted_features = compute_morphological_features(ct_vol, mask)
    print("Extracted features:")
    for name, value in extracted_features.items():
        print(f"  {name}: {value:.4f}")
        
    # Model parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.2f}M")
