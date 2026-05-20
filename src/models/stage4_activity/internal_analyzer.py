"""
Internal Structure Analyzer
内部结构分析器

Analyzes internal lesion characteristics for activity assessment:
- Sub-cyst detection (for CE)
- Density distribution and heterogeneity
- Calcification patterns
- Necrosis detection (for AE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class InternalStructureAnalyzer(nn.Module):
    """Analyzes internal structure of echinococcosis lesions.
    
    Key features:
    - For CE: Sub-cyst detection, membrane patterns
    - For AE: Necrosis, calcification distribution
    - Common: Density heterogeneity, texture features
    """
    
    FEATURE_NAMES = [
        'num_subcysts',
        'subcyst_ratio',
        'density_heterogeneity',
        'central_density',
        'peripheral_density',
        'calcification_ratio',
        'calcification_pattern',  # 0=none, 1=peripheral, 2=central, 3=mixed
        'necrosis_ratio',
        'texture_entropy',
        'texture_contrast',
    ]
    
    def __init__(
        self,
        in_channels: int = 2,  # CT + mask
        hidden_channels: List[int] = None,
        output_dim: int = 128,
        use_3d: bool = True,
    ):
        """Initialize internal structure analyzer.
        
        Args:
            in_channels: Number of input channels
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
        Pool = nn.MaxPool3d if use_3d else nn.MaxPool2d
        
        # Multi-scale feature extraction
        self.conv_blocks = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in hidden_channels:
            block = nn.Sequential(
                Conv(prev_ch, ch, kernel_size=3, padding=1),
                BatchNorm(ch),
                nn.ReLU(inplace=True),
                Conv(ch, ch, kernel_size=3, padding=1),
                BatchNorm(ch),
                nn.ReLU(inplace=True),
            )
            self.conv_blocks.append(block)
            prev_ch = ch
            
        # Pooling
        self.pool = Pool(kernel_size=2, stride=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1) if use_3d else nn.AdaptiveAvgPool2d(1)
        
        # Feature aggregation
        total_channels = sum(hidden_channels)
        self.fc = nn.Sequential(
            nn.Linear(total_channels, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        
        # Density histogram embedding
        self.histogram_embed = nn.Sequential(
            nn.Linear(64, 64),  # 64 histogram bins
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )
        
    def compute_density_histogram(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
        num_bins: int = 64,
        hu_range: Tuple[float, float] = (-100, 200),
    ) -> torch.Tensor:
        """Compute density histogram within lesion.
        
        Args:
            ct: CT volume
            mask: Lesion mask
            num_bins: Number of histogram bins
            hu_range: HU range for histogram
            
        Returns:
            Normalized histogram (B, num_bins)
        """
        B = ct.size(0)
        device = ct.device
        
        histograms = []
        for b in range(B):
            lesion_voxels = ct[b, 0][mask[b, 0] > 0.5]
            
            if len(lesion_voxels) > 0:
                # Clamp to HU range
                lesion_voxels = lesion_voxels.clamp(hu_range[0], hu_range[1])
                
                # Compute histogram
                hist = torch.histc(lesion_voxels, bins=num_bins, min=hu_range[0], max=hu_range[1])
                hist = hist / (hist.sum() + 1e-8)  # Normalize
            else:
                hist = torch.zeros(num_bins, device=device)
                
            histograms.append(hist)
            
        return torch.stack(histograms, dim=0)
        
    def forward(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            ct: CT volume (B, 1, D, H, W)
            mask: Lesion mask (B, 1, D, H, W)
            
        Returns:
            Tuple of (features, feature_dict)
        """
        # Mask the CT to focus on lesion interior
        masked_ct = ct * mask
        
        # Concatenate CT and mask
        x = torch.cat([masked_ct, mask], dim=1)
        
        # Multi-scale features
        scale_features = []
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            # Global pool at this scale
            pooled = self.global_pool(x)
            scale_features.append(pooled.view(pooled.size(0), -1))
            
            if i < len(self.conv_blocks) - 1:
                x = self.pool(x)
                
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=1)
        
        # Density histogram features
        histogram = self.compute_density_histogram(ct, mask)
        hist_features = self.histogram_embed(histogram)
        
        # Combine and output
        combined = torch.cat([multi_scale, hist_features], dim=1)
        
        # Adjust FC input size if needed
        if not hasattr(self, '_fc_adjusted'):
            actual_dim = combined.size(1)
            expected_dim = self.fc[0].in_features
            if actual_dim != expected_dim:
                self.fc = nn.Sequential(
                    nn.Linear(actual_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, self.fc[-1].out_features),
                ).to(combined.device)
            self._fc_adjusted = True
            
        features = self.fc(combined)
        
        # Compute interpretable features
        feature_dict = self._compute_interpretable_features(ct, mask)
        
        return features, feature_dict
        
    def _compute_interpretable_features(
        self,
        ct: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute interpretable internal features.
        
        Args:
            ct: CT volume
            mask: Lesion mask
            
        Returns:
            Dictionary of interpretable features
        """
        B = ct.size(0)
        device = ct.device
        
        features = {}
        
        for b in range(B):
            lesion_mask = mask[b, 0] > 0.5
            
            if lesion_mask.sum() > 0:
                lesion_values = ct[b, 0][lesion_mask]
                
                # Density statistics
                features[f'density_mean_{b}'] = lesion_values.mean()
                features[f'density_std_{b}'] = lesion_values.std()
                features[f'density_heterogeneity_{b}'] = lesion_values.std() / (lesion_values.mean().abs() + 1e-8)
                
                # Calcification ratio (HU > 150)
                calcified = (ct[b, 0] > 150) & lesion_mask
                features[f'calcification_ratio_{b}'] = calcified.sum().float() / lesion_mask.sum().float()
                
                # Necrosis ratio (HU in range -20 to 20, typical for necrotic tissue)
                necrotic = ((ct[b, 0] > -20) & (ct[b, 0] < 20)) & lesion_mask
                features[f'necrosis_ratio_{b}'] = necrotic.sum().float() / lesion_mask.sum().float()
                
                # Fluid content (HU in range 0 to 20, typical for cyst fluid)
                fluid = ((ct[b, 0] > 0) & (ct[b, 0] < 20)) & lesion_mask
                features[f'fluid_ratio_{b}'] = fluid.sum().float() / lesion_mask.sum().float()
                
            else:
                features[f'density_mean_{b}'] = torch.tensor(0.0, device=device)
                features[f'density_std_{b}'] = torch.tensor(0.0, device=device)
                features[f'density_heterogeneity_{b}'] = torch.tensor(0.0, device=device)
                features[f'calcification_ratio_{b}'] = torch.tensor(0.0, device=device)
                features[f'necrosis_ratio_{b}'] = torch.tensor(0.0, device=device)
                features[f'fluid_ratio_{b}'] = torch.tensor(0.0, device=device)
                
        # Aggregate
        aggregated = {
            'density_mean': torch.stack([features[f'density_mean_{b}'] for b in range(B)]),
            'density_std': torch.stack([features[f'density_std_{b}'] for b in range(B)]),
            'density_heterogeneity': torch.stack([features[f'density_heterogeneity_{b}'] for b in range(B)]),
            'calcification_ratio': torch.stack([features[f'calcification_ratio_{b}'] for b in range(B)]),
            'necrosis_ratio': torch.stack([features[f'necrosis_ratio_{b}'] for b in range(B)]),
            'fluid_ratio': torch.stack([features[f'fluid_ratio_{b}'] for b in range(B)]),
        }
        
        return aggregated


def detect_subcysts(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    threshold_low: float = 0,
    threshold_high: float = 30,
    min_size: int = 100,
) -> Tuple[np.ndarray, int]:
    """Detect sub-cysts within a cystic echinococcosis lesion.
    
    Sub-cysts appear as separate fluid-filled regions within the main cyst.
    
    Args:
        ct_volume: CT volume in HU
        mask: Lesion mask
        threshold_low: Lower HU threshold for fluid
        threshold_high: Upper HU threshold for fluid
        min_size: Minimum sub-cyst size in voxels
        
    Returns:
        Tuple of (labeled_subcysts, num_subcysts)
    """
    from scipy import ndimage
    
    # Find fluid regions within lesion
    fluid_mask = (
        (ct_volume >= threshold_low) & 
        (ct_volume <= threshold_high) & 
        mask.astype(bool)
    )
    
    # Label connected components
    labeled, num_features = ndimage.label(fluid_mask)
    
    # Filter by size
    if num_features > 0:
        component_sizes = ndimage.sum(fluid_mask, labeled, range(1, num_features + 1))
        
        # Keep only components above minimum size
        valid_components = []
        for i, size in enumerate(component_sizes, 1):
            if size >= min_size:
                valid_components.append(i)
                
        # Relabel with only valid components
        new_labeled = np.zeros_like(labeled)
        for new_label, old_label in enumerate(valid_components, 1):
            new_labeled[labeled == old_label] = new_label
            
        return new_labeled, len(valid_components)
        
    return labeled, 0


def compute_internal_features(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    lesion_type: str = 'CE',  # 'CE' or 'AE'
) -> Dict[str, float]:
    """Compute detailed internal structure features.
    
    Args:
        ct_volume: CT volume in HU
        mask: Binary lesion mask
        lesion_type: Type of echinococcosis ('CE' or 'AE')
        
    Returns:
        Dictionary of internal features
    """
    from scipy import ndimage
    
    features = {}
    mask = mask.astype(bool)
    
    if mask.sum() == 0:
        return {name: 0.0 for name in InternalStructureAnalyzer.FEATURE_NAMES}
        
    lesion_values = ct_volume[mask]
    
    # Density statistics
    features['density_heterogeneity'] = float(lesion_values.std() / (np.abs(lesion_values.mean()) + 1e-8))
    
    # Central vs peripheral density
    dist_transform = ndimage.distance_transform_edt(mask)
    max_dist = dist_transform.max()
    
    if max_dist > 0:
        central_mask = mask & (dist_transform > max_dist * 0.5)
        peripheral_mask = mask & (dist_transform <= max_dist * 0.5)
        
        if central_mask.sum() > 0:
            features['central_density'] = float(ct_volume[central_mask].mean())
        else:
            features['central_density'] = 0.0
            
        if peripheral_mask.sum() > 0:
            features['peripheral_density'] = float(ct_volume[peripheral_mask].mean())
        else:
            features['peripheral_density'] = 0.0
    else:
        features['central_density'] = float(lesion_values.mean())
        features['peripheral_density'] = float(lesion_values.mean())
        
    # Calcification analysis
    calcified = (ct_volume > 150) & mask
    features['calcification_ratio'] = float(calcified.sum() / mask.sum())
    
    # Calcification pattern
    if features['calcification_ratio'] > 0.01:
        calc_central = calcified & (dist_transform > max_dist * 0.5) if max_dist > 0 else calcified
        calc_peripheral = calcified & (dist_transform <= max_dist * 0.5) if max_dist > 0 else np.zeros_like(calcified)
        
        central_ratio = calc_central.sum() / (calcified.sum() + 1e-8)
        if central_ratio > 0.7:
            features['calcification_pattern'] = 2  # Central
        elif central_ratio < 0.3:
            features['calcification_pattern'] = 1  # Peripheral
        else:
            features['calcification_pattern'] = 3  # Mixed
    else:
        features['calcification_pattern'] = 0  # None
        
    # Necrosis (for AE)
    necrotic = ((ct_volume > -20) & (ct_volume < 20)) & mask
    features['necrosis_ratio'] = float(necrotic.sum() / mask.sum())
    
    # Sub-cyst detection (for CE)
    if lesion_type == 'CE':
        _, num_subcysts = detect_subcysts(ct_volume, mask)
        features['num_subcysts'] = float(num_subcysts)
        
        # Sub-cyst ratio (fluid content)
        fluid = ((ct_volume > 0) & (ct_volume < 30)) & mask
        features['subcyst_ratio'] = float(fluid.sum() / mask.sum())
    else:
        features['num_subcysts'] = 0.0
        features['subcyst_ratio'] = 0.0
        
    # Texture features (entropy and contrast)
    # Compute histogram
    hist, _ = np.histogram(lesion_values, bins=64, density=True)
    hist = hist[hist > 0]
    features['texture_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
    
    # Contrast (standard deviation normalized by range)
    value_range = lesion_values.max() - lesion_values.min()
    if value_range > 0:
        features['texture_contrast'] = float(lesion_values.std() / value_range)
    else:
        features['texture_contrast'] = 0.0
        
    return features


if __name__ == "__main__":
    # Test internal structure analyzer
    print("Testing Internal Structure Analyzer:")
    
    analyzer = InternalStructureAnalyzer(
        in_channels=2,
        hidden_channels=[32, 64, 128],
        output_dim=128,
        use_3d=True,
    )
    
    # Simulate input
    batch_size = 2
    ct = torch.randn(batch_size, 1, 32, 64, 64) * 50 + 30  # Simulated CT
    mask = (torch.rand(batch_size, 1, 32, 64, 64) > 0.7).float()
    
    # Forward pass
    features, feature_dict = analyzer(ct, mask)
    
    print(f"Input CT shape: {ct.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Output features shape: {features.shape}")
    print("\nInterpretable features:")
    for key, value in feature_dict.items():
        print(f"  {key}: {value}")
        
    # Test numpy feature extraction
    print("\nTesting numpy feature extraction:")
    ct_np = ct[0, 0].numpy()
    mask_np = mask[0, 0].numpy() > 0.5
    
    internal_features = compute_internal_features(ct_np, mask_np, lesion_type='CE')
    print("Internal features (CE):")
    for key, value in internal_features.items():
        print(f"  {key}: {value:.4f}")
        
    # Model parameters
    num_params = sum(p.numel() for p in analyzer.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.4f}M")
