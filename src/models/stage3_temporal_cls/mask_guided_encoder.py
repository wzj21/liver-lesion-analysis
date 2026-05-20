"""
Mask-Guided Feature Encoder
掩膜引导的特征编码器

Extracts features from CT slices with mask guidance for lesion-focused encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class ConvNeXtBlock2d(nn.Module):
    """2D ConvNeXt Block for slice-level feature extraction."""
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return input + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic Depth."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXt2d(nn.Module):
    """2D ConvNeXt Backbone for slice-level encoding."""
    
    CONFIGS = {
        'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        variant: str = 'tiny',
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        config = self.CONFIGS.get(variant, self.CONFIGS['tiny'])
        depths = config['depths']
        dims = config['dims']
        
        self.dims = dims
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )
        
        # Stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock2d(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            if i < 3:
                downsample = nn.Sequential(
                    nn.LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())
                
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, D)
        """
        x = self.stem(x)
        
        # Handle LayerNorm for channel-first input
        if isinstance(self.stem[1], nn.LayerNorm):
            x = x.permute(0, 2, 3, 1)
            x = self.stem[1](x)
            x = x.permute(0, 3, 1, 2)
        
        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            if isinstance(downsample, nn.Sequential):
                x = x.permute(0, 2, 3, 1)
                x = downsample[0](x)  # LayerNorm
                x = x.permute(0, 3, 1, 2)
                x = downsample[1](x)  # Conv2d
            else:
                x = downsample(x)
                
        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        
        return x
    
    @property
    def feature_dim(self) -> int:
        return self.dims[-1]


class MaskGuidedEncoder(nn.Module):
    """Mask-Guided Feature Encoder.
    
    Extracts features from CT slices with lesion mask guidance.
    The mask helps the encoder focus on relevant lesion regions.
    
    Input channels:
    - CT slice (1 channel)
    - Lesion mask (1 channel)
    - Mask boundary (1 channel, optional)
    """
    
    def __init__(
        self,
        backbone_variant: str = 'tiny',
        pretrained_path: Optional[str] = None,
        include_boundary: bool = True,
        mask_guidance_method: str = 'channel_concat',
        drop_path_rate: float = 0.1,
    ):
        """Initialize mask-guided encoder.
        
        Args:
            backbone_variant: ConvNeXt variant ('tiny', 'small', 'base')
            pretrained_path: Path to pretrained weights
            include_boundary: Whether to include mask boundary channel
            mask_guidance_method: How to incorporate mask ('channel_concat', 'attention', 'multiply')
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.include_boundary = include_boundary
        self.mask_guidance_method = mask_guidance_method
        
        # Determine input channels based on method
        if mask_guidance_method == 'channel_concat':
            in_channels = 3 if include_boundary else 2  # CT + mask + (boundary)
        else:
            in_channels = 1  # CT only, mask applied differently
            
        # Backbone
        self.backbone = ConvNeXt2d(
            in_channels=in_channels,
            variant=backbone_variant,
            drop_path_rate=drop_path_rate,
        )
        
        # For attention-based mask guidance
        if mask_guidance_method == 'attention':
            self.mask_attention = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            
        # Load pretrained weights
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            
    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
                
            # Handle input channel mismatch
            if 'stem.0.weight' in state_dict:
                pretrained_in_ch = state_dict['stem.0.weight'].shape[1]
                current_in_ch = self.backbone.stem[0].in_channels
                
                if pretrained_in_ch != current_in_ch:
                    # Repeat or truncate pretrained weights
                    weight = state_dict['stem.0.weight']
                    if current_in_ch > pretrained_in_ch:
                        # Repeat weights for additional channels
                        repeat_times = math.ceil(current_in_ch / pretrained_in_ch)
                        weight = weight.repeat(1, repeat_times, 1, 1)[:, :current_in_ch]
                    else:
                        weight = weight[:, :current_in_ch]
                    state_dict['stem.0.weight'] = weight
                    
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            
    def forward(
        self,
        ct_slice: torch.Tensor,
        mask: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            ct_slice: CT slice (B, 1, H, W)
            mask: Lesion mask (B, 1, H, W)
            boundary: Mask boundary (B, 1, H, W), optional
            
        Returns:
            Feature tensor (B, D)
        """
        if self.mask_guidance_method == 'channel_concat':
            # Concatenate CT, mask, and optionally boundary
            if self.include_boundary and boundary is not None:
                x = torch.cat([ct_slice, mask, boundary], dim=1)
            else:
                x = torch.cat([ct_slice, mask], dim=1)
                
        elif self.mask_guidance_method == 'attention':
            # Apply mask as spatial attention
            attention = self.mask_attention(mask)
            x = ct_slice * attention
            
        elif self.mask_guidance_method == 'multiply':
            # Direct multiplication
            x = ct_slice * mask
            
        else:
            raise ValueError(f"Unknown mask guidance method: {self.mask_guidance_method}")
            
        # Extract features
        features = self.backbone(x)
        
        return features
    
    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim


def extract_mask_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract boundary from binary mask.
    
    Args:
        mask: Binary mask (B, 1, H, W)
        kernel_size: Morphological kernel size
        
    Returns:
        Boundary mask (B, 1, H, W)
    """
    # Dilation
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    dilated = F.conv2d(mask.float(), kernel, padding=padding)
    dilated = (dilated > 0).float()
    
    # Erosion
    eroded = F.conv2d(mask.float(), kernel, padding=padding)
    eroded = (eroded >= kernel_size * kernel_size).float()
    
    # Boundary = dilation - erosion
    boundary = dilated - eroded
    
    return boundary


if __name__ == "__main__":
    # Test mask-guided encoder
    encoder = MaskGuidedEncoder(
        backbone_variant='tiny',
        include_boundary=True,
        mask_guidance_method='channel_concat',
    )
    
    # Simulate input
    batch_size = 4
    ct_slice = torch.randn(batch_size, 1, 224, 224)
    mask = (torch.rand(batch_size, 1, 224, 224) > 0.5).float()
    boundary = extract_mask_boundary(mask)
    
    # Forward pass
    features = encoder(ct_slice, mask, boundary)
    
    print(f"Input shape: {ct_slice.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Feature dimension: {encoder.feature_dim}")
    
    # Model parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
