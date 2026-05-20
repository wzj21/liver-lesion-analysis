"""
3D ConvNeXt Backbone Network
3D ConvNeXt骨干网络

Based on "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
Extended to 3D for volumetric medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


class LayerNorm3d(nn.Module):
    """LayerNorm for 3D inputs (channels-first format)."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
        return x


class DropPath3d(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
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


class GRN3d(nn.Module):
    """Global Response Normalization for 3D inputs (ConvNeXt V2)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock3d(nn.Module):
    """ConvNeXt Block for 3D inputs."""
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,
        kernel_size: int = 7,
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN3d(4 * dim) if use_grn else nn.Identity()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath3d(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        
        if isinstance(self.grn, GRN3d):
            x = x.permute(0, 4, 1, 2, 3)
            x = self.grn(x)
            x = x.permute(0, 2, 3, 4, 1)
            
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        
        return input + self.drop_path(x)


class ConvNeXt3d(nn.Module):
    """3D ConvNeXt Backbone."""
    
    CONFIGS = {
        'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]},
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,
        stem_kernel_size: int = 4,
        stem_stride: int = 4,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        variant: str = 'tiny',
    ):
        super().__init__()
        
        if depths is None or dims is None:
            config = self.CONFIGS.get(variant, self.CONFIGS['tiny'])
            depths = depths or config['depths']
            dims = dims or config['dims']
            
        self.depths = depths
        self.dims = dims
        self.out_indices = out_indices
        self.num_stages = len(depths)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=stem_kernel_size,
                     stride=stem_stride, padding=stem_kernel_size // 4),
            LayerNorm3d(dims[0]),
        )
        
        # Stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                ConvNeXtBlock3d(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            if i < self.num_stages - 1:
                downsample = nn.Sequential(
                    LayerNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())
                
        self.out_norms = nn.ModuleList([
            LayerNorm3d(dims[i]) if i in out_indices else nn.Identity()
            for i in range(self.num_stages)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        features = {}
        
        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i in self.out_indices:
                features[f'stage{i}'] = self.out_norms[i](x)
            x = self.downsamples[i](x)
            
        return features
    
    @property
    def feature_dims(self) -> List[int]:
        return [self.dims[i] for i in self.out_indices]


class ConvNeXt3dEncoder(nn.Module):
    """ConvNeXt 3D Encoder with FPN-style feature extraction."""
    
    def __init__(
        self,
        in_channels: int = 1,
        variant: str = 'small',
        pretrained_path: Optional[str] = None,
        out_channels: int = 256,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        freeze_stages: int = 0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = ConvNeXt3d(
            in_channels=in_channels,
            variant=variant,
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
        )
        
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            
        self._freeze_stages(freeze_stages)
        
        self.in_dims = self.backbone.feature_dims
        self.laterals = nn.ModuleList([
            nn.Conv3d(dim, out_channels, kernel_size=1)
            for dim in self.in_dims
        ])
        
    def _load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model_state = self.backbone.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                       if k in model_state and v.shape == model_state[k].shape}
            self.backbone.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)}/{len(model_state)} weights from {path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            
    def _freeze_stages(self, num_stages: int):
        if num_stages > 0:
            self.backbone.stem.eval()
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
            for i in range(min(num_stages, len(self.backbone.stages))):
                self.backbone.stages[i].eval()
                for p in self.backbone.stages[i].parameters():
                    p.requires_grad = False
                    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        return [self.laterals[i](features[f'stage{i}']) 
                for i in range(len(self.laterals))]


def build_convnext_3d(variant: str = 'small', **kwargs) -> ConvNeXt3d:
    """Build ConvNeXt 3D backbone."""
    return ConvNeXt3d(variant=variant, **kwargs)


if __name__ == "__main__":
    model = ConvNeXt3d(in_channels=1, variant='tiny')
    x = torch.randn(2, 1, 64, 64, 64)
    features = model(x)
    for k, v in features.items():
        print(f"{k}: {v.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
