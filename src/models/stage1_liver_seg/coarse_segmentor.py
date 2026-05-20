"""
Coarse Liver Segmentor
粗分割网络

Low-resolution segmentation for initial liver localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from ...backbones.convnext_3d import ConvNeXt3d, LayerNorm3d


class UNetDecoder3d(nn.Module):
    """3D U-Net style decoder with skip connections."""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 2,
    ):
        """Initialize decoder.
        
        Args:
            encoder_channels: Channel dimensions from encoder stages
            decoder_channels: Channel dimensions for decoder stages
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Build decoder blocks
        self.upsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        in_channels = encoder_channels[-1]
        
        for i, (enc_ch, dec_ch) in enumerate(zip(
            reversed(encoder_channels[:-1]),
            decoder_channels
        )):
            # Upsample
            self.upsample_blocks.append(
                nn.ConvTranspose3d(in_channels, dec_ch, kernel_size=2, stride=2)
            )
            
            # Conv block after concatenation
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv3d(dec_ch + enc_ch, dec_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(dec_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(dec_ch, dec_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(dec_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = dec_ch
            
        # Final convolution
        self.final_conv = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)
        
    def forward(
        self,
        encoder_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            encoder_features: List of features from encoder stages
            
        Returns:
            Segmentation logits
        """
        x = encoder_features[-1]
        
        for i, (upsample, conv) in enumerate(zip(
            self.upsample_blocks, self.conv_blocks
        )):
            x = upsample(x)
            
            # Get corresponding encoder feature
            enc_feat = encoder_features[-(i + 2)]
            
            # Handle size mismatch
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='trilinear', align_corners=False)
                
            # Concatenate and convolve
            x = torch.cat([x, enc_feat], dim=1)
            x = conv(x)
            
        return self.final_conv(x)


class CoarseLiverSegmentor(nn.Module):
    """Coarse liver segmentation network.
    
    Uses ConvNeXt backbone with U-Net decoder for initial liver localization.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        backbone_variant: str = 'tiny',
        pretrained_path: Optional[str] = None,
        decoder_channels: List[int] = None,
    ):
        """Initialize coarse segmentor.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            backbone_variant: ConvNeXt variant ('tiny', 'small', etc.)
            pretrained_path: Path to pretrained backbone weights
            decoder_channels: Decoder channel dimensions
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build backbone
        self.backbone = ConvNeXt3d(
            in_channels=in_channels,
            variant=backbone_variant,
            out_indices=(0, 1, 2, 3),
        )
        
        # Load pretrained weights
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            
        # Get encoder channels
        encoder_channels = self.backbone.feature_dims
        
        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [256, 128, 64]
            
        # Build decoder
        self.decoder = UNetDecoder3d(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        
    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input CT volume (B, C, D, H, W)
            
        Returns:
            Dictionary with:
            - 'logits': Segmentation logits
            - 'pred': Predicted segmentation (argmax)
        """
        # Encode
        features = self.backbone(x)
        encoder_features = [features[f'stage{i}'] for i in range(4)]
        
        # Decode
        logits = self.decoder(encoder_features)
        
        # Upsample to input size if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:],
                mode='trilinear', align_corners=False
            )
            
        pred = logits.argmax(dim=1)
        
        return {
            'logits': logits,
            'pred': pred,
        }
        
    def get_liver_bbox(
        self,
        pred: torch.Tensor,
        margin: int = 10,
    ) -> List[Tuple[slice, ...]]:
        """Get bounding boxes for liver regions.
        
        Args:
            pred: Predicted segmentation (B, D, H, W)
            margin: Margin around liver region in voxels
            
        Returns:
            List of slice tuples for cropping
        """
        bboxes = []
        
        for b in range(pred.shape[0]):
            mask = pred[b] == 1  # Liver class
            
            if mask.sum() == 0:
                # No liver detected, return full volume
                bboxes.append((
                    slice(0, pred.shape[1]),
                    slice(0, pred.shape[2]),
                    slice(0, pred.shape[3]),
                ))
                continue
                
            # Find bounding box
            nonzero = torch.nonzero(mask)
            z_min, y_min, x_min = nonzero.min(dim=0).values
            z_max, y_max, x_max = nonzero.max(dim=0).values
            
            # Add margin
            z_min = max(0, z_min - margin)
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            z_max = min(pred.shape[1], z_max + margin + 1)
            y_max = min(pred.shape[2], y_max + margin + 1)
            x_max = min(pred.shape[3], x_max + margin + 1)
            
            bboxes.append((
                slice(int(z_min), int(z_max)),
                slice(int(y_min), int(y_max)),
                slice(int(x_min), int(x_max)),
            ))
            
        return bboxes


if __name__ == "__main__":
    # Test coarse segmentor
    model = CoarseLiverSegmentor(
        in_channels=1,
        num_classes=2,
        backbone_variant='tiny',
    )
    
    x = torch.randn(2, 1, 64, 64, 64)
    output = model(x)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Pred shape: {output['pred'].shape}")
    
    # Test bbox extraction
    bboxes = model.get_liver_bbox(output['pred'])
    print(f"Bounding boxes: {bboxes}")
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
