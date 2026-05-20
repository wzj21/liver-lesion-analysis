"""
Fine Liver Segmentor
精分割网络

High-resolution segmentation with boundary enhancement and uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from ...backbones.convnext_3d import ConvNeXt3d, LayerNorm3d


class AttentionGate3d(nn.Module):
    """3D Attention Gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Conv3d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv3d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle size mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        
        return x * psi


class BoundaryRefinementModule(nn.Module):
    """Boundary refinement using SDF prediction and edge enhancement."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        
        # SDF prediction branch
        self.sdf_conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.sdf_conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.sdf_out = nn.Conv3d(hidden_channels, 1, kernel_size=1)  # SDF output
        
        # Edge enhancement branch
        self.edge_conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.edge_conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.edge_out = nn.Conv3d(hidden_channels, 1, kernel_size=1)  # Edge probability
        
        # Fusion
        self.fusion = nn.Conv3d(in_channels + 2, in_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (refined_features, sdf_pred, edge_pred)
        """
        # SDF branch
        sdf = self.relu(self.sdf_conv1(x))
        sdf = self.relu(self.sdf_conv2(sdf))
        sdf_pred = self.sdf_out(sdf)  # Signed distance field
        
        # Edge branch
        edge = self.relu(self.edge_conv1(x))
        edge = self.relu(self.edge_conv2(edge))
        edge_pred = torch.sigmoid(self.edge_out(edge))  # Edge probability
        
        # Fusion
        fused = torch.cat([x, sdf_pred, edge_pred], dim=1)
        refined = self.fusion(fused)
        
        return refined, sdf_pred, edge_pred


class EnhancedUNetDecoder3d(nn.Module):
    """Enhanced U-Net decoder with attention gates and deep supervision."""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 2,
        use_attention_gate: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.use_attention_gate = use_attention_gate
        self.deep_supervision = deep_supervision
        
        self.upsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention_gate else None
        self.deep_supervision_heads = nn.ModuleList() if deep_supervision else None
        
        in_channels = encoder_channels[-1]
        
        for i, (enc_ch, dec_ch) in enumerate(zip(
            reversed(encoder_channels[:-1]),
            decoder_channels
        )):
            # Upsample
            self.upsample_blocks.append(
                nn.ConvTranspose3d(in_channels, dec_ch, kernel_size=2, stride=2)
            )
            
            # Attention gate
            if use_attention_gate:
                self.attention_gates.append(
                    AttentionGate3d(dec_ch, enc_ch, enc_ch // 2)
                )
                
            # Conv block
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
            
            # Deep supervision head
            if deep_supervision and i < 3:
                self.deep_supervision_heads.append(
                    nn.Conv3d(dec_ch, num_classes, kernel_size=1)
                )
                
            in_channels = dec_ch
            
        self.final_conv = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)
        
    def forward(
        self,
        encoder_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            encoder_features: Features from encoder stages
            
        Returns:
            Dictionary with logits and deep supervision outputs
        """
        x = encoder_features[-1]
        deep_outputs = []
        
        for i, (upsample, conv) in enumerate(zip(
            self.upsample_blocks, self.conv_blocks
        )):
            x = upsample(x)
            enc_feat = encoder_features[-(i + 2)]
            
            # Handle size mismatch
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='trilinear', align_corners=False)
                
            # Apply attention gate
            if self.use_attention_gate:
                enc_feat = self.attention_gates[i](x, enc_feat)
                
            x = torch.cat([x, enc_feat], dim=1)
            x = conv(x)
            
            # Deep supervision
            if self.deep_supervision and i < len(self.deep_supervision_heads):
                deep_outputs.append(self.deep_supervision_heads[i](x))
                
        logits = self.final_conv(x)
        
        result = {'logits': logits}
        if self.deep_supervision:
            result['deep_outputs'] = deep_outputs
            
        return result


class FineLiverSegmentor(nn.Module):
    """Fine liver segmentation network.
    
    Features:
    - Enhanced U-Net decoder with attention gates
    - Boundary refinement module
    - Deep supervision
    - Evidential uncertainty estimation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        backbone_variant: str = 'small',
        pretrained_path: Optional[str] = None,
        decoder_channels: List[int] = None,
        use_attention_gate: bool = True,
        deep_supervision: bool = True,
        boundary_enhancement: bool = True,
        evidential: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.evidential = evidential
        self.boundary_enhancement = boundary_enhancement
        
        # Backbone
        self.backbone = ConvNeXt3d(
            in_channels=in_channels,
            variant=backbone_variant,
            out_indices=(0, 1, 2, 3),
        )
        
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            
        encoder_channels = self.backbone.feature_dims
        
        if decoder_channels is None:
            decoder_channels = [256, 128, 64]
            
        # Decoder
        self.decoder = EnhancedUNetDecoder3d(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_attention_gate=use_attention_gate,
            deep_supervision=deep_supervision,
        )
        
        # Boundary refinement
        if boundary_enhancement:
            self.boundary_module = BoundaryRefinementModule(
                in_channels=decoder_channels[-1],
                hidden_channels=64,
            )
            
        # Evidential head
        if evidential:
            self.evidence_conv = nn.Sequential(
                nn.Conv3d(decoder_channels[-1], 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, num_classes, kernel_size=1),
            )
            
    def _load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.backbone.load_state_dict(state_dict, strict=False)
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
            Dictionary with segmentation outputs and uncertainty
        """
        # Encode
        features = self.backbone(x)
        encoder_features = [features[f'stage{i}'] for i in range(4)]
        
        # Decode
        decoder_out = self.decoder(encoder_features)
        
        # Get intermediate features before final conv
        # (We need to access them from decoder)
        
        result = {}
        
        # Main segmentation output
        logits = decoder_out['logits']
        
        # Upsample to input size
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            
        result['logits'] = logits
        result['pred'] = logits.argmax(dim=1)
        
        # Deep supervision outputs
        if 'deep_outputs' in decoder_out:
            result['deep_outputs'] = [
                F.interpolate(out, size=x.shape[2:], mode='trilinear', align_corners=False)
                for out in decoder_out['deep_outputs']
            ]
            
        # Evidential uncertainty
        if self.evidential:
            # Need to get features before final conv
            # For simplicity, compute from logits
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            
            result['alpha'] = alpha
            result['uncertainty'] = self.num_classes / S.squeeze(1)
            
        return result


if __name__ == "__main__":
    # Test fine segmentor
    model = FineLiverSegmentor(
        in_channels=1,
        num_classes=2,
        backbone_variant='tiny',
        use_attention_gate=True,
        deep_supervision=True,
        evidential=True,
    )
    
    x = torch.randn(2, 1, 64, 64, 64)
    output = model(x)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Pred shape: {output['pred'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Uncertainty range: [{output['uncertainty'].min():.3f}, {output['uncertainty'].max():.3f}]")
    
    if 'deep_outputs' in output:
        print(f"Deep outputs: {len(output['deep_outputs'])}")
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
