"""
Cascade Liver Segmentation Pipeline
级联肝脏分割流水线

Combines coarse and fine segmentation for high-quality results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np

from .coarse_segmentor import CoarseLiverSegmentor
from .fine_segmentor import FineLiverSegmentor


class CascadeLiverSegmentation(nn.Module):
    """Cascade liver segmentation combining coarse and fine networks.
    
    Pipeline:
    1. Coarse segmentation at low resolution
    2. Extract liver ROI with bounding box
    3. Fine segmentation at high resolution
    4. Post-processing (morphology, connected components)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        coarse_backbone: str = 'tiny',
        fine_backbone: str = 'small',
        coarse_pretrained: Optional[str] = None,
        fine_pretrained: Optional[str] = None,
        coarse_input_size: Tuple[int, int, int] = (128, 128, 128),
        roi_margin: int = 10,
    ):
        """Initialize cascade segmentation.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            coarse_backbone: Backbone variant for coarse network
            fine_backbone: Backbone variant for fine network
            coarse_pretrained: Pretrained weights for coarse network
            fine_pretrained: Pretrained weights for fine network
            coarse_input_size: Input size for coarse network
            roi_margin: Margin around liver ROI in voxels
        """
        super().__init__()
        
        self.coarse_input_size = coarse_input_size
        self.roi_margin = roi_margin
        
        # Coarse segmentation network
        self.coarse_net = CoarseLiverSegmentor(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone_variant=coarse_backbone,
            pretrained_path=coarse_pretrained,
        )
        
        # Fine segmentation network
        self.fine_net = FineLiverSegmentor(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone_variant=fine_backbone,
            pretrained_path=fine_pretrained,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input CT volume (B, C, D, H, W)
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary with segmentation outputs
        """
        original_size = x.shape[2:]
        
        # Step 1: Coarse segmentation
        x_coarse = F.interpolate(
            x, size=self.coarse_input_size,
            mode='trilinear', align_corners=False
        )
        coarse_out = self.coarse_net(x_coarse)
        
        # Upsample coarse prediction to original size
        coarse_pred = F.interpolate(
            coarse_out['logits'], size=original_size,
            mode='trilinear', align_corners=False
        ).argmax(dim=1)
        
        # Step 2: Extract ROI and run fine segmentation
        bboxes = self.coarse_net.get_liver_bbox(coarse_pred, margin=self.roi_margin)
        
        fine_outputs = []
        for b, bbox in enumerate(bboxes):
            # Crop ROI
            x_roi = x[b:b+1, :, bbox[0], bbox[1], bbox[2]]
            
            # Run fine segmentation
            fine_out = self.fine_net(x_roi)
            fine_outputs.append({
                'bbox': bbox,
                'output': fine_out,
            })
            
        # Step 3: Combine fine outputs into full volume
        final_logits = torch.zeros_like(coarse_out['logits'])
        final_logits = F.interpolate(final_logits, size=original_size, mode='trilinear', align_corners=False)
        final_uncertainty = torch.zeros(x.shape[0], *original_size, device=x.device)
        
        for b, fine_data in enumerate(fine_outputs):
            bbox = fine_data['bbox']
            fine_out = fine_data['output']
            
            # Resize fine output to ROI size
            roi_size = (
                bbox[0].stop - bbox[0].start,
                bbox[1].stop - bbox[1].start,
                bbox[2].stop - bbox[2].start,
            )
            
            fine_logits = F.interpolate(
                fine_out['logits'], size=roi_size,
                mode='trilinear', align_corners=False
            )
            
            # Place in full volume
            final_logits[b, :, bbox[0], bbox[1], bbox[2]] = fine_logits[0]
            
            if 'uncertainty' in fine_out:
                fine_unc = F.interpolate(
                    fine_out['uncertainty'].unsqueeze(1), size=roi_size,
                    mode='trilinear', align_corners=False
                ).squeeze(1)
                final_uncertainty[b, bbox[0], bbox[1], bbox[2]] = fine_unc[0]
                
        result = {
            'logits': final_logits,
            'pred': final_logits.argmax(dim=1),
            'uncertainty': final_uncertainty,
        }
        
        if return_intermediate:
            result['coarse_pred'] = coarse_pred
            result['bboxes'] = bboxes
            result['fine_outputs'] = fine_outputs
            
        return result
        
    def postprocess(
        self,
        pred: torch.Tensor,
        min_volume: int = 1000,
        closing_kernel: int = 5,
        opening_kernel: int = 3,
    ) -> torch.Tensor:
        """Post-process segmentation.
        
        Args:
            pred: Predicted segmentation (B, D, H, W)
            min_volume: Minimum volume in voxels
            closing_kernel: Kernel size for morphological closing
            opening_kernel: Kernel size for morphological opening
            
        Returns:
            Post-processed segmentation
        """
        try:
            from scipy import ndimage
        except ImportError:
            print("scipy required for post-processing")
            return pred
            
        result = torch.zeros_like(pred)
        
        for b in range(pred.shape[0]):
            mask = pred[b].cpu().numpy()
            
            # Morphological closing (fill holes)
            if closing_kernel > 0:
                struct = ndimage.generate_binary_structure(3, 1)
                struct = ndimage.iterate_structure(struct, closing_kernel // 2)
                mask = ndimage.binary_closing(mask, structure=struct)
                
            # Morphological opening (remove noise)
            if opening_kernel > 0:
                struct = ndimage.generate_binary_structure(3, 1)
                struct = ndimage.iterate_structure(struct, opening_kernel // 2)
                mask = ndimage.binary_opening(mask, structure=struct)
                
            # Keep largest connected component
            labeled, num_features = ndimage.label(mask)
            if num_features > 0:
                component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
                largest_component = np.argmax(component_sizes) + 1
                mask = (labeled == largest_component)
                
                # Remove small components
                if component_sizes[largest_component - 1] < min_volume:
                    mask = np.zeros_like(mask)
                    
            result[b] = torch.from_numpy(mask.astype(np.float32)).to(pred.device)
            
        return result


if __name__ == "__main__":
    # Test cascade segmentation
    model = CascadeLiverSegmentation(
        in_channels=1,
        num_classes=2,
        coarse_backbone='tiny',
        fine_backbone='tiny',
        coarse_input_size=(64, 64, 64),
    )
    
    x = torch.randn(2, 1, 128, 128, 128)
    output = model(x, return_intermediate=True)
    
    print(f"Final logits shape: {output['logits'].shape}")
    print(f"Final pred shape: {output['pred'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Coarse pred shape: {output['coarse_pred'].shape}")
    print(f"Bounding boxes: {output['bboxes']}")
    
    # Post-process
    processed = model.postprocess(output['pred'])
    print(f"Post-processed shape: {processed.shape}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")
