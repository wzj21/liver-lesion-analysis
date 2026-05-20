"""
Liver Lesion Analysis Pipeline
肝脏病灶智能分析流水线

Complete end-to-end pipeline for liver lesion detection, segmentation, and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
import yaml
import logging

from .preprocessing.pipeline import PreprocessingPipeline
from .models.stage1_liver_seg import CascadeLiverSegmentation
from .models.stage2_det_cls_deform_seg import LesionDetClsDeformSegNet
from .models.stage3_temporal_cls import TemporalLesionClassifier
from .models.stage3_temporal_cls.morphology_encoder import (
    compute_morphological_features, features_to_tensor
)
from .models.stage3_temporal_cls.mask_guided_encoder import extract_mask_boundary
from .models.stage4_activity import EchinococcosisActivityNet

logger = logging.getLogger(__name__)


class LiverLesionAnalysisPipeline:
    """Complete pipeline for liver lesion analysis.
    
    Pipeline stages:
    1. Preprocessing (window normalization, resampling)
    2. Stage1: Liver segmentation
    3. Stage2: Lesion detection + classification + segmentation
    4. Stage3: Temporal lesion classification (4-class)
    5. Stage4: Echinococcosis activity classification
    6. Report generation
    """
    
    STAGE3_CLASSES = {
        0: ('benign', '良性'),
        1: ('malignant', '恶性'),
        2: ('cystic_echinococcosis', '肝囊型包虫病'),
        3: ('alveolar_echinococcosis', '肝泡型包虫病'),
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = 'cuda',
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        self._build_pipeline()
        
    def _default_config(self) -> Dict:
        return {
            'preprocessing': {
                'window_width': 160,
                'window_level': 60,
                'target_spacing': [1.0, 1.0, 1.0],
            },
            'stage1': {
                'coarse_backbone': 'tiny',
                'fine_backbone': 'small',
                'coarse_input_size': [128, 128, 128],
            },
            'stage2': {
                'backbone': 'small',
                'num_queries': 100,
                'hidden_dim': 256,
            },
            'stage3': {
                'num_slices': 16,
                'num_classes': 4,
                'slice_encoder_variant': 'tiny',
                'temporal_hidden_dim': 512,
                'temporal_num_layers': 4,
                'temporal_num_heads': 8,
            },
            'stage4': {
                'num_classes': 2,
                'boundary_output_dim': 128,
                'internal_output_dim': 128,
            },
        }
        
    def _build_pipeline(self):
        """Build pipeline components."""
        self.preprocessor = PreprocessingPipeline(
            window_width=self.config['preprocessing']['window_width'],
            window_level=self.config['preprocessing']['window_level'],
            target_spacing=tuple(self.config['preprocessing']['target_spacing']),
        )
        
        self.stage1 = CascadeLiverSegmentation(
            coarse_backbone=self.config['stage1']['coarse_backbone'],
            fine_backbone=self.config['stage1']['fine_backbone'],
            coarse_input_size=tuple(self.config['stage1']['coarse_input_size']),
        ).to(self.device)
        
        self.stage2 = LesionDetClsDeformSegNet(
            backbone_variant=self.config['stage2']['backbone'],
            num_queries=self.config['stage2']['num_queries'],
            hidden_dim=self.config['stage2']['hidden_dim'],
        ).to(self.device)
        
        s3_cfg = self.config['stage3']
        self.stage3 = TemporalLesionClassifier(
            slice_encoder_variant=s3_cfg.get('slice_encoder_variant', 'tiny'),
            temporal_hidden_dim=s3_cfg.get('temporal_hidden_dim', 512),
            temporal_num_layers=s3_cfg.get('temporal_num_layers', 4),
            temporal_num_heads=s3_cfg.get('temporal_num_heads', 8),
            num_classes=s3_cfg.get('num_classes', 4),
        ).to(self.device)
        
        s4_cfg = self.config['stage4']
        self.stage4 = EchinococcosisActivityNet(
            boundary_output_dim=s4_cfg.get('boundary_output_dim', 128),
            internal_output_dim=s4_cfg.get('internal_output_dim', 128),
            stage3_feature_dim=s3_cfg.get('temporal_hidden_dim', 512),
        ).to(self.device)
        
    def load_weights(
        self,
        stage1_path: Optional[str] = None,
        stage2_path: Optional[str] = None,
        stage3_path: Optional[str] = None,
        stage4_path: Optional[str] = None,
    ):
        """Load model weights for each stage."""
        weight_map = {
            'stage1': (self.stage1, stage1_path),
            'stage2': (self.stage2, stage2_path),
            'stage3': (self.stage3, stage3_path),
            'stage4': (self.stage4, stage4_path),
        }
        for name, (model, path) in weight_map.items():
            if path and model is not None:
                try:
                    state_dict = torch.load(path, map_location=self.device, weights_only=True)
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded {name} weights from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name} weights from {path}: {e}")
            
    @torch.no_grad()
    def analyze(
        self,
        ct_path: Union[str, Path, np.ndarray],
        spacing: Optional[tuple] = None,
        return_intermediate: bool = False,
    ) -> Dict:
        """Run complete analysis pipeline."""
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            stage.eval()
        
        # Step 1: Preprocessing
        if isinstance(ct_path, np.ndarray):
            prep_result = self.preprocessor(ct_path, spacing=spacing)
        else:
            prep_result = self.preprocessor(ct_path)
            
        ct_normalized = prep_result['image']
        ct_tensor = torch.from_numpy(ct_normalized).float()
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        results = {'preprocessing': prep_result['metadata']}
        
        # Step 2: Stage1 - Liver segmentation
        stage1_out = self.stage1(ct_tensor)
        liver_mask = stage1_out['pred'][0].cpu().numpy()
        results['stage1'] = {
            'liver_mask': liver_mask,
            'liver_uncertainty': stage1_out['uncertainty'][0].cpu().numpy(),
        }
        
        # Extract liver ROI
        liver_coords = np.argwhere(liver_mask > 0)
        if len(liver_coords) > 0:
            mins = np.maximum(liver_coords.min(axis=0) - 10, 0)
            maxs = np.minimum(liver_coords.max(axis=0) + 11, ct_normalized.shape)
            z_min, y_min, x_min = mins
            z_max, y_max, x_max = maxs
            liver_roi = ct_tensor[:, :, z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            liver_roi = ct_tensor
            z_min, y_min, x_min = 0, 0, 0
            z_max, y_max, x_max = ct_normalized.shape
            
        # Step 3: Stage2 - Lesion detection/segmentation
        stage2_out = self.stage2(liver_roi)
        has_lesion = stage2_out['has_lesion'][0].item() > 0.5
        
        results['stage2'] = {
            'has_lesion': has_lesion,
            'confidence': float(stage2_out['pred_logits'].softmax(dim=-1)[:, :, 1].max()),
            'num_detections': 0,
            'lesion_masks': None,
            'lesion_boxes': None,
        }
        
        if has_lesion:
            detections = self.stage2.get_detections(stage2_out, score_threshold=0.5)
            if len(detections[0]['scores']) > 0:
                results['stage2']['num_detections'] = len(detections[0]['scores'])
                results['stage2']['lesion_boxes'] = detections[0]['boxes'].cpu().numpy()
                
                full_masks = np.zeros((len(detections[0]['scores']),) + ct_normalized.shape)
                for i, mask in enumerate(detections[0]['masks']):
                    mask_np = mask.sigmoid().cpu().numpy()
                    d_s = min(mask_np.shape[0], z_max - z_min)
                    h_s = min(mask_np.shape[1], y_max - y_min)
                    w_s = min(mask_np.shape[2], x_max - x_min)
                    full_masks[i, z_min:z_min+d_s, y_min:y_min+h_s, x_min:x_min+w_s] = \
                        mask_np[:d_s, :h_s, :w_s]
                results['stage2']['lesion_masks'] = full_masks
                
        # Step 4: Stage3 - Classification
        if has_lesion and results['stage2']['num_detections'] > 0:
            s3_result = self._run_stage3(
                ct_normalized, results['stage2']['lesion_masks'][0],
                spacing=prep_result.get('metadata', {}).get('spacing', (1., 1., 1.))
            )
            results['stage3'] = s3_result
            
            # Step 5: Stage4 - Activity (if echinococcosis)
            if s3_result['pred_class'] in [2, 3]:
                roi_mask = results['stage2']['lesion_masks'][0][z_min:z_max, y_min:y_max, x_min:x_max]
                s4_result = self._run_stage4(
                    liver_roi,
                    torch.from_numpy(roi_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device),
                    s3_result['aggregated_features'],
                    lesion_type=s3_result['pred_class'] - 2
                )
                results['stage4'] = s4_result
        
        results['summary'] = self._generate_summary(results)
        return results
    
    def _run_stage3(self, ct_volume, lesion_mask, spacing=(1., 1., 1.)):
        """Run Stage3 temporal classification."""
        num_slices = self.config['stage3'].get('num_slices', 16)
        
        slice_content = lesion_mask.sum(axis=(1, 2))
        nonzero_slices = np.where(slice_content > 0)[0]
        
        if len(nonzero_slices) == 0:
            return {'pred_class': -1, 'probs': np.zeros(4), 'uncertainty': 1.0,
                    'aggregated_features': torch.zeros(1, self.config['stage3']['temporal_hidden_dim'],
                                                        device=self.device)}
        
        if len(nonzero_slices) > num_slices:
            top_idx = np.argsort(slice_content[nonzero_slices])[::-1][:num_slices]
            selected = sorted(nonzero_slices[top_idx])
        else:
            selected = list(nonzero_slices)
            while len(selected) < num_slices:
                selected.append(selected[-1])
        
        K = len(selected)
        H, W = ct_volume.shape[1], ct_volume.shape[2]
        
        slices_t = torch.zeros(1, K, 1, H, W, device=self.device)
        masks_t = torch.zeros(1, K, 1, H, W, device=self.device)
        
        for i, s_idx in enumerate(selected):
            slices_t[0, i, 0] = torch.from_numpy(ct_volume[s_idx]).float()
            masks_t[0, i, 0] = torch.from_numpy(lesion_mask[s_idx].astype(np.float32))
        
        morph_features = compute_morphological_features(ct_volume, lesion_mask, spacing)
        morph_t = features_to_tensor(morph_features).unsqueeze(0).to(self.device)
        
        output = self.stage3(slices_t, masks_t, morph_t, return_attention=True)
        
        pred_class = output['pred'][0].item()
        class_en, class_cn = self.STAGE3_CLASSES.get(pred_class, ('unknown', '未知'))
        
        return {
            'pred_class': pred_class,
            'class_name_en': class_en,
            'class_name_cn': class_cn,
            'probs': output['probs'][0].cpu().numpy(),
            'uncertainty': output['uncertainty'][0].item(),
            'aggregated_features': output['aggregated_features'],
            'slice_weights': output['slice_weights'][0].cpu().numpy() if output.get('slice_weights') is not None else None,
            'selected_slices': selected,
        }
    
    def _run_stage4(self, ct_roi, mask_roi, stage3_features, lesion_type):
        """Run Stage4 activity classification."""
        lt_tensor = torch.tensor([lesion_type], device=self.device)
        output = self.stage4(ct_roi, mask_roi, stage3_features, lt_tensor)
        assessments = self.stage4.get_activity_assessment(output, lt_tensor)
        
        return {
            'pred': output['pred'][0].item(),
            'active_prob': output['active_prob'][0].item(),
            'inactive_prob': output['inactive_prob'][0].item(),
            'uncertainty': output['uncertainty'][0].item(),
            'assessment': assessments[0] if assessments else {},
        }
        
    def _generate_summary(self, results):
        summary = {
            'status': 'completed',
            'has_lesion': results['stage2']['has_lesion'],
            'num_lesions': results['stage2']['num_detections'],
            'detection_confidence': results['stage2']['confidence'],
            'needs_review': False,
        }
        
        if results['stage2']['confidence'] < 0.7:
            summary['needs_review'] = True
            summary['review_reason'] = '检测置信度较低'
        
        if 'stage3' in results and results['stage3'].get('pred_class', -1) >= 0:
            s3 = results['stage3']
            summary['classification'] = s3['class_name_cn']
            summary['classification_en'] = s3['class_name_en']
            if s3['uncertainty'] > 0.5:
                summary['needs_review'] = True
                summary['review_reason'] = '分类不确定性较高'
        
        if 'stage4' in results:
            summary['activity_status'] = results['stage4'].get('assessment', {}).get('activity_status_cn', '')
            
        return summary
        
    def generate_report(self, results, output_path=None):
        """Generate structured report."""
        lines = [
            "=" * 60, "肝脏病灶智能分析报告 / Liver Lesion Analysis Report", "=" * 60, "",
            "【检查摘要】",
            f"  检测状态: {'检测到病灶' if results['summary']['has_lesion'] else '未检测到病灶'}",
            f"  病灶数量: {results['summary']['num_lesions']}",
            f"  检测置信度: {results['summary']['detection_confidence']:.2%}",
            f"  是否需要复核: {'是' if results['summary']['needs_review'] else '否'}", "",
        ]
        
        if 'stage3' in results and results['stage3'].get('pred_class', -1) >= 0:
            s3 = results['stage3']
            lines.extend([
                "【病灶分类】",
                f"  诊断: {s3['class_name_cn']} ({s3['class_name_en']})",
                f"  不确定性: {s3['uncertainty']:.4f}", "",
            ])
            
        if 'stage4' in results:
            a = results['stage4'].get('assessment', {})
            lines.extend([
                "【包虫病活性评估】",
                f"  活性: {a.get('activity_status_cn', 'N/A')}",
                f"  建议: {a.get('interpretation', 'N/A')}", "",
            ])
            
        if results['summary']['needs_review']:
            lines.extend(["【复核建议】", f"  原因: {results['summary'].get('review_reason', '')}", ""])
            
        lines.append("=" * 60)
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        return report
