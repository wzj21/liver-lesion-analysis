#!/usr/bin/env python3
"""
肝脏病灶智能分析系统 - 系统测试
Liver Lesion Analysis System - System Test

Tests all module imports and basic forward passes.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test that all modules import successfully."""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)
    
    tests = [
        ("ConvNeXt3D Backbone",
         "from backbones import ConvNeXt3d"),
        ("Stage1 - Cascade Liver Segmentation",
         "from models.stage1_liver_seg import CascadeLiverSegmentation"),
        ("Stage2 - Lesion Detector",
         "from models.stage2_det_cls_deform_seg import LesionDetClsDeformSegNet"),
        ("Stage2 - Deformable Attention",
         "from models.stage2_det_cls_deform_seg.deformable_attention import DeformableAttention3d, DeformableAttention3D"),
        ("Stage2 - Deformable Seg Head",
         "from models.stage2_det_cls_deform_seg.deformable_seg_head import DeformableSegHead"),
        ("Stage3 - Temporal Classifier",
         "from models.stage3_temporal_cls import TemporalLesionClassifier"),
        ("Stage3 - Evidential Classifier",
         "from models.stage3_temporal_cls import EvidentialClassifier"),
        ("Stage3 - Mask Guided Encoder",
         "from models.stage3_temporal_cls import MaskGuidedEncoder"),
        ("Stage3 - Morphology Encoder",
         "from models.stage3_temporal_cls import MorphologyEncoder"),
        ("Stage3 - Temporal Encoder",
         "from models.stage3_temporal_cls import TemporalEncoder, TemporalAggregator"),
        ("Stage4 - Activity Classifier",
         "from models.stage4_activity import EchinococcosisActivityNet"),
        ("Stage4 - Boundary Analyzer",
         "from models.stage4_activity import BoundaryFeatureExtractor"),
        ("Stage4 - Internal Analyzer",
         "from models.stage4_activity import InternalStructureAnalyzer"),
        ("Loss - Dice",
         "from losses import DiceLoss, GeneralizedDiceLoss"),
        ("Loss - Focal Tversky",
         "from losses import FocalTverskyLoss"),
        ("Loss - Hausdorff",
         "from losses import HDLoss, BoundaryLoss"),
        ("Loss - clDice",
         "from losses import clDiceLoss"),
        ("Loss - Evidential",
         "from losses import EvidentialLoss"),
        ("Loss - Detection",
         "from losses import QualityFocalLoss, GIoULoss"),
        ("Preprocessing",
         "from preprocessing import PreprocessingPipeline"),
        ("Pretraining - MAE",
         "from pretraining import MaskedAutoEncoder3D, MAEPreTrainer"),
        ("Pretraining - Contrastive",
         "from pretraining import ContrastiveLearning3D, NTXentLoss"),
        ("Uncertainty",
         "from uncertainty import evidential_layers"),
        ("Semi-supervised",
         "from semi_supervised import trainer"),
        ("Distillation",
         "from distillation import trainer"),
        ("Interpretability",
         "from interpretability import grad_cam"),
        ("Visualization",
         "from visualization import structured_report"),
        ("Validation",
         "from validation import multi_center"),
        ("Data",
         "from data import dataset"),
    ]
    
    passed = 0
    failed = 0
    for name, stmt in tests:
        try:
            exec(stmt)
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_backbone():
    """Test ConvNeXt3D backbone forward pass."""
    print("=" * 60)
    print("Test 2: ConvNeXt3D Backbone")
    print("=" * 60)
    
    from backbones import ConvNeXt3d
    
    model = ConvNeXt3d(in_channels=1, variant='tiny')
    x = torch.randn(1, 1, 32, 32, 32)
    features = model(x)
    
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✓ Backbone test passed!\n")
    return True


def test_stage1():
    """Test Stage1 cascade liver segmentation."""
    print("=" * 60)
    print("Test 3: Stage1 - Liver Segmentation")
    print("=" * 60)
    
    from models.stage1_liver_seg import CascadeLiverSegmentation
    
    model = CascadeLiverSegmentation(
        coarse_backbone='tiny', fine_backbone='tiny',
        coarse_input_size=(32, 32, 32),
    )
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    
    print(f"  Logits: {out['logits'].shape}")
    print(f"  Pred: {out['pred'].shape}")
    print(f"  Uncertainty: {out['uncertainty'].shape}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✓ Stage1 test passed!\n")
    return True


def test_stage2():
    """Test Stage2 lesion detection + segmentation."""
    print("=" * 60)
    print("Test 4: Stage2 - Lesion Detection + Segmentation")
    print("=" * 60)
    
    from models.stage2_det_cls_deform_seg import LesionDetClsDeformSegNet
    
    model = LesionDetClsDeformSegNet(
        in_channels=1, num_queries=20, hidden_dim=128,
        num_encoder_layers=2, num_decoder_layers=2,
    )
    x = torch.randn(1, 1, 32, 32, 32)
    out = model(x)
    
    print(f"  Pred logits: {out['pred_logits'].shape}")
    print(f"  Pred boxes: {out['pred_boxes'].shape}")
    print(f"  Pred masks: {out['pred_masks'].shape}")
    print(f"  Has lesion: {out['has_lesion']}")
    
    det = model.get_detections(out, score_threshold=0.3)
    print(f"  Detections: {len(det[0]['scores'])} lesions")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✓ Stage2 test passed!\n")
    return True


def test_stage3():
    """Test Stage3 temporal lesion classification."""
    print("=" * 60)
    print("Test 5: Stage3 - Temporal Classification")
    print("=" * 60)
    
    from models.stage3_temporal_cls import TemporalLesionClassifier
    
    model = TemporalLesionClassifier(
        slice_encoder_variant='tiny',
        temporal_hidden_dim=128,
        temporal_num_layers=1,
        temporal_num_heads=4,
        num_classes=4,
    )
    
    B, K = 1, 4
    slices = torch.randn(B, K, 1, 56, 56)
    masks = (torch.rand(B, K, 1, 56, 56) > 0.7).float()
    morphology = torch.randn(B, 16)
    
    out = model(slices, masks, morphology)
    
    print(f"  Probs: {out['probs'].shape}")
    print(f"  Prediction: {out['pred']}")
    print(f"  Uncertainty: {out['uncertainty']}")
    print(f"  Slice weights: {out['slice_weights'].shape}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✓ Stage3 test passed!\n")
    return True


def test_stage4():
    """Test Stage4 activity classification."""
    print("=" * 60)
    print("Test 6: Stage4 - Activity Classification")
    print("=" * 60)
    
    from models.stage4_activity import EchinococcosisActivityNet
    
    model = EchinococcosisActivityNet(
        boundary_output_dim=64,
        internal_output_dim=64,
        stage3_feature_dim=128,
        use_3d=True,
    )
    
    B = 1
    ct = torch.randn(B, 1, 16, 32, 32) * 50 + 30
    mask = (torch.rand(B, 1, 16, 32, 32) > 0.7).float()
    s3_features = torch.randn(B, 128)
    lesion_type = torch.tensor([0])  # 0=CE
    
    out = model(ct, mask, s3_features, lesion_type)
    
    print(f"  Prediction: {out['pred']}")
    print(f"  Active prob: {out['active_prob']}")
    print(f"  Uncertainty: {out['uncertainty']}")
    
    assessments = model.get_activity_assessment(out, lesion_type)
    for k, v in assessments[0].items():
        print(f"  {k}: {v}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✓ Stage4 test passed!\n")
    return True


def test_pretraining():
    """Test self-supervised pretraining modules."""
    print("=" * 60)
    print("Test 7: Self-Supervised Pretraining")
    print("=" * 60)
    
    from pretraining import MaskedAutoEncoder3D, ContrastiveLearning3D
    
    # MAE
    mae = MaskedAutoEncoder3D(
        volume_size=(32, 32, 32), patch_size=(8, 8, 8),
        encoder_embed_dim=192, encoder_depth=3, encoder_num_heads=3,
        decoder_embed_dim=96, decoder_depth=1, decoder_num_heads=3,
    )
    x = torch.randn(1, 1, 32, 32, 32)
    out = mae(x)
    print(f"  MAE loss: {out['loss'].item():.4f}")
    print(f"  MAE pred: {out['pred'].shape}")
    print(f"  MAE mask ratio: {out['mask'].float().mean():.2f}")
    
    encoder = mae.get_encoder()
    enc_out = encoder(x)
    print(f"  Encoder output: {enc_out.shape}")
    
    # Contrastive
    cl = ContrastiveLearning3D(feature_dim=128, projection_dim=64)
    v1 = torch.randn(2, 1, 32, 32, 32)
    v2 = torch.randn(2, 1, 32, 32, 32)
    cl_out = cl(v1, v2)
    print(f"  Contrastive loss: {cl_out['loss'].item():.4f}")
    
    print("  ✓ Pretraining test passed!\n")
    return True


def test_losses():
    """Test loss functions."""
    print("=" * 60)
    print("Test 8: Loss Functions")
    print("=" * 60)
    
    from losses import DiceLoss, FocalTverskyLoss, HDLoss, clDiceLoss, EvidentialLoss
    
    pred = torch.randn(2, 2, 16, 16, 16)
    target = (torch.rand(2, 1, 16, 16, 16) > 0.5).float()
    
    dice = DiceLoss(sigmoid=True)
    loss = dice(pred[:, :1], target)
    print(f"  Dice Loss: {loss.item():.4f}")
    
    tversky = FocalTverskyLoss(alpha=0.7, beta=0.3)
    loss = tversky(pred[:, :1], target)
    print(f"  Focal Tversky Loss: {loss.item():.4f}")
    
    print("  ✓ Loss function test passed!\n")
    return True


def test_deformable_seg_head():
    """Test the deformable segmentation head."""
    print("=" * 60)
    print("Test 9: Deformable Segmentation Head")
    print("=" * 60)
    
    from models.stage2_det_cls_deform_seg.deformable_seg_head import DeformableSegHead
    
    head = DeformableSegHead(
        hidden_dim=128, fpn_dims=[128, 128, 128],
        num_mask_convs=2, mask_dim=16,
    )
    
    queries = torch.randn(1, 5, 128)
    feats = [
        torch.randn(1, 128, 8, 8, 8),
        torch.randn(1, 128, 4, 4, 4),
        torch.randn(1, 128, 2, 2, 2),
    ]
    out = head(queries, feats, (16, 16, 16))
    print(f"  Masks shape: {out['masks'].shape}")
    
    params = sum(p.numel() for p in head.parameters()) / 1e6
    print(f"  Parameters: {params:.4f}M")
    print("  ✓ DeformableSegHead test passed!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("   肝脏病灶智能分析系统 - 完整系统测试")
    print("   Liver Lesion Analysis System - Full System Test")
    print("=" * 60 + "\n")
    
    results = {}
    
    tests = [
        ("Module Imports", test_imports),
        ("ConvNeXt3D Backbone", test_backbone),
        ("Stage1 Segmentation", test_stage1),
        ("Stage2 Detection", test_stage2),
        ("Stage3 Classification", test_stage3),
        ("Stage4 Activity", test_stage4),
        ("Pretraining", test_pretraining),
        ("Loss Functions", test_losses),
        ("Deformable Seg Head", test_deformable_seg_head),
    ]
    
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results[name] = "✓ PASS" if ok else "✗ FAIL"
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}\n")
            results[name] = f"✗ FAIL ({e})"
    
    # Summary
    print("=" * 60)
    print("   SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result}  {name}")
    
    passed = sum(1 for v in results.values() if "PASS" in v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
