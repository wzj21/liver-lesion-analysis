"""
知识蒸馏与模型轻量化模块
Knowledge Distillation and Model Compression Module

本模块提供完整的知识蒸馏和模型压缩解决方案:

主要功能:
1. 知识蒸馏
   - 特征蒸馏 (Feature Distillation)
   - 输出蒸馏 (Logit Distillation)
   - 关系蒸馏 (Relational Knowledge Distillation)
   - 注意力蒸馏 (Attention Transfer)
   - 不确定性蒸馏 (Uncertainty Distillation)

2. 模型轻量化
   - 轻量级学生模型设计
   - 通道数减少
   - 层数减少

3. 模型量化
   - 动态量化 (Dynamic Quantization)
   - 静态量化 (Static Quantization)
   - 量化感知训练 (QAT)

4. 推理优化
   - ONNX导出
   - TensorRT优化
   - 推理引擎封装

模型版本对比:
--------------
| 版本 | 参数量 | GPU推理 | CPU推理 | 精度保持 |
|------|--------|---------|---------|----------|
| 完整版 | 62M | <4s | <10s | 100% |
| 轻量版(FP32) | 25M | <2s | <5s | 97% |
| 轻量版(FP16) | 25M | <1.5s | <4s | 97% |
| 轻量版(INT8) | 25M | <1s | <3s | 95% |

使用示例:
---------
# 1. 创建学生模型
from distillation import create_student_model, LightweightLiverLesionModel

student = create_student_model({
    'in_channels': 1,
    'num_seg_classes': 2,
    'num_cls_classes': 4,
    'use_evidential': True
})

# 2. 知识蒸馏训练
from distillation import DistillationTrainer

trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=distillation_config
)
trainer.train()

# 3. 模型量化
from distillation import ModelQuantizer

quantizer = ModelQuantizer(student)
quantized_model = quantizer.dynamic_quantize()

# 4. 导出ONNX
from distillation import ONNXExporter

exporter = ONNXExporter(student)
exporter.export('model.onnx')

# 5. TensorRT优化
from distillation import TensorRTOptimizer

optimizer = TensorRTOptimizer()
optimizer.optimize('model.onnx', 'model.engine', precision='fp16')

# 6. 动态推理
from distillation import DynamicInference

dynamic_inference = DynamicInference(teacher, student, difficulty_threshold=0.3)
result = dynamic_inference(input_image)
"""

from .losses import (
    FeatureDistillationLoss,
    FeatureProjector,
    OutputDistillationLoss,
    SegmentationDistillationLoss,
    DetectionDistillationLoss,
    RelationDistillationLoss,
    AttentionDistillationLoss,
    UncertaintyDistillationLoss,
    CombinedDistillationLoss
)

from .student_models import (
    ConvNeXtFemtoBlock,
    LightweightConvNeXt3D,
    LightweightSegmentationHead,
    LightweightDetectionHead,
    LightweightClassificationHead,
    LightweightLiverLesionModel,
    create_student_model
)

from .trainer import (
    DistillationTrainer,
    DynamicInference
)

from .quantization import (
    ModelQuantizer,
    ONNXExporter,
    TensorRTOptimizer,
    InferenceEngine
)


__all__ = [
    # Losses
    'FeatureDistillationLoss',
    'FeatureProjector',
    'OutputDistillationLoss',
    'SegmentationDistillationLoss',
    'DetectionDistillationLoss',
    'RelationDistillationLoss',
    'AttentionDistillationLoss',
    'UncertaintyDistillationLoss',
    'CombinedDistillationLoss',
    
    # Student Models
    'ConvNeXtFemtoBlock',
    'LightweightConvNeXt3D',
    'LightweightSegmentationHead',
    'LightweightDetectionHead',
    'LightweightClassificationHead',
    'LightweightLiverLesionModel',
    'create_student_model',
    
    # Trainer
    'DistillationTrainer',
    'DynamicInference',
    
    # Quantization & Optimization
    'ModelQuantizer',
    'ONNXExporter',
    'TensorRTOptimizer',
    'InferenceEngine'
]

__version__ = '1.0.0'
