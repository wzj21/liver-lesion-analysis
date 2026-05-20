"""
半监督学习框架模块
Semi-Supervised Learning Framework

本模块提供完整的半监督学习解决方案，用于肝脏病灶智能分析系统的各个阶段：

- Stage1 (肝脏分割): Cross Pseudo Supervision (CPS) / Mean Teacher
- Stage2 (检测分割): Teacher-Student Framework  
- Stage3 (分类): FixMatch / FlexMatch / Evidential FixMatch

主要组件:
- base: 基础模块（EMA、伪标签生成、一致性损失等）
- teacher_student: Teacher-Student半监督框架
- fixmatch: FixMatch及其变体
- cross_pseudo_supervision: CPS和Mean Teacher
- augmentation: 数据增强（弱增强、强增强、Copy-Paste）
- trainer: 统一训练接口

使用示例:
---------
# Stage1: 肝脏分割 - 使用CPS
from semi_supervised import CrossPseudoSupervision, SemiSupervisedTrainer

network_a = YourSegmentationModel()
network_b = YourSegmentationModel()

framework = CrossPseudoSupervision(
    network_a, network_b,
    num_classes=2,
    config={'cps_weight': 1.0, 'pseudo_threshold': 0.8}
)

trainer = SemiSupervisedTrainer(
    framework,
    labeled_dataloader,
    unlabeled_dataloader,
    val_dataloader,
    config=training_config
)
trainer.train()

# Stage3: 分类 - 使用FixMatch
from semi_supervised import FixMatch

model = YourClassificationModel()
framework = FixMatch(
    model,
    num_classes=4,
    config={'confidence_threshold': 0.95}
)
"""

from .base import (
    EMAModel,
    PseudoLabelGenerator,
    SegmentationPseudoLabelGenerator,
    ClassificationPseudoLabelGenerator,
    DetectionPseudoLabelGenerator,
    ConsistencyLoss,
    RampUpScheduler,
    AdaptiveThreshold,
    SemiSupervisedDataset,
    PseudoLabelQualityMonitor
)

from .teacher_student import (
    TeacherStudentFramework,
    MultiTaskTeacherStudent
)

from .fixmatch import (
    FixMatch,
    FlexMatch,
    EvidentialFixMatch
)

from .cross_pseudo_supervision import (
    CrossPseudoSupervision,
    MeanTeacher
)

from .augmentation import (
    WeakAugmentation3D,
    StrongAugmentation3D,
    CopyPasteAugmentation,
    MixUp3D
)

from .trainer import (
    SemiSupervisedTrainer,
    create_semi_supervised_framework
)


__all__ = [
    # Base
    'EMAModel',
    'PseudoLabelGenerator',
    'SegmentationPseudoLabelGenerator',
    'ClassificationPseudoLabelGenerator',
    'DetectionPseudoLabelGenerator',
    'ConsistencyLoss',
    'RampUpScheduler',
    'AdaptiveThreshold',
    'SemiSupervisedDataset',
    'PseudoLabelQualityMonitor',
    
    # Teacher-Student
    'TeacherStudentFramework',
    'MultiTaskTeacherStudent',
    
    # FixMatch
    'FixMatch',
    'FlexMatch',
    'EvidentialFixMatch',
    
    # CPS
    'CrossPseudoSupervision',
    'MeanTeacher',
    
    # Augmentation
    'WeakAugmentation3D',
    'StrongAugmentation3D',
    'CopyPasteAugmentation',
    'MixUp3D',
    
    # Trainer
    'SemiSupervisedTrainer',
    'create_semi_supervised_framework'
]

__version__ = '1.0.0'
