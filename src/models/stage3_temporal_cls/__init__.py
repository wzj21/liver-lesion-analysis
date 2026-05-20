"""
Stage3: Temporal-Enhanced Lesion Classification
Stage3: 时序增强的占位疾病分类

Multi-slice temporal aggregation for 4-class classification:
- 良性 (Benign)
- 恶性 (Malignant)  
- 肝囊型包虫病 (Cystic Echinococcosis, CE)
- 肝泡型包虫病 (Alveolar Echinococcosis, AE)
"""

from .mask_guided_encoder import MaskGuidedEncoder
from .morphology_encoder import MorphologyEncoder
from .temporal_encoder import TemporalEncoder, TemporalAggregator
from .evidential_classifier import EvidentialClassifier
from .temporal_classifier import TemporalLesionClassifier

__all__ = [
    'MaskGuidedEncoder',
    'MorphologyEncoder', 
    'TemporalEncoder',
    'TemporalAggregator',
    'EvidentialClassifier',
    'TemporalLesionClassifier',
]
