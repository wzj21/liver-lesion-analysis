"""
Stage4: Echinococcosis Activity Classification
Stage4: 包虫病活性判定

Binary classification for echinococcosis activity status:
- 活动性 (Active): Requires treatment
- 非活动性 (Inactive): Calcified/stable, observation only

Supports both:
- 肝囊型包虫病 (Cystic Echinococcosis, CE)
- 肝泡型包虫病 (Alveolar Echinococcosis, AE)
"""

from .boundary_analyzer import BoundaryFeatureExtractor
from .internal_analyzer import InternalStructureAnalyzer
from .activity_classifier import ActivityClassifier, EchinococcosisActivityNet

__all__ = [
    'BoundaryFeatureExtractor',
    'InternalStructureAnalyzer',
    'ActivityClassifier',
    'EchinococcosisActivityNet',
]
