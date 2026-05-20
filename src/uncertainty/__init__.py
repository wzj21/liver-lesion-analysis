"""
Uncertainty Estimation Module
不确定性估计模块
"""

from .evidential_layers import EvidentialSegmentationHead, EvidentialClassificationHead
from .uncertainty_calibration import TemperatureScaling, calibrate_model
from .uncertainty_metrics import expected_calibration_error, compute_uncertainty_metrics

__all__ = [
    'EvidentialSegmentationHead',
    'EvidentialClassificationHead',
    'TemperatureScaling',
    'calibrate_model',
    'expected_calibration_error',
    'compute_uncertainty_metrics',
]
