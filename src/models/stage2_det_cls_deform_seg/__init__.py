"""
Stage2: Lesion Detection, Classification, and Deformable Segmentation
Stage2: 病灶检测分类与Deformable Attention分割
"""

from .lesion_detector import LesionDetector, LesionDetClsDeformSegNet

__all__ = [
    'LesionDetector',
    'LesionDetClsDeformSegNet',
]
