"""
Self-supervised Pretraining Module
自监督预训练模块

Supports:
- 3D Masked Autoencoder (MAE)
- 3D Contrastive Learning (SimCLR-style)
"""

from .mae_3d import MaskedAutoEncoder3D, MAEPreTrainer
from .contrastive import ContrastiveLearning3D, NTXentLoss

__all__ = [
    'MaskedAutoEncoder3D',
    'MAEPreTrainer',
    'ContrastiveLearning3D',
    'NTXentLoss',
]
