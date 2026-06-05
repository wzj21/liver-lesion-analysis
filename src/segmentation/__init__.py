"""
Segmentation model comparison utilities.

This package provides a lightweight model zoo and evaluation helpers so the
project can compare several 3D segmentation algorithms under the same data
split and metrics before selecting a production model.
"""

from .model_zoo import (
    SegmentationModelSpec,
    build_segmentation_model,
    get_model_spec,
    list_model_specs,
)

__all__ = [
    "SegmentationModelSpec",
    "build_segmentation_model",
    "get_model_spec",
    "list_model_specs",
]
