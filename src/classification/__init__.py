"""
Classification model comparison utilities.

This package mirrors the segmentation benchmark workflow: register candidate
classification algorithms, evaluate exported predictions with one metric suite,
and rank models before selecting a production classifier.
"""

from .model_zoo import (
    ClassificationModelSpec,
    build_classification_model,
    get_model_spec,
    list_model_specs,
)

__all__ = [
    "ClassificationModelSpec",
    "build_classification_model",
    "get_model_spec",
    "list_model_specs",
]
