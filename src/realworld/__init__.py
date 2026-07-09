"""
Real-world multi-lesion data structures and safety utilities.

This package is intentionally dependency-light. It provides the structured
objects used to represent lesion-level predictions, patient-level multi-label
summaries, safety-gate decisions, and report payloads.
"""

from .aggregation import aggregate_patient_summary
from .reporting import build_structured_report, render_markdown_report
from .safety_gate import SafetyGateConfig, evaluate_safety_gate
from .schema import (
    LesionPrediction,
    PatientSummary,
    SafetyGateResult,
    StructuredReport,
)

__all__ = [
    "LesionPrediction",
    "PatientSummary",
    "SafetyGateConfig",
    "SafetyGateResult",
    "StructuredReport",
    "aggregate_patient_summary",
    "build_structured_report",
    "evaluate_safety_gate",
    "render_markdown_report",
]
