"""
Clinical software wrapper for the liver lesion analysis system.

This package turns the research pipeline into a safer application-facing
workflow: DICOM/NIfTI input, model inference, mask export, structured results,
and clinician-oriented report generation.
"""

__all__ = [
    "LiverLesionSoftware",
    "ClinicalDecisionSupport",
    "LLMAdvisor",
]


def __getattr__(name):
    """Lazy-load software components to keep lightweight imports cheap."""
    if name == "LiverLesionSoftware":
        from .inference_app import LiverLesionSoftware

        return LiverLesionSoftware
    if name == "ClinicalDecisionSupport":
        from .clinical_rules import ClinicalDecisionSupport

        return ClinicalDecisionSupport
    if name == "LLMAdvisor":
        from .llm_advisor import LLMAdvisor

        return LLMAdvisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
