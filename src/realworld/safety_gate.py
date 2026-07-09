"""
Safety gate for real-world multi-lesion cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .schema import LesionPrediction, PatientSummary, SafetyGateResult


@dataclass(frozen=True)
class SafetyGateConfig:
    confidence_review_threshold: float = 0.65
    uncertainty_review_threshold: float = 0.35
    always_review_malignancy: bool = True
    always_review_ce_ae_coexistence: bool = True
    always_review_echinococcosis_malignancy_coexistence: bool = True
    always_review_uncertain_class: bool = True


def evaluate_safety_gate(
    lesions: Iterable[LesionPrediction],
    summary: PatientSummary,
    config: SafetyGateConfig | None = None,
) -> SafetyGateResult:
    """Evaluate review triggers for a patient/study."""

    config = config or SafetyGateConfig()
    lesion_list = list(lesions)
    reasons: List[str] = []
    rules: List[str] = []

    if not lesion_list:
        _add(reasons, rules, "No lesion was detected or provided.", "no_lesion_detected")

    for lesion in lesion_list:
        if lesion.confidence < config.confidence_review_threshold:
            _add(
                reasons,
                rules,
                f"Lesion {lesion.lesion_id} has low confidence ({lesion.confidence:.2f}).",
                "low_confidence",
            )
        if lesion.uncertainty > config.uncertainty_review_threshold:
            _add(
                reasons,
                rules,
                f"Lesion {lesion.lesion_id} has high uncertainty ({lesion.uncertainty:.2f}).",
                "high_uncertainty",
            )
        if config.always_review_uncertain_class and lesion.is_uncertain:
            _add(
                reasons,
                rules,
                f"Lesion {lesion.lesion_id} is classified as UNCERTAIN.",
                "uncertain_main_class",
            )
        if lesion.flags.get("model_disagreement"):
            _add(
                reasons,
                rules,
                f"Lesion {lesion.lesion_id} has model disagreement.",
                "model_disagreement",
            )
        if lesion.flags.get("segmentation_failure"):
            _add(
                reasons,
                rules,
                f"Lesion {lesion.lesion_id} has segmentation failure flag.",
                "segmentation_failure",
            )

    if config.always_review_malignancy and summary.has_malignant:
        _add(reasons, rules, "At least one malignant lesion is suspected.", "malignant_lesion_detected")

    if config.always_review_ce_ae_coexistence and summary.has_ce and summary.has_ae:
        _add(reasons, rules, "CE and AE lesions coexist in the same patient/study.", "ce_and_ae_coexist")

    if (
        config.always_review_echinococcosis_malignancy_coexistence
        and summary.has_echinococcosis
        and summary.has_malignant
    ):
        _add(
            reasons,
            rules,
            "Echinococcosis and malignant lesion coexist.",
            "echinococcosis_and_malignancy_coexist",
        )

    if summary.has_complex_coexistence:
        _add(reasons, rules, "Multiple clinically distinct lesion groups coexist.", "complex_multilesion_case")

    return SafetyGateResult(
        requires_review=bool(reasons),
        review_reasons=reasons,
        triggered_rules=rules,
        highest_risk_lesion_id=summary.highest_risk_lesion_id,
    )


def _add(reasons: List[str], rules: List[str], reason: str, rule: str) -> None:
    if rule not in rules:
        rules.append(rule)
        reasons.append(reason)
