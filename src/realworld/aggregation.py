"""
Aggregate lesion-level results into patient-level multi-label summaries.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .schema import MAIN_CLASSES, LesionPrediction, PatientSummary


RISK_PRIORITY = {
    "MALIGNANT": 100,
    "AE": 80,
    "CE": 70,
    "UNCERTAIN": 60,
    "BENIGN": 20,
}


def aggregate_patient_summary(
    lesions: Iterable[LesionPrediction],
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
) -> PatientSummary:
    """Create a patient-level summary from lesion predictions."""

    lesion_list = list(lesions)
    if lesion_list and patient_id is None:
        patient_id = lesion_list[0].patient_id
    if lesion_list and study_id is None:
        study_id = lesion_list[0].study_id
    if patient_id is None:
        raise ValueError("patient_id is required when no lesions are provided.")

    class_counts = {name: 0 for name in sorted(MAIN_CLASSES)}
    lesion_ids_by_class = {name: [] for name in sorted(MAIN_CLASSES)}

    highest_risk_lesion_id = None
    highest_risk_score = -1.0
    confidences: List[float] = []
    uncertainties: List[float] = []

    for lesion in lesion_list:
        class_counts[lesion.main_class] += 1
        lesion_ids_by_class[lesion.main_class].append(lesion.lesion_id)
        confidences.append(lesion.confidence)
        uncertainties.append(lesion.uncertainty)

        risk_score = RISK_PRIORITY.get(lesion.main_class, 0) + lesion.confidence
        if risk_score > highest_risk_score:
            highest_risk_score = risk_score
            highest_risk_lesion_id = lesion.lesion_id

    has_ce = class_counts["CE"] > 0
    has_ae = class_counts["AE"] > 0
    has_benign = class_counts["BENIGN"] > 0
    has_malignant = class_counts["MALIGNANT"] > 0
    has_uncertain = class_counts["UNCERTAIN"] > 0
    positive_groups = sum([has_ce, has_ae, has_benign, has_malignant])
    has_complex_coexistence = positive_groups >= 2 or (has_uncertain and len(lesion_list) > 1)

    return PatientSummary(
        patient_id=patient_id,
        study_id=study_id,
        lesion_count=len(lesion_list),
        has_ce=has_ce,
        has_ae=has_ae,
        has_benign=has_benign,
        has_malignant=has_malignant,
        has_uncertain=has_uncertain,
        has_complex_coexistence=has_complex_coexistence,
        highest_risk_lesion_id=highest_risk_lesion_id,
        lesion_ids_by_class=lesion_ids_by_class,
        class_counts=class_counts,
        max_confidence=max(confidences) if confidences else None,
        max_uncertainty=max(uncertainties) if uncertainties else None,
    )
