"""
Structured objects for real-world multi-lesion analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


MAIN_CLASSES = {"CE", "AE", "BENIGN", "MALIGNANT", "UNCERTAIN"}


@dataclass
class LesionPrediction:
    """One lesion-level prediction or curated label."""

    lesion_id: str
    patient_id: str
    study_id: str
    main_class: str
    confidence: float
    uncertainty: float = 0.0
    subtype: Optional[str] = None
    ce_stage: Optional[str] = None
    ce_activity: Optional[str] = None
    ae_activity: Optional[str] = None
    ae_p_stage: Optional[str] = None
    kodama_type: Optional[str] = None
    location_segment: Optional[str] = None
    max_diameter_mm: Optional[float] = None
    volume_ml: Optional[float] = None
    bbox_path: Optional[str] = None
    mask_path: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    evidence_level: str = "unknown"
    label_source: str = "model"
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.main_class = self.main_class.upper()
        if self.main_class not in MAIN_CLASSES:
            raise ValueError(f"Invalid main_class {self.main_class!r}. Expected one of {sorted(MAIN_CLASSES)}")
        self.confidence = _clip01(self.confidence)
        self.uncertainty = _clip01(self.uncertainty)

    @property
    def is_echinococcosis(self) -> bool:
        return self.main_class in {"CE", "AE"}

    @property
    def is_malignant(self) -> bool:
        return self.main_class == "MALIGNANT"

    @property
    def is_uncertain(self) -> bool:
        return self.main_class == "UNCERTAIN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatientSummary:
    """Patient-level multi-label summary derived from lesions."""

    patient_id: str
    study_id: Optional[str] = None
    lesion_count: int = 0
    has_ce: bool = False
    has_ae: bool = False
    has_benign: bool = False
    has_malignant: bool = False
    has_uncertain: bool = False
    has_complex_coexistence: bool = False
    highest_risk_lesion_id: Optional[str] = None
    lesion_ids_by_class: Dict[str, List[str]] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    max_confidence: Optional[float] = None
    max_uncertainty: Optional[float] = None

    @property
    def has_echinococcosis(self) -> bool:
        return self.has_ce or self.has_ae

    @property
    def has_non_echinococcal_lesion(self) -> bool:
        return self.has_benign or self.has_malignant

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyGateResult:
    """Safety-gate decision for one patient/study."""

    requires_review: bool
    review_reasons: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    highest_risk_lesion_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredReport:
    """Structured report payload ready for JSON or Markdown export."""

    patient_id: str
    study_id: Optional[str]
    lesions: List[LesionPrediction]
    patient_summary: PatientSummary
    safety_gate: SafetyGateResult
    recommendations: List[str] = field(default_factory=list)
    disclaimer: str = (
        "AI output is for clinical decision support only and must be reviewed by qualified clinicians."
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["lesions"] = [lesion.to_dict() for lesion in self.lesions]
        payload["patient_summary"] = self.patient_summary.to_dict()
        payload["safety_gate"] = self.safety_gate.to_dict()
        return payload


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
