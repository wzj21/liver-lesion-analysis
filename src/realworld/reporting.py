"""
Structured report helpers for real-world multi-lesion outputs.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .aggregation import aggregate_patient_summary
from .safety_gate import SafetyGateConfig, evaluate_safety_gate
from .schema import LesionPrediction, StructuredReport


def build_structured_report(
    lesions: Iterable[LesionPrediction],
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
    safety_config: SafetyGateConfig | None = None,
) -> StructuredReport:
    """Build a structured report from lesion predictions."""

    lesion_list = list(lesions)
    summary = aggregate_patient_summary(lesion_list, patient_id=patient_id, study_id=study_id)
    safety = evaluate_safety_gate(lesion_list, summary, config=safety_config)
    recommendations = _recommendations(summary, safety.triggered_rules)
    return StructuredReport(
        patient_id=summary.patient_id,
        study_id=summary.study_id,
        lesions=lesion_list,
        patient_summary=summary,
        safety_gate=safety,
        recommendations=recommendations,
    )


def render_markdown_report(report: StructuredReport) -> str:
    """Render a compact Markdown report."""

    lines: List[str] = [
        "# Real-world Liver Lesion AI Report",
        "",
        f"- Patient ID: {report.patient_id}",
        f"- Study ID: {report.study_id or 'N/A'}",
        f"- Lesion count: {report.patient_summary.lesion_count}",
        f"- Requires review: {'yes' if report.safety_gate.requires_review else 'no'}",
        "",
        "## Patient-level Summary",
        "",
        f"- CE: {report.patient_summary.has_ce}",
        f"- AE: {report.patient_summary.has_ae}",
        f"- Benign lesion: {report.patient_summary.has_benign}",
        f"- Malignant lesion: {report.patient_summary.has_malignant}",
        f"- Complex coexistence: {report.patient_summary.has_complex_coexistence}",
        f"- Highest-risk lesion: {report.patient_summary.highest_risk_lesion_id or 'N/A'}",
        "",
        "## Lesions",
        "",
    ]

    for lesion in report.lesions:
        subtype = f" / {lesion.subtype}" if lesion.subtype else ""
        lines.append(
            f"- {lesion.lesion_id}: {lesion.main_class}{subtype}, "
            f"confidence={lesion.confidence:.2f}, uncertainty={lesion.uncertainty:.2f}"
        )

    if report.safety_gate.review_reasons:
        lines.extend(["", "## Review Reasons", ""])
        for reason in report.safety_gate.review_reasons:
            lines.append(f"- {reason}")

    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for recommendation in report.recommendations:
            lines.append(f"- {recommendation}")

    lines.extend(["", "## Disclaimer", "", report.disclaimer])
    return "\n".join(lines) + "\n"


def _recommendations(summary, triggered_rules: List[str]) -> List[str]:
    recommendations: List[str] = []
    if summary.has_malignant:
        recommendations.append("Prioritize clinician review for suspected malignant lesion.")
    if summary.has_ce:
        recommendations.append("Review CE stage and activity using curated imaging and clinical evidence.")
    if summary.has_ae:
        recommendations.append("Review AE activity and invasion risk; PET/CT or MRI-DWI evidence may be needed.")
    if "ce_and_ae_coexist" in triggered_rules:
        recommendations.append("CE and AE coexistence should be confirmed by multidisciplinary review.")
    if "echinococcosis_and_malignancy_coexist" in triggered_rules:
        recommendations.append("Coexisting echinococcosis and malignancy requires urgent expert review.")
    if not recommendations:
        recommendations.append("Review AI output in the clinical context before decision-making.")
    return recommendations
