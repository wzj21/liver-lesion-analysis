"""
Dependency-light tests for the real-world multi-lesion layer.
"""

from src.realworld import (
    LesionPrediction,
    aggregate_patient_summary,
    build_structured_report,
    evaluate_safety_gate,
    render_markdown_report,
)


def _sample_lesions():
    return [
        LesionPrediction(
            lesion_id="L1",
            patient_id="P1",
            study_id="S1",
            main_class="CE",
            subtype="cystic_echinococcosis",
            ce_stage="CE2",
            confidence=0.91,
            uncertainty=0.08,
        ),
        LesionPrediction(
            lesion_id="L2",
            patient_id="P1",
            study_id="S1",
            main_class="MALIGNANT",
            subtype="suspected_HCC",
            confidence=0.82,
            uncertainty=0.18,
        ),
    ]


def test_patient_summary_detects_complex_coexistence():
    summary = aggregate_patient_summary(_sample_lesions())

    assert summary.has_ce is True
    assert summary.has_malignant is True
    assert summary.has_complex_coexistence is True
    assert summary.highest_risk_lesion_id == "L2"


def test_safety_gate_reviews_echinococcosis_with_malignancy():
    lesions = _sample_lesions()
    summary = aggregate_patient_summary(lesions)
    safety = evaluate_safety_gate(lesions, summary)

    assert safety.requires_review is True
    assert "echinococcosis_and_malignancy_coexist" in safety.triggered_rules
    assert "malignant_lesion_detected" in safety.triggered_rules


def test_structured_report_renders_markdown():
    report = build_structured_report(_sample_lesions())
    markdown = render_markdown_report(report)

    assert report.patient_summary.has_ce is True
    assert "Real-world Liver Lesion AI Report" in markdown
    assert "L1" in markdown
    assert "L2" in markdown


def _run_without_pytest():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


if __name__ == "__main__":
    _run_without_pytest()
