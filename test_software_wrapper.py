"""
Lightweight tests for the software-facing layer.

These tests intentionally avoid loading torch or the neural-network pipeline so
they can run in minimal environments.
"""

from src.software.clinical_rules import ClinicalDecisionSupport
from src.software.llm_advisor import LLMAdvisor


def test_clinical_rules_malignant_case_needs_action():
    results = {
        "stage2": {"has_lesion": True, "confidence": 0.91, "num_detections": 1},
        "stage3": {
            "pred_class": 1,
            "probs": [0.02, 0.93, 0.03, 0.02],
            "uncertainty": 0.12,
        },
        "summary": {"needs_review": False},
    }

    summary = ClinicalDecisionSupport().summarize(results)

    assert summary.diagnosis == "恶性肝脏病变"
    assert summary.diagnosis_code == 1
    assert any("MDT" in item or "肝胆肿瘤" in item for item in summary.recommendations)


def test_clinical_rules_low_confidence_triggers_review():
    results = {
        "stage2": {"has_lesion": True, "confidence": 0.40, "num_detections": 1},
        "stage3": {
            "pred_class": 0,
            "probs": [0.70, 0.10, 0.10, 0.10],
            "uncertainty": 0.20,
        },
    }

    summary = ClinicalDecisionSupport().summarize(results)

    assert summary.needs_review is True
    assert any("置信度较低" in reason for reason in summary.review_reasons)


def test_llm_json_block_parser():
    parsed = LLMAdvisor._parse_json_content(
        """```json
{"impression": "良性倾向", "risk_notes": ["需复核"], "next_steps": ["随访"], "disclaimer": "仅供参考"}
```"""
    )

    assert parsed is not None
    assert parsed["impression"] == "良性倾向"


def _run_without_pytest():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


if __name__ == "__main__":
    _run_without_pytest()
