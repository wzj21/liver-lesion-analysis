"""
Clinical decision support rules for liver lesion inference results.

The rules here deliberately provide review-oriented recommendations instead of
autonomous diagnosis. They are intended to help a clinician decide what should
be checked next after the model has produced segmentation and classification
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STAGE3_CLASSES: Dict[int, Dict[str, str]] = {
    0: {"name_cn": "良性肝脏病变", "name_en": "benign"},
    1: {"name_cn": "恶性肝脏病变", "name_en": "malignant"},
    2: {"name_cn": "肝囊型包虫病", "name_en": "cystic_echinococcosis"},
    3: {"name_cn": "肝泡型包虫病", "name_en": "alveolar_echinococcosis"},
}


@dataclass
class ClinicalSummary:
    """Compact clinical view derived from raw model outputs."""

    has_lesion: bool
    lesion_count: int
    detection_confidence: float
    diagnosis: str = "未形成分类结论"
    diagnosis_code: Optional[int] = None
    classification_confidence: Optional[float] = None
    classification_uncertainty: Optional[float] = None
    activity_status: Optional[str] = None
    activity_confidence: Optional[float] = None
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_lesion": self.has_lesion,
            "lesion_count": self.lesion_count,
            "detection_confidence": self.detection_confidence,
            "diagnosis": self.diagnosis,
            "diagnosis_code": self.diagnosis_code,
            "classification_confidence": self.classification_confidence,
            "classification_uncertainty": self.classification_uncertainty,
            "activity_status": self.activity_status,
            "activity_confidence": self.activity_confidence,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "recommendations": self.recommendations,
            "next_steps": self.next_steps,
        }


class ClinicalDecisionSupport:
    """Rule-based report and recommendation engine."""

    def __init__(
        self,
        detection_review_threshold: float = 0.70,
        uncertainty_review_threshold: float = 0.50,
    ):
        self.detection_review_threshold = detection_review_threshold
        self.uncertainty_review_threshold = uncertainty_review_threshold

    def summarize(self, results: Dict[str, Any]) -> ClinicalSummary:
        stage2 = results.get("stage2", {})
        stage3 = results.get("stage3", {})
        stage4 = results.get("stage4", {})
        raw_summary = results.get("summary", {})

        has_lesion = bool(stage2.get("has_lesion", raw_summary.get("has_lesion", False)))
        lesion_count = int(stage2.get("num_detections", raw_summary.get("num_lesions", 0)) or 0)
        detection_conf = float(stage2.get("confidence", raw_summary.get("detection_confidence", 0.0)) or 0.0)

        summary = ClinicalSummary(
            has_lesion=has_lesion,
            lesion_count=lesion_count,
            detection_confidence=detection_conf,
        )

        if not has_lesion or lesion_count == 0:
            summary.diagnosis = "未发现明确肝脏病灶"
            summary.recommendations.extend([
                "如临床症状、实验室指标或既往影像提示异常，建议由影像科医生复核原始DICOM。",
                "若为筛查场景，可结合临床风险因素决定是否定期随访。",
            ])
            summary.next_steps.append("保留本次检查作为后续随访基线。")
        else:
            self._fill_classification(summary, stage3)
            self._fill_activity(summary, stage4)
            self._add_condition_specific_recommendations(summary)

        if detection_conf < self.detection_review_threshold:
            summary.needs_review = True
            summary.review_reasons.append(
                f"病灶检测置信度较低({detection_conf:.1%})，需要人工复核。"
            )

        if (
            summary.classification_uncertainty is not None
            and summary.classification_uncertainty > self.uncertainty_review_threshold
        ):
            summary.needs_review = True
            summary.review_reasons.append(
                f"分类不确定性较高({summary.classification_uncertainty:.1%})。"
            )

        if raw_summary.get("needs_review") and raw_summary.get("review_reason"):
            summary.needs_review = True
            reason = str(raw_summary["review_reason"])
            if reason not in summary.review_reasons:
                summary.review_reasons.append(reason)

        if summary.needs_review:
            summary.next_steps.insert(0, "建议影像科/肝胆专科医生复核AI分割、关键切片和原始增强序列。")

        return summary

    def _fill_classification(self, summary: ClinicalSummary, stage3: Dict[str, Any]) -> None:
        pred_class = stage3.get("pred_class")
        if pred_class is None or pred_class == -1:
            summary.diagnosis = "检测到病灶，但分类结果不足"
            summary.recommendations.append("建议补充增强CT/MRI或人工标注关键病灶后重新分析。")
            summary.needs_review = True
            summary.review_reasons.append("模型未能生成可靠分类。")
            return

        pred_class = int(pred_class)
        class_info = STAGE3_CLASSES.get(pred_class, {"name_cn": "未知类型肝脏病灶", "name_en": "unknown"})
        summary.diagnosis = class_info["name_cn"]
        summary.diagnosis_code = pred_class
        summary.classification_uncertainty = float(stage3.get("uncertainty", 0.0) or 0.0)

        probs = stage3.get("probs")
        if probs is not None:
            try:
                summary.classification_confidence = float(probs[pred_class])
            except Exception:
                summary.classification_confidence = None

    def _fill_activity(self, summary: ClinicalSummary, stage4: Dict[str, Any]) -> None:
        if not stage4:
            return

        assessment = stage4.get("assessment", {}) or {}
        summary.activity_status = assessment.get("activity_status_cn")
        if summary.activity_status is None:
            pred = stage4.get("pred")
            if pred is not None:
                summary.activity_status = "活动性" if int(pred) == 0 else "非活动性"

        active_prob = stage4.get("active_prob")
        inactive_prob = stage4.get("inactive_prob")
        if active_prob is not None and inactive_prob is not None:
            summary.activity_confidence = float(max(active_prob, inactive_prob))

    def _add_condition_specific_recommendations(self, summary: ClinicalSummary) -> None:
        diagnosis_code = summary.diagnosis_code

        if diagnosis_code == 0:
            summary.recommendations.extend([
                "良性倾向不等于无需处理，建议结合病灶大小、增强模式和既往影像变化判断。",
                "若病灶增大、形态不典型或患者高危，建议进一步MRI或短期随访。",
            ])
            summary.next_steps.append("将AI结果与既往影像对照，评估病灶是否稳定。")
        elif diagnosis_code == 1:
            summary.recommendations.extend([
                "恶性倾向结果应尽快由影像科和肝胆肿瘤团队复核。",
                "建议结合肿瘤标志物、肝功能、乙肝/丙肝背景和增强MRI进一步评估。",
                "必要时按临床路径考虑MDT会诊和病理确认。",
            ])
            summary.next_steps.append("优先安排专科复诊或MDT评估。")
        elif diagnosis_code == 2:
            summary.recommendations.extend([
                "囊型包虫病倾向时，建议结合流行病学史和包虫病血清学检查。",
                "如判断为活动性或过渡期，应由感染/肝胆外科评估药物、介入或手术方案。",
            ])
            summary.next_steps.append("补充包虫病血清学检查，并评估WHO-IWGE分期。")
        elif diagnosis_code == 3:
            summary.recommendations.extend([
                "泡型包虫病倾向时，建议评估浸润边界、血管胆管受侵和远处转移。",
                "建议结合增强MRI/CT、血清学和肝胆外科评估PNM分期。",
            ])
            summary.next_steps.append("建议肝胆外科或感染病专科进一步评估可切除性和治疗方案。")
        else:
            summary.recommendations.append("分类结果未落入已知类别，建议人工复核。")
            summary.needs_review = True

        if summary.activity_status:
            if "活动" in summary.activity_status and "非" not in summary.activity_status:
                summary.next_steps.append("若临床一致，尽快评估抗寄生虫治疗或外科/介入治疗适应证。")
            elif "非" in summary.activity_status or "静止" in summary.activity_status:
                summary.next_steps.append("若病灶稳定且无症状，可由医生决定随访间隔。")
