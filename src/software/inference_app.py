"""
Application-facing inference workflow.

This module is intentionally separate from the research models. It provides a
stable software contract:

1. Accept a DICOM series folder, a single DICOM file, or a NIfTI volume.
2. Run the existing multi-stage analysis pipeline.
3. Export segmentation masks and compact JSON results.
4. Produce a clinician-facing report with optional LLM wording support.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .clinical_rules import ClinicalDecisionSupport, ClinicalSummary
from .llm_advisor import LLMAdvisor


@dataclass
class SoftwareInferenceConfig:
    config_path: Optional[str] = None
    device: str = "cuda"
    require_checkpoints: bool = True
    stage1_path: Optional[str] = None
    stage2_path: Optional[str] = None
    stage3_path: Optional[str] = None
    stage4_path: Optional[str] = None
    export_masks: bool = True
    llm: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "SoftwareInferenceConfig":
        if not path:
            return cls()

        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except ImportError:
            raw = _load_simple_yaml(path)

        checkpoints = raw.get("checkpoints", {}) or {}
        output = raw.get("output", {}) or {}
        base_path = Path(path).resolve()
        pipeline_config = _resolve_config_path(raw.get("pipeline_config"), base_path)
        return cls(
            config_path=pipeline_config,
            device=raw.get("device", "cuda"),
            require_checkpoints=bool(raw.get("require_checkpoints", True)),
            stage1_path=_resolve_config_path(checkpoints.get("stage1"), base_path),
            stage2_path=_resolve_config_path(checkpoints.get("stage2"), base_path),
            stage3_path=_resolve_config_path(checkpoints.get("stage3"), base_path),
            stage4_path=_resolve_config_path(checkpoints.get("stage4"), base_path),
            export_masks=bool(output.get("export_masks", True)),
            llm=raw.get("llm", {}) or {},
        )


def _load_simple_yaml(path: str) -> Dict[str, Any]:
    """Small YAML subset parser for software_inference.yaml.

    This keeps the CLI configuration readable in minimal environments where
    PyYAML has not been installed yet. It supports nested dictionaries,
    strings, booleans, ints, floats, and empty values.
    """
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            line = line.split("#", 1)[0].rstrip()
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in line:
                continue

            key, value = line.strip().split(":", 1)
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]

            if value == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _coerce_simple_yaml_value(value)

    return root


def _resolve_config_path(value: Optional[str], config_path: Path) -> Optional[str]:
    if not value:
        return None

    path = Path(str(value))
    if path.is_absolute():
        return str(path)

    candidates = [
        config_path.parent / path,
        config_path.parent.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Prefer project-root relative paths for clear missing-checkpoint errors.
    return str((config_path.parent.parent / path).resolve())


def _coerce_simple_yaml_value(value: str) -> Any:
    value = value.strip()
    if value in {"''", '""'}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


class LiverLesionSoftware:
    """High-level software wrapper around ``LiverLesionAnalysisPipeline``."""

    def __init__(self, config: SoftwareInferenceConfig):
        self.config = config
        self.decision_support = ClinicalDecisionSupport()
        self.llm_advisor = LLMAdvisor.from_dict(config.llm)
        self.pipeline = None
        self.loaded_weights = self._resolve_weight_status()
        self._validate_weight_policy()

    @classmethod
    def from_config_file(cls, config_path: Optional[str]) -> "LiverLesionSoftware":
        return cls(SoftwareInferenceConfig.from_yaml(config_path))

    def analyze(
        self,
        input_path: str,
        output_dir: str,
        patient_id: Optional[str] = None,
        spacing: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """Run the full software workflow."""
        started = time.time()
        case_id = patient_id or Path(input_path).stem.replace(".nii", "") or str(uuid.uuid4())[:8]
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        self._ensure_pipeline()

        raw_results = self.pipeline.analyze(
            ct_path=input_path,
            spacing=spacing,
            return_intermediate=True,
        )
        clinical_summary = self.decision_support.summarize(raw_results)

        mask_paths: Dict[str, Any] = {}
        if self.config.export_masks:
            mask_paths = self._export_masks(raw_results, output_root)

        compact_results = self._compact_results(
            raw_results=raw_results,
            clinical_summary=clinical_summary,
            case_id=case_id,
            input_path=input_path,
            output_root=output_root,
            mask_paths=mask_paths,
            processing_time_seconds=time.time() - started,
        )

        llm_advice = self.llm_advisor.generate_advice(compact_results)
        if llm_advice:
            compact_results["llm_advice"] = llm_advice

        result_json = output_root / "result.json"
        report_md = output_root / "report.md"
        report_html = output_root / "report.html"

        self._write_json(result_json, compact_results)
        report_text = self._render_markdown_report(compact_results)
        report_md.write_text(report_text, encoding="utf-8")
        report_html.write_text(self._render_html_report(report_text), encoding="utf-8")

        compact_results["outputs"] = {
            "result_json": str(result_json),
            "report_md": str(report_md),
            "report_html": str(report_html),
            "masks": mask_paths,
        }
        self._write_json(result_json, compact_results)
        return compact_results

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return

        from ..pipeline import LiverLesionAnalysisPipeline

        self.pipeline = LiverLesionAnalysisPipeline(
            config_path=self.config.config_path,
            device=self.config.device,
        )
        self.pipeline.load_weights(
            stage1_path=self.config.stage1_path,
            stage2_path=self.config.stage2_path,
            stage3_path=self.config.stage3_path,
            stage4_path=self.config.stage4_path,
        )

    def _resolve_weight_status(self) -> Dict[str, Dict[str, Any]]:
        paths = {
            "stage1": self.config.stage1_path,
            "stage2": self.config.stage2_path,
            "stage3": self.config.stage3_path,
            "stage4": self.config.stage4_path,
        }
        return {
            name: {
                "path": path,
                "exists": bool(path and Path(path).exists()),
            }
            for name, path in paths.items()
        }

    def _validate_weight_policy(self) -> None:
        if not self.config.require_checkpoints:
            return
        missing = [name for name, status in self.loaded_weights.items() if not status["exists"]]
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                "软件推理默认要求加载训练好的四阶段权重，缺少: "
                f"{missing_text}。如果只是做流程调试，可在配置中设置 "
                "require_checkpoints: false。"
            )

    def _export_masks(self, results: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
        masks_dir = output_root / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        spacing = tuple(results.get("preprocessing", {}).get("target_spacing", (1.0, 1.0, 1.0)))
        exported: Dict[str, Any] = {}

        liver_mask = results.get("stage1", {}).get("liver_mask")
        if liver_mask is not None:
            exported["liver_mask"] = self._save_volume(
                liver_mask.astype(np.uint8),
                masks_dir / "liver_mask_preprocessed.nii.gz",
                spacing,
            )

        lesion_masks = results.get("stage2", {}).get("lesion_masks")
        if lesion_masks is not None:
            lesion_paths: List[str] = []
            combined = np.zeros(lesion_masks.shape[1:], dtype=np.uint8)
            for idx, mask in enumerate(lesion_masks):
                binary = (mask > 0.5).astype(np.uint8)
                combined = np.maximum(combined, binary * (idx + 1))
                lesion_paths.append(
                    self._save_volume(
                        binary,
                        masks_dir / f"lesion_{idx + 1:02d}_mask_preprocessed.nii.gz",
                        spacing,
                    )
                )
            exported["lesion_masks"] = lesion_paths
            exported["lesion_mask_combined"] = self._save_volume(
                combined,
                masks_dir / "lesion_mask_combined_preprocessed.nii.gz",
                spacing,
            )

        return exported

    @staticmethod
    def _save_volume(volume: np.ndarray, output_path: Path, spacing: tuple) -> str:
        try:
            import nibabel as nib

            affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
            nii = nib.Nifti1Image(volume.astype(np.float32), affine)
            nib.save(nii, str(output_path))
            return str(output_path)
        except Exception:
            fallback = output_path.with_suffix("").with_suffix(".npy")
            np.save(str(fallback), volume)
            return str(fallback)

    def _compact_results(
        self,
        raw_results: Dict[str, Any],
        clinical_summary: ClinicalSummary,
        case_id: str,
        input_path: str,
        output_root: Path,
        mask_paths: Dict[str, Any],
        processing_time_seconds: float,
    ) -> Dict[str, Any]:
        stage2 = raw_results.get("stage2", {})
        stage3 = raw_results.get("stage3", {})
        stage4 = raw_results.get("stage4", {})
        lesion_measurements = self._measure_lesions(raw_results)

        return {
            "case": {
                "case_id": case_id,
                "input_path": str(Path(input_path).resolve()),
                "output_dir": str(output_root),
                "analysis_time": datetime.now().isoformat(timespec="seconds"),
                "processing_time_seconds": round(processing_time_seconds, 3),
            },
            "model_status": {
                "weights": self.loaded_weights,
                "llm_advisor_enabled": self.llm_advisor.config.enabled,
                "llm_advisor_available": self.llm_advisor.available(),
            },
            "preprocessing": self._json_safe(raw_results.get("preprocessing", {})),
            "segmentation": {
                "has_liver_mask": raw_results.get("stage1", {}).get("liver_mask") is not None,
                "has_lesion_masks": stage2.get("lesion_masks") is not None,
                "mask_paths": mask_paths,
                "lesion_measurements": lesion_measurements,
            },
            "detection": {
                "has_lesion": bool(stage2.get("has_lesion", False)),
                "confidence": float(stage2.get("confidence", 0.0) or 0.0),
                "num_detections": int(stage2.get("num_detections", 0) or 0),
                "boxes": self._json_safe(stage2.get("lesion_boxes")),
            },
            "classification": {
                "diagnosis": clinical_summary.diagnosis,
                "diagnosis_code": clinical_summary.diagnosis_code,
                "confidence": clinical_summary.classification_confidence,
                "uncertainty": clinical_summary.classification_uncertainty,
                "raw_stage3": self._json_safe({
                    "pred_class": stage3.get("pred_class"),
                    "class_name_cn": stage3.get("class_name_cn"),
                    "class_name_en": stage3.get("class_name_en"),
                    "probs": stage3.get("probs"),
                    "selected_slices": stage3.get("selected_slices"),
                }),
            },
            "activity": self._json_safe({
                "activity_status": clinical_summary.activity_status,
                "confidence": clinical_summary.activity_confidence,
                "raw_stage4": {
                    "pred": stage4.get("pred"),
                    "active_prob": stage4.get("active_prob"),
                    "inactive_prob": stage4.get("inactive_prob"),
                    "uncertainty": stage4.get("uncertainty"),
                    "assessment": stage4.get("assessment"),
                },
            }),
            "clinical_summary": clinical_summary.to_dict(),
            "safety": {
                "intended_use": "AI辅助影像分析，不能替代医生诊断。",
                "needs_review": clinical_summary.needs_review,
                "review_reasons": clinical_summary.review_reasons,
            },
        }

    @staticmethod
    def _measure_lesions(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        lesion_masks = results.get("stage2", {}).get("lesion_masks")
        if lesion_masks is None:
            return []

        spacing = tuple(results.get("preprocessing", {}).get("target_spacing", (1.0, 1.0, 1.0)))
        voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
        measurements = []
        for idx, mask in enumerate(lesion_masks):
            binary = mask > 0.5
            coords = np.argwhere(binary)
            if coords.size == 0:
                continue
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            size_voxels = maxs - mins + 1
            size_mm = [
                float(size_voxels[0] * spacing[0]),
                float(size_voxels[1] * spacing[1]),
                float(size_voxels[2] * spacing[2]),
            ]
            measurements.append({
                "id": f"L{idx + 1:02d}",
                "voxel_count": int(binary.sum()),
                "volume_mm3": float(binary.sum() * voxel_volume),
                "bbox_zyx": [int(x) for x in [*mins, *maxs]],
                "size_mm_zyx": size_mm,
            })
        return measurements

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size > 1000:
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "note": "array omitted from JSON; see exported mask files when available",
                }
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _render_markdown_report(result: Dict[str, Any]) -> str:
        case = result["case"]
        summary = result["clinical_summary"]
        detection = result["detection"]
        classification = result["classification"]
        activity = result["activity"]
        safety = result["safety"]

        lines = [
            "# 肝脏病灶AI辅助分析报告",
            "",
            f"- 病例ID: {case['case_id']}",
            f"- 分析时间: {case['analysis_time']}",
            f"- 输入数据: {case['input_path']}",
            "",
            "## AI结论摘要",
            "",
            f"- 是否检测到病灶: {'是' if detection['has_lesion'] else '否'}",
            f"- 病灶数量: {detection['num_detections']}",
            f"- 检测置信度: {detection['confidence']:.1%}",
            f"- 分类结果: {classification['diagnosis']}",
        ]

        if classification.get("confidence") is not None:
            lines.append(f"- 分类置信度: {classification['confidence']:.1%}")
        if classification.get("uncertainty") is not None:
            lines.append(f"- 分类不确定性: {classification['uncertainty']:.1%}")
        if activity.get("activity_status"):
            lines.append(f"- 活性判断: {activity['activity_status']}")

        lines.extend(["", "## 复核提示", ""])
        if safety["needs_review"]:
            for reason in safety["review_reasons"]:
                lines.append(f"- {reason}")
        else:
            lines.append("- 当前规则未触发强制复核阈值，但仍建议医生结合原始影像和临床资料判断。")

        lines.extend(["", "## 建议", ""])
        for item in summary.get("recommendations", []):
            lines.append(f"- {item}")

        lines.extend(["", "## 下一步", ""])
        for item in summary.get("next_steps", []):
            lines.append(f"- {item}")

        if result.get("llm_advice"):
            advice = result["llm_advice"]
            lines.extend(["", "## 大模型报告建议", ""])
            if advice.get("impression"):
                lines.append(f"**印象:** {advice['impression']}")
            for key, title in [("risk_notes", "风险提示"), ("next_steps", "建议动作")]:
                values = advice.get(key) or []
                if isinstance(values, str):
                    values = [values]
                if values:
                    lines.append("")
                    lines.append(f"**{title}:**")
                    for value in values:
                        lines.append(f"- {value}")
            if advice.get("disclaimer"):
                lines.append("")
                lines.append(f"**声明:** {advice['disclaimer']}")

        lines.extend([
            "",
            "## 软件声明",
            "",
            "本报告由AI辅助系统生成，仅供临床参考，不能替代影像科医生、专科医生或病理诊断。",
            "当AI结果与临床表现、实验室检查或医生判断不一致时，应以医生综合判断为准。",
        ])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_html_report(markdown_text: str) -> str:
        import html

        body = html.escape(markdown_text)
        body = body.replace("\n", "<br>\n")
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>肝脏病灶AI辅助分析报告</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      margin: 32px auto;
      max-width: 960px;
      line-height: 1.65;
      color: #1f2933;
    }}
    .report {{
      border: 1px solid #d8dee9;
      padding: 28px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <div class="report">{body}</div>
</body>
</html>
"""
