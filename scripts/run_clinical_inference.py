#!/usr/bin/env python3
"""
Clinical software inference entrypoint.

Example:
    python scripts/run_clinical_inference.py \
        --input D:/ct_cases/case_001_dicom \
        --output D:/liver_outputs/case_001 \
        --config configs/software_inference.yaml \
        --patient_id case_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run liver lesion clinical inference from DICOM/NIfTI input."
    )
    parser.add_argument("--input", required=True, help="DICOM folder, DICOM file, or NIfTI file.")
    parser.add_argument("--output", required=True, help="Output directory for masks and reports.")
    parser.add_argument("--config", default="configs/software_inference.yaml", help="Software inference config.")
    parser.add_argument("--patient_id", default=None, help="Optional patient/case ID.")
    parser.add_argument("--allow_missing_weights", action="store_true", help="Allow random/unloaded weights for smoke tests only.")
    parser.add_argument("--stage1", default=None, help="Override Stage1 checkpoint path.")
    parser.add_argument("--stage2", default=None, help="Override Stage2 checkpoint path.")
    parser.add_argument("--stage3", default=None, help="Override Stage3 checkpoint path.")
    parser.add_argument("--stage4", default=None, help="Override Stage4 checkpoint path.")
    parser.add_argument("--enable_llm", action="store_true", help="Enable optional LLM report advisor from config/env.")
    parser.add_argument("--llm_model", default=None, help="Override LLM model name.")
    parser.add_argument("--llm_api_base", default=None, help="Override OpenAI-compatible API base URL.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.software.inference_app import LiverLesionSoftware, SoftwareInferenceConfig

    config_path = str((project_root / args.config).resolve()) if args.config else None
    config = SoftwareInferenceConfig.from_yaml(config_path)

    if args.allow_missing_weights:
        config.require_checkpoints = False
    for name in ["stage1", "stage2", "stage3", "stage4"]:
        override = getattr(args, name)
        if override:
            setattr(config, f"{name}_path", override)

    if args.enable_llm:
        config.llm["enabled"] = True
    if args.llm_model:
        config.llm["model"] = args.llm_model
    if args.llm_api_base:
        config.llm["api_base"] = args.llm_api_base

    app = LiverLesionSoftware(config)
    result = app.analyze(
        input_path=args.input,
        output_dir=args.output,
        patient_id=args.patient_id,
    )

    outputs = result.get("outputs", {})
    print("分析完成")
    print(f"  JSON: {outputs.get('result_json')}")
    print(f"  Markdown报告: {outputs.get('report_md')}")
    print(f"  HTML报告: {outputs.get('report_html')}")
    if outputs.get("masks"):
        print(f"  分割文件: {outputs.get('masks')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
