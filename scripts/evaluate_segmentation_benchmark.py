#!/usr/bin/env python3
"""
Evaluate exported 3D segmentation masks from multiple model candidates.

Directory layout:

    predictions/
      monai_segresnet/
        case_001.nii.gz
      external_nnunet/
        case_001.nii.gz
    ground_truth/
      case_001.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare segmentation model predictions with shared metrics."
    )
    parser.add_argument("--list-models", action="store_true", help="Print registered model candidates and exit.")
    parser.add_argument("--gt-dir", help="Directory containing ground-truth masks.")
    parser.add_argument("--pred-root", help="Root directory containing one prediction folder per model.")
    parser.add_argument("--output-dir", default="outputs/segmentation_benchmark", help="Directory for CSV/JSON metrics.")
    parser.add_argument("--label", type=int, default=None, help="Evaluate one label value; default evaluates all >0 as foreground.")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"), help="Voxel spacing in z y x order for HD95.")
    parser.add_argument("--gt-key", default=None, help="Array key for NPZ ground-truth masks.")
    parser.add_argument("--pred-key", default=None, help="Array key for NPZ prediction masks.")
    parser.add_argument("--no-hd95", action="store_true", help="Skip HD95 computation when scipy is unavailable or too slow.")
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    args = build_parser().parse_args()

    if args.list_models:
        _print_model_catalog()
        return 0

    if not args.gt_dir or not args.pred_root:
        raise SystemExit("--gt-dir and --pred-root are required unless --list-models is used.")

    from src.segmentation.evaluation import (
        collect_mask_paths,
        json_ready,
        load_mask,
        rank_model_summaries,
        segmentation_metrics,
        summarize_case_metrics,
    )

    gt_paths = collect_mask_paths(args.gt_dir)
    pred_root = Path(args.pred_root)
    if not pred_root.exists() or not pred_root.is_dir():
        raise SystemExit(f"Prediction root does not exist or is not a directory: {pred_root}")

    model_dirs = [path for path in sorted(pred_root.iterdir()) if path.is_dir()]
    if not model_dirs:
        raise SystemExit(f"No model prediction folders found under: {pred_root}")

    case_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, Any] = {}

    for model_dir in model_dirs:
        pred_paths = collect_mask_paths(model_dir)
        common_case_ids = sorted(set(gt_paths) & set(pred_paths))
        skipped[model_dir.name] = {
            "missing_predictions": sorted(set(gt_paths) - set(pred_paths)),
            "extra_predictions": sorted(set(pred_paths) - set(gt_paths)),
        }

        for case_id in common_case_ids:
            target = load_mask(gt_paths[case_id], key=args.gt_key)
            pred = load_mask(pred_paths[case_id], key=args.pred_key)
            metrics = segmentation_metrics(
                pred=pred,
                target=target,
                spacing=tuple(args.spacing),
                label=args.label,
                compute_hd95=not args.no_hd95,
            )
            case_rows.append({"model": model_dir.name, "case_id": case_id, **metrics})

    if not case_rows:
        raise SystemExit("No matching prediction/ground-truth case ids were found.")

    summaries = summarize_case_metrics(case_rows)
    ranked = rank_model_summaries(summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_csv = output_dir / "case_metrics.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    _write_csv(case_csv, case_rows)
    _write_csv(summary_csv, ranked)

    payload = {
        "selection_rule": {
            "primary": "highest mean_dice",
            "tie_breaker": "lowest mean_hd95",
        },
        "spacing_zyx": list(args.spacing),
        "label": args.label,
        "ranked_models": ranked,
        "skipped_cases": skipped,
    }
    summary_json.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    best = ranked[0]
    print(f"Evaluated {len(case_rows)} matched cases across {len(model_dirs)} model folders.")
    print(f"Best model: {best['model']} (mean_dice={best.get('mean_dice')}, mean_hd95={best.get('mean_hd95')})")
    print(f"Wrote: {case_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")
    return 0


def _print_model_catalog() -> None:
    from src.segmentation.model_zoo import list_model_specs

    for spec in list_model_specs():
        print(f"{spec.name} [{spec.implementation}; {spec.status}]")
        print(f"  family: {spec.family}")
        print(f"  task: {spec.recommended_task}")
        if spec.references:
            print(f"  references: {', '.join(spec.references)}")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ordered_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = ["rank", "model", "case_id", "num_cases"]
    fieldnames: List[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


if __name__ == "__main__":
    raise SystemExit(main())
