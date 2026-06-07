#!/usr/bin/env python3
"""
Evaluate exported classification predictions from multiple model candidates.

Directory layout:

    labels.csv
    predictions/
      project_stage3_temporal_evidential/
        predictions.csv
      monai_densenet121_3d/
        predictions.csv

Each predictions.csv should contain:
    case_id,pred,prob_0,prob_1,...
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
        description="Compare classification model predictions with shared metrics."
    )
    parser.add_argument("--list-models", action="store_true", help="Print registered model candidates and exit.")
    parser.add_argument("--labels-csv", help="CSV containing case_id and ground-truth label columns.")
    parser.add_argument("--pred-root", help="Root directory containing model prediction CSV files/folders.")
    parser.add_argument("--output-dir", default="outputs/classification_benchmark", help="Directory for CSV/JSON metrics.")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes for this classification task.")
    parser.add_argument("--class-names", default=None, help="Comma-separated class names in label-index order.")
    parser.add_argument("--case-col", default="case_id", help="Case ID column name.")
    parser.add_argument("--label-col", default="label", help="Ground-truth label column name.")
    parser.add_argument("--pred-col", default="pred", help="Prediction class column name.")
    parser.add_argument("--prob-prefix", default="prob_", help="Probability column prefix, e.g. prob_0.")
    parser.add_argument("--prob-cols", default=None, help="Comma-separated probability columns, overriding --prob-prefix.")
    parser.add_argument("--primary-metric", default="macro_auc_ovr", help="Metric used first for model ranking.")
    parser.add_argument("--tie-breaker", default="macro_f1", help="Metric used second for model ranking.")
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    args = build_parser().parse_args()

    if args.list_models:
        _print_model_catalog()
        return 0

    if not args.labels_csv or not args.pred_root:
        raise SystemExit("--labels-csv and --pred-root are required unless --list-models is used.")

    from src.classification.evaluation import (
        classification_metrics,
        collect_prediction_files,
        csv_ready,
        json_ready,
        load_labels_csv,
        load_predictions_csv,
        rank_model_summaries,
    )

    class_names = _parse_csv_list(args.class_names)
    prob_cols = _parse_csv_list(args.prob_cols)

    labels = load_labels_csv(args.labels_csv, case_col=args.case_col, label_col=args.label_col)
    prediction_files = collect_prediction_files(args.pred_root)
    if not prediction_files:
        raise SystemExit(f"No prediction CSV files found under: {args.pred_root}")

    case_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    skipped: Dict[str, Any] = {}

    for model, prediction_file in prediction_files.items():
        predictions = load_predictions_csv(
            prediction_file,
            num_classes=args.num_classes,
            case_col=args.case_col,
            pred_col=args.pred_col,
            prob_cols=prob_cols,
            prob_prefix=args.prob_prefix,
        )
        common_case_ids = sorted(set(labels) & set(predictions))
        skipped[model] = {
            "missing_predictions": sorted(set(labels) - set(predictions)),
            "extra_predictions": sorted(set(predictions) - set(labels)),
        }
        if not common_case_ids:
            continue

        targets = [labels[case_id] for case_id in common_case_ids]
        preds = [predictions[case_id]["pred"] for case_id in common_case_ids]
        probs = [predictions[case_id]["probs"] for case_id in common_case_ids]
        probs_for_metrics = probs if all(prob is not None for prob in probs) else None

        metrics = classification_metrics(
            targets=targets,
            preds=preds,
            probs=probs_for_metrics,
            num_classes=args.num_classes,
            class_names=class_names,
        )
        summaries.append({"model": model, **metrics})

        for case_id, target, pred, prob in zip(common_case_ids, targets, preds, probs):
            row: Dict[str, Any] = {
                "model": model,
                "case_id": case_id,
                "label": target,
                "pred": pred,
                "correct": int(target == pred),
            }
            if prob is not None:
                for index, value in enumerate(prob):
                    row[f"prob_{index}"] = value
            case_rows.append(row)

    if not summaries:
        raise SystemExit("No matching prediction/label case ids were found.")

    ranked = rank_model_summaries(
        summaries,
        primary_metric=args.primary_metric,
        tie_breaker=args.tie_breaker,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_csv = output_dir / "case_predictions.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    _write_csv(case_csv, case_rows, csv_ready)
    _write_csv(summary_csv, ranked, csv_ready)

    payload = {
        "selection_rule": {
            "primary_metric": args.primary_metric,
            "tie_breaker": args.tie_breaker,
        },
        "num_classes": args.num_classes,
        "class_names": class_names,
        "ranked_models": ranked,
        "skipped_cases": skipped,
    }
    summary_json.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    best = ranked[0]
    print(f"Evaluated {len(case_rows)} matched predictions across {len(ranked)} model files.")
    print(
        "Best model: "
        f"{best['model']} ({args.primary_metric}={best.get(args.primary_metric)}, "
        f"{args.tie_breaker}={best.get(args.tie_breaker)})"
    )
    print(f"Wrote: {case_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")
    return 0


def _print_model_catalog() -> None:
    from src.classification.model_zoo import list_model_specs

    for spec in list_model_specs():
        print(f"{spec.name} [{spec.implementation}; {spec.status}]")
        print(f"  family: {spec.family}")
        print(f"  task: {spec.recommended_task}")
        if spec.references:
            print(f"  references: {', '.join(spec.references)}")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], value_converter) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: value_converter(row.get(name, "")) for name in fieldnames})


def _ordered_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = ["rank", "model", "case_id", "num_cases", "label", "pred", "correct"]
    fieldnames: List[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


def _parse_csv_list(value: str | None) -> List[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
