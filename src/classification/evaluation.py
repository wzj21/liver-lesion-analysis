"""
Classification benchmark metric helpers.

The functions use only the Python standard library until metrics are computed;
numpy is imported lazily so model catalog operations stay lightweight.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def collect_prediction_files(pred_root: Path | str) -> Dict[str, Path]:
    """Find prediction CSV files under a benchmark prediction root."""

    root = Path(pred_root)
    if not root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Prediction root is not a directory: {root}")

    prediction_files: Dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_dir():
            predictions_csv = path / "predictions.csv"
            if predictions_csv.exists():
                prediction_files[path.name] = predictions_csv
        elif path.is_file() and path.suffix.lower() == ".csv":
            prediction_files[path.stem] = path

    return prediction_files


def load_labels_csv(
    path: Path | str,
    case_col: str = "case_id",
    label_col: str = "label",
) -> Dict[str, int]:
    """Load case labels from a CSV file."""

    labels: Dict[str, int] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        _require_columns(reader.fieldnames, [case_col, label_col], path)
        for row in reader:
            case_id = row[case_col].strip()
            if not case_id:
                continue
            labels[case_id] = int(row[label_col])
    return labels


def load_predictions_csv(
    path: Path | str,
    num_classes: int,
    case_col: str = "case_id",
    pred_col: str = "pred",
    prob_cols: Optional[Sequence[str]] = None,
    prob_prefix: str = "prob_",
) -> Dict[str, Dict[str, Any]]:
    """Load predictions and optional probabilities from a CSV file."""

    predictions: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        _require_columns(reader.fieldnames, [case_col], path)
        fieldnames = list(reader.fieldnames or [])
        resolved_prob_cols = _resolve_probability_columns(
            fieldnames=fieldnames,
            num_classes=num_classes,
            prob_cols=prob_cols,
            prob_prefix=prob_prefix,
        )
        has_pred = pred_col in fieldnames

        for row in reader:
            case_id = row[case_col].strip()
            if not case_id:
                continue
            probs = _parse_probabilities(row, resolved_prob_cols)
            pred = int(row[pred_col]) if has_pred and row.get(pred_col, "") != "" else None
            if pred is None and probs is not None:
                pred = max(range(len(probs)), key=probs.__getitem__)
            if pred is None:
                raise ValueError(f"Prediction row has neither {pred_col!r} nor probabilities: {case_id}")
            predictions[case_id] = {"pred": pred, "probs": probs}

    return predictions


def classification_metrics(
    targets: Sequence[int],
    preds: Sequence[int],
    probs: Optional[Sequence[Sequence[float]]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute multi-class or binary classification metrics."""

    import numpy as np

    y_true = np.asarray(targets, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Target and prediction lengths differ: {len(y_true)} vs {len(y_pred)}")
    if y_true.size == 0:
        raise ValueError("No classification cases were provided.")

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    names = _class_names(num_classes, class_names)
    confusion = _confusion_matrix(y_true, y_pred, num_classes)
    total = float(confusion.sum())

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    specificities: List[float] = []
    f1s: List[float] = []
    supports: List[float] = []

    for index, name in enumerate(names):
        tp = float(confusion[index, index])
        fn = float(confusion[index, :].sum() - tp)
        fp = float(confusion[:, index].sum() - tp)
        tn = float(total - tp - fn - fp)
        support = tp + fn

        precision = _safe_divide(tp, tp + fp, empty_value=0.0)
        recall = _safe_divide(tp, tp + fn, empty_value=0.0)
        specificity = _safe_divide(tn, tn + fp, empty_value=0.0)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall, empty_value=0.0)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1s.append(f1)
        supports.append(support)

    support_sum = sum(supports)
    weights = [support / support_sum if support_sum else 0.0 for support in supports]

    metrics: Dict[str, Any] = {
        "num_cases": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": fmean(recalls),
        "macro_precision": fmean(precisions),
        "macro_recall": fmean(recalls),
        "macro_specificity": fmean(specificities),
        "macro_f1": fmean(f1s),
        "weighted_precision": _weighted_mean(precisions, weights),
        "weighted_recall": _weighted_mean(recalls, weights),
        "weighted_f1": _weighted_mean(f1s, weights),
        "confusion_matrix": confusion.astype(int).tolist(),
        "per_class": per_class,
    }

    for index, name in enumerate(names):
        metrics[f"precision_{name}"] = per_class[name]["precision"]
        metrics[f"recall_{name}"] = per_class[name]["recall"]
        metrics[f"specificity_{name}"] = per_class[name]["specificity"]
        metrics[f"f1_{name}"] = per_class[name]["f1"]

    if probs is not None:
        prob_array = np.asarray(probs, dtype=float)
        if prob_array.shape != (y_true.size, num_classes):
            raise ValueError(
                f"Probability shape must be {(y_true.size, num_classes)}, got {prob_array.shape}"
            )
        aucs = []
        auc_weights = []
        for index, name in enumerate(names):
            auc = binary_auc((y_true == index).astype(int), prob_array[:, index])
            if auc is not None:
                metrics[f"auc_{name}"] = auc
                aucs.append(auc)
                auc_weights.append(supports[index])
        if aucs:
            metrics["macro_auc_ovr"] = fmean(aucs)
            metrics["weighted_auc_ovr"] = _weighted_mean(aucs, _normalize_weights(auc_weights))

    return metrics


def binary_auc(binary_targets: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Compute binary ROC AUC with average ranks for ties."""

    import numpy as np

    y = np.asarray(binary_targets, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]

    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rank_model_summaries(
    summaries: Iterable[Mapping[str, Any]],
    primary_metric: str = "macro_auc_ovr",
    tie_breaker: str = "macro_f1",
) -> List[Dict[str, Any]]:
    """Rank models by primary metric, then tie breaker, then accuracy."""

    ranked = [dict(row) for row in summaries]

    def sort_key(row: Mapping[str, Any]):
        primary = _metric_or_none(row, primary_metric)
        tie = _metric_or_none(row, tie_breaker)
        accuracy = _metric_or_none(row, "accuracy")
        return (
            -(primary if primary is not None else -1.0),
            -(tie if tie is not None else -1.0),
            -(accuracy if accuracy is not None else -1.0),
            str(row.get("model", "")),
        )

    ranked.sort(key=sort_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def json_ready(value: Any) -> Any:
    """Convert values to strict JSON-safe types."""

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def csv_ready(value: Any) -> Any:
    """Convert nested values for CSV export."""

    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False)
    return value


def _require_columns(fieldnames: Optional[Sequence[str]], required: Sequence[str], path: Path | str) -> None:
    fields = set(fieldnames or [])
    missing = [column for column in required if column not in fields]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _resolve_probability_columns(
    fieldnames: Sequence[str],
    num_classes: int,
    prob_cols: Optional[Sequence[str]],
    prob_prefix: str,
) -> Optional[List[str]]:
    if prob_cols:
        missing = [column for column in prob_cols if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")
        return list(prob_cols)

    expected = [f"{prob_prefix}{index}" for index in range(num_classes)]
    if all(column in fieldnames for column in expected):
        return expected
    return None


def _parse_probabilities(row: Mapping[str, str], prob_cols: Optional[Sequence[str]]) -> Optional[List[float]]:
    if not prob_cols:
        return None
    values = [row.get(column, "") for column in prob_cols]
    if any(value == "" for value in values):
        return None
    return [float(value) for value in values]


def _class_names(num_classes: int, class_names: Optional[Sequence[str]]) -> List[str]:
    if class_names:
        if len(class_names) != num_classes:
            raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")
        return [_sanitize_metric_name(name) for name in class_names]
    return [f"class_{index}" for index in range(num_classes)]


def _sanitize_metric_name(name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def _confusion_matrix(y_true: Any, y_pred: Any, num_classes: int) -> Any:
    import numpy as np

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for target, pred in zip(y_true, y_pred):
        if 0 <= int(target) < num_classes and 0 <= int(pred) < num_classes:
            confusion[int(target), int(pred)] += 1
    return confusion


def _safe_divide(numerator: float, denominator: float, empty_value: float) -> float:
    if denominator == 0.0:
        return empty_value
    return float(numerator / denominator)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    if not weights or sum(weights) == 0.0:
        return fmean(values)
    return float(sum(value * weight for value, weight in zip(values, weights)))


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total == 0.0:
        return [1.0 / len(weights)] * len(weights) if weights else []
    return [float(weight) / total for weight in weights]


def _metric_or_none(row: Mapping[str, Any], metric: str) -> Optional[float]:
    value = row.get(metric)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
