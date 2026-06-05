"""
Dependency-light segmentation metric helpers.

The public functions lazily import numpy, scipy, and nibabel so model catalog
operations remain usable in a minimal GitHub Actions environment.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SUPPORTED_MASK_SUFFIXES = (".nii.gz", ".nii", ".npy", ".npz")


def case_id_from_path(path: Path | str) -> str:
    """Return a stable case id after stripping known mask suffixes."""

    name = Path(path).name
    for suffix in SUPPORTED_MASK_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def collect_mask_paths(directory: Path | str) -> Dict[str, Path]:
    """Collect supported mask files in one directory, keyed by case id."""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {root}")

    mask_paths: Dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.endswith(suffix) for suffix in SUPPORTED_MASK_SUFFIXES):
            continue
        case_id = case_id_from_path(path)
        if case_id in mask_paths:
            raise ValueError(f"Duplicate mask for case id {case_id!r} in {root}")
        mask_paths[case_id] = path
    return mask_paths


def load_mask(path: Path | str, key: Optional[str] = None) -> Any:
    """Load a NIfTI, NumPy, or NPZ mask array."""

    mask_path = Path(path)
    suffix_name = mask_path.name.lower()

    if suffix_name.endswith(".npy"):
        import numpy as np

        return np.load(mask_path)

    if suffix_name.endswith(".npz"):
        import numpy as np

        data = np.load(mask_path)
        if key is None:
            if not data.files:
                raise ValueError(f"NPZ file has no arrays: {mask_path}")
            key = data.files[0]
        return data[key]

    if suffix_name.endswith(".nii") or suffix_name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "nibabel is required to load NIfTI masks. Install project "
                "dependencies with 'pip install -r requirements.txt'."
            ) from exc

        return nib.load(str(mask_path)).get_fdata()

    raise ValueError(f"Unsupported mask file: {mask_path}")


def binarize_mask(mask: Any, label: Optional[int] = None) -> Any:
    """Convert a label mask to a boolean foreground mask."""

    import numpy as np

    array = np.asarray(mask)
    if label is None:
        return array > 0
    return array == label


def segmentation_metrics(
    pred: Any,
    target: Any,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    label: Optional[int] = None,
    compute_hd95: bool = True,
) -> Dict[str, float]:
    """Compute common binary segmentation metrics for one case."""

    import numpy as np

    pred_mask = binarize_mask(pred, label=label)
    target_mask = binarize_mask(target, label=label)

    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred_mask.shape} vs {target_mask.shape}"
        )

    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    tp = float(np.logical_and(pred_mask, target_mask).sum())
    fp = float(np.logical_and(pred_mask, ~target_mask).sum())
    fn = float(np.logical_and(~pred_mask, target_mask).sum())
    tn = float(np.logical_and(~pred_mask, ~target_mask).sum())

    pred_volume = tp + fp
    target_volume = tp + fn

    metrics = {
        "dice": _safe_divide(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0),
        "iou": _safe_divide(tp, tp + fp + fn, empty_value=1.0),
        "precision": _safe_divide(tp, tp + fp, empty_value=1.0),
        "recall": _safe_divide(tp, tp + fn, empty_value=1.0),
        "specificity": _safe_divide(tn, tn + fp, empty_value=1.0),
        "volume_similarity": _volume_similarity(pred_volume, target_volume),
        "pred_volume_voxels": pred_volume,
        "target_volume_voxels": target_volume,
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
    }

    if compute_hd95:
        metrics["hd95"] = hausdorff95(pred_mask, target_mask, spacing=spacing)

    return metrics


def hausdorff95(
    pred_mask: Any,
    target_mask: Any,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute symmetric 95th percentile Hausdorff distance."""

    import numpy as np

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for HD95. Install project dependencies or run "
            "with --no-hd95."
        ) from exc

    pred_mask = np.asarray(pred_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)

    pred_has = bool(pred_mask.any())
    target_has = bool(target_mask.any())
    if not pred_has and not target_has:
        return 0.0
    if pred_has != target_has:
        return math.inf

    pred_surface = _surface_voxels(pred_mask, ndimage)
    target_surface = _surface_voxels(target_mask, ndimage)

    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_pred[target_surface]

    if pred_to_target.size == 0 or target_to_pred.size == 0:
        return math.inf

    return float(
        max(
            np.percentile(pred_to_target, 95),
            np.percentile(target_to_pred, 95),
        )
    )


def summarize_case_metrics(case_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate case-level metric rows into one row per model."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in case_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    metric_names = (
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "volume_similarity",
        "hd95",
    )

    summaries: List[Dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {"model": model, "num_cases": len(rows)}
        for metric_name in metric_names:
            values = [_as_float(row.get(metric_name)) for row in rows]
            values = [value for value in values if value is not None]
            if values:
                summary[f"mean_{metric_name}"] = fmean(values)
        summaries.append(summary)
    return summaries


def rank_model_summaries(
    summaries: Iterable[Mapping[str, Any]],
    primary_metric: str = "mean_dice",
    hd95_metric: str = "mean_hd95",
) -> List[Dict[str, Any]]:
    """Rank models by high Dice and then low HD95."""

    ranked = [dict(row) for row in summaries]

    def sort_key(row: Mapping[str, Any]):
        primary = _as_float(row.get(primary_metric))
        hd95 = _as_float(row.get(hd95_metric))
        primary_sort = -primary if primary is not None else math.inf
        hd95_sort = hd95 if hd95 is not None else math.inf
        return (primary_sort, hd95_sort, str(row.get("model", "")))

    ranked.sort(key=sort_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def json_ready(value: Any) -> Any:
    """Convert NaN/Inf values to strings for strict JSON export."""

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


def _safe_divide(numerator: float, denominator: float, empty_value: float) -> float:
    if denominator == 0.0:
        return empty_value
    return float(numerator / denominator)


def _volume_similarity(pred_volume: float, target_volume: float) -> float:
    denominator = pred_volume + target_volume
    if denominator == 0.0:
        return 1.0
    return float(1.0 - abs(pred_volume - target_volume) / denominator)


def _surface_voxels(mask: Any, ndimage: Any) -> Any:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
