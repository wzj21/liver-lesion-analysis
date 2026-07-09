#!/usr/bin/env python3
"""
Validate real-world data templates or curated CSV manifests.

This script uses only the Python standard library so it can run in CI before
heavy medical imaging dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REQUIRED_FILES = {
    "patients.csv": ["patient_id", "sex", "birth_year", "region", "center_id"],
    "studies.csv": ["study_id", "patient_id", "modality", "phase_or_sequence", "split"],
    "lesions.csv": ["lesion_id", "study_id", "patient_id", "lesion_index", "main_class", "confidence", "uncertainty"],
    "patient_multilabels.csv": ["patient_id", "study_id", "has_ce", "has_ae", "has_benign", "has_malignant"],
    "followups.csv": ["patient_id", "study_id", "followup_date"],
    "surgeries.csv": ["patient_id", "study_id", "surgery_date"],
    "petct_activity.csv": ["patient_id", "study_id", "lesion_id", "activity_label"],
    "synthetic_manifest.csv": [
        "synthetic_id",
        "source_patient_id",
        "source_study_id",
        "source_lesion_id",
        "split",
        "image_path",
        "mask_path",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check real-world liver lesion CSV schema.")
    parser.add_argument("--templates-dir", default="data_templates", help="Directory containing CSV templates/manifests.")
    parser.add_argument("--label-schema", default="configs/label_schema.json", help="Label schema JSON path.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    templates_dir = Path(args.templates_dir)
    label_schema_path = Path(args.label_schema)

    errors: List[str] = []
    warnings: List[str] = []

    if not templates_dir.exists():
        errors.append(f"Template directory does not exist: {templates_dir}")
    if not label_schema_path.exists():
        errors.append(f"Label schema does not exist: {label_schema_path}")
    if errors:
        _print_findings(errors, warnings)
        return 1

    schema = json.loads(label_schema_path.read_text(encoding="utf-8"))
    allowed_main_classes = set((schema.get("main_classes") or {}).keys())
    protected_splits = {"val", "internal_test", "external_test", "prospective_test"}

    loaded_tables: Dict[str, List[Mapping[str, str]]] = {}
    for filename, required_columns in REQUIRED_FILES.items():
        path = templates_dir / filename
        if not path.exists():
            errors.append(f"Missing required CSV: {path}")
            continue
        rows, fieldnames = _read_csv(path)
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            errors.append(f"{filename} is missing columns: {missing}")
        if not rows:
            warnings.append(f"{filename} has no data rows.")
        else:
            _check_required_values(filename, rows, required_columns, errors)
        loaded_tables[filename] = rows

    _check_lesions(loaded_tables.get("lesions.csv", []), allowed_main_classes, errors)
    _check_synthetic_manifest(
        loaded_tables.get("synthetic_manifest.csv", []),
        protected_splits=protected_splits,
        errors=errors,
    )
    _check_cross_table_ids(loaded_tables, errors, warnings)

    _print_findings(errors, warnings)
    if errors or (args.strict and warnings):
        return 1
    return 0


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _check_required_values(
    filename: str,
    rows: Sequence[Mapping[str, str]],
    required_columns: Sequence[str],
    errors: List[str],
) -> None:
    for row_index, row in enumerate(rows, start=2):
        for column in required_columns:
            if (row.get(column) or "").strip() == "":
                errors.append(f"{filename} row {row_index} has empty required value: {column}")


def _check_lesions(rows: Sequence[Mapping[str, str]], allowed_main_classes: Iterable[str], errors: List[str]) -> None:
    seen = set()
    allowed = set(allowed_main_classes)
    for row in rows:
        lesion_id = row.get("lesion_id", "")
        if lesion_id in seen:
            errors.append(f"Duplicate lesion_id: {lesion_id}")
        seen.add(lesion_id)

        main_class = (row.get("main_class") or "").upper()
        if main_class and main_class not in allowed:
            errors.append(f"Invalid main_class for lesion {lesion_id}: {main_class}")

        for column in ["confidence", "uncertainty"]:
            value = row.get(column, "")
            if value == "":
                continue
            try:
                numeric = float(value)
            except ValueError:
                errors.append(f"{column} for lesion {lesion_id} is not numeric: {value}")
                continue
            if not 0.0 <= numeric <= 1.0:
                errors.append(f"{column} for lesion {lesion_id} must be between 0 and 1: {value}")


def _check_synthetic_manifest(
    rows: Sequence[Mapping[str, str]],
    protected_splits: set[str],
    errors: List[str],
) -> None:
    for row in rows:
        split = (row.get("split") or "").strip()
        synthetic_id = row.get("synthetic_id", "")
        if split in protected_splits:
            errors.append(f"Synthetic sample {synthetic_id} is assigned to protected split: {split}")


def _check_cross_table_ids(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    errors: List[str],
    warnings: List[str],
) -> None:
    patient_ids = {row.get("patient_id", "") for row in tables.get("patients.csv", [])}
    studies = tables.get("studies.csv", [])
    study_ids = {row.get("study_id", "") for row in studies}
    study_patient_by_id = {row.get("study_id", ""): row.get("patient_id", "") for row in studies}
    lesion_ids = {row.get("lesion_id", "") for row in tables.get("lesions.csv", [])}

    _check_duplicate_ids(tables.get("patients.csv", []), "patient_id", "patients.csv", errors)
    _check_duplicate_ids(studies, "study_id", "studies.csv", errors)

    for table_name in ["studies.csv", "lesions.csv", "patient_multilabels.csv", "followups.csv", "surgeries.csv"]:
        for row in tables.get(table_name, []):
            patient_id = row.get("patient_id", "")
            if patient_ids and patient_id and patient_id not in patient_ids:
                errors.append(f"{table_name} references unknown patient_id: {patient_id}")
            study_id = row.get("study_id", "")
            if study_ids and study_id and study_id not in study_ids:
                errors.append(f"{table_name} references unknown study_id: {study_id}")
            expected_patient_id = study_patient_by_id.get(study_id)
            if expected_patient_id and patient_id and patient_id != expected_patient_id:
                errors.append(
                    f"{table_name} row patient_id {patient_id} does not match study {study_id} patient_id {expected_patient_id}"
                )

    for row in tables.get("petct_activity.csv", []):
        patient_id = row.get("patient_id", "")
        study_id = row.get("study_id", "")
        lesion_id = row.get("lesion_id", "")
        if patient_ids and patient_id and patient_id not in patient_ids:
            errors.append(f"petct_activity.csv references unknown patient_id: {patient_id}")
        if study_ids and study_id and study_id not in study_ids:
            errors.append(f"petct_activity.csv references unknown study_id: {study_id}")
        if lesion_ids and lesion_id and lesion_id not in lesion_ids:
            errors.append(f"petct_activity.csv references unknown lesion_id: {lesion_id}")

    for row in tables.get("synthetic_manifest.csv", []):
        source_patient_id = row.get("source_patient_id", "")
        source_study_id = row.get("source_study_id", "")
        source_lesion_id = row.get("source_lesion_id", "")
        if patient_ids and source_patient_id and source_patient_id not in patient_ids:
            errors.append(f"synthetic_manifest.csv references unknown source_patient_id: {source_patient_id}")
        if study_ids and source_study_id and source_study_id not in study_ids:
            errors.append(f"synthetic_manifest.csv references unknown source_study_id: {source_study_id}")
        if lesion_ids and source_lesion_id and source_lesion_id not in lesion_ids:
            errors.append(f"synthetic_manifest.csv references unknown source_lesion_id: {source_lesion_id}")

    if patient_ids and len(patient_ids) < 2:
        warnings.append("Template currently contains fewer than two patients; this is fine for templates only.")


def _check_duplicate_ids(
    rows: Sequence[Mapping[str, str]],
    column: str,
    table_name: str,
    errors: List[str],
) -> None:
    seen = set()
    for row in rows:
        value = row.get(column, "")
        if value in seen:
            errors.append(f"Duplicate {column} in {table_name}: {value}")
        seen.add(value)


def _print_findings(errors: Sequence[str], warnings: Sequence[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if not errors:
        print("Real-world dataset schema check passed.")


if __name__ == "__main__":
    raise SystemExit(main())
