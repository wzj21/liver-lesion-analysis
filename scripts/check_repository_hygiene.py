#!/usr/bin/env python3
"""
Check that sensitive or oversized medical AI artifacts are not tracked by Git.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import PurePosixPath


BLOCKED_ROOTS = {"data", "checkpoints", "outputs", "logs", "reports", "secrets"}
BLOCKED_SUFFIXES = {
    ".dcm",
    ".dicom",
    ".nii",
    ".gz",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".engine",
    ".trt",
}
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key(?![_-]?env)|secret|token|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)bearer\s+[a-z0-9._-]{20,}"),
]
PLACEHOLDER_MARKERS = {
    "your",
    "example",
    "placeholder",
    "dummy",
    "test",
    "你的",
    "<",
    "...",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_blocked_path(path: str) -> bool:
    posix = path.replace("\\", "/")
    parts = PurePosixPath(posix).parts
    if parts and parts[0] in BLOCKED_ROOTS:
        return True
    lowered = posix.lower()
    return any(lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES)


def scan_for_secret_patterns(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return []

    findings = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0).lower()
            if any(marker in matched_text for marker in PLACEHOLDER_MARKERS):
                continue
            findings.append(pattern.pattern)
            break
    return findings


def main() -> int:
    files = tracked_files()
    blocked = [path for path in files if is_blocked_path(path)]
    secret_hits = {
        path: hits
        for path in files
        if not is_blocked_path(path)
        for hits in [scan_for_secret_patterns(path)]
        if hits
    }

    if blocked:
        print("Blocked data/model/output files are tracked:")
        for path in blocked:
            print(f"  - {path}")

    if secret_hits:
        print("Potential secret patterns found:")
        for path in secret_hits:
            print(f"  - {path}")

    if blocked or secret_hits:
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
