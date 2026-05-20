#!/usr/bin/env python3
"""Launch the Windows desktop GUI."""

from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.software.gui_app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
