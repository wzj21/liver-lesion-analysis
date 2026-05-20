#!/usr/bin/env python3
"""Launch the Windows desktop GUI."""

from pathlib import Path
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the liver lesion desktop GUI.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the desktop launcher version and exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.version:
        print("LiverLesionAI desktop launcher 1.0.0")
        return 0

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.software.gui_app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
