"""
CLI wrapper around recon_core (power users). Lab partners should use the GUI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recon_core import load_settings, run_full, run_preview  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bruker Algotom recon (CLI)")
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-row", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.scan_dir.is_dir():
        print(f"ERROR: scan dir not found: {args.scan_dir}")
        return 1
    if not args.config.is_file():
        print(f"ERROR: config not found: {args.config}")
        return 1

    settings = load_settings(args.config)
    if args.preview_row is not None:
        settings.preview_row = args.preview_row

    if args.preview:
        run_preview(args.scan_dir, settings, args.out_dir)
    else:
        run_full(args.scan_dir, settings, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
