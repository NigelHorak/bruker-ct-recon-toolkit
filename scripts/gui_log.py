"""
Unified GUI logging: console (black window) + toolkit_gui.log + status text for Gradio.
"""
from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
LOG_PATH = ROOT / "toolkit_gui.log"


def log_path() -> Path:
    return LOG_PATH


def log_line(msg: str) -> str:
    """Append one line to console + file. Returns the stamped line."""
    line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with LOG_PATH.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass
    return line


def log_exception(where: str, exc: BaseException) -> str:
    """Log full traceback; return a multi-line status block for the UI."""
    tb = traceback.format_exc()
    log_line(f"ERROR in {where}: {type(exc).__name__}: {exc}")
    for row in tb.splitlines():
        log_line(f"  | {row}")
    return (
        f"FAILED: {where}\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"--- traceback ---\n{tb}\n"
        f"Also written to: {LOG_PATH}"
    )


class ProgressLog:
    """Collect progress lines for the Activity log and mirror to disk/console."""

    def __init__(self, title: str = "") -> None:
        self.lines: List[str] = []
        if title:
            self(title)

    def __call__(self, msg: str) -> None:
        stamped = log_line(msg)
        self.lines.append(stamped)

    def text(self, limit: int = 40) -> str:
        if not self.lines:
            return "(no progress lines yet)"
        return "\n".join(self.lines[-limit:])


def startup_banner() -> str:
    log_line("=" * 60)
    log_line("Bruker CT Algotom Toolkit GUI started")
    log_line(f"Log file: {LOG_PATH}")
    return (
        f"GUI ready.\n"
        f"Log file: {LOG_PATH}\n"
        f"Paste a scan folder and click Load scan.\n"
        f"Errors and progress appear here AND in the black console window."
    )


def safe_history(scan_dir: Optional[str]) -> list:
    if not scan_dir:
        return []
    try:
        from history_store import list_history_gallery

        return list_history_gallery(Path(str(scan_dir).strip().strip('"')))
    except Exception as exc:
        log_exception("history gallery", exc)
        return []
