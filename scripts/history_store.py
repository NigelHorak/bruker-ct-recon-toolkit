"""
Save every quick-align / preview / full-recon QC shot for side-by-side comparison.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def history_root(scan_dir: Path) -> Path:
    scan_dir = Path(scan_dir)
    return scan_dir.parent / f"{scan_dir.name}_algotom_history"


def settings_caption(settings_dict: Dict[str, Any], kind: str, extra: str = "") -> str:
    ring = settings_dict.get("ring", {}) or {}
    recon = settings_dict.get("recon", {}) or {}
    lines = [
        f"{kind}",
        f"type={recon.get('recon_type', '?')}  method={recon.get('method', '?')}",
        f"shift={float(recon.get('pixel_shift', 0) or 0):+.3f}  "
        f"center={recon.get('center', 'auto')}",
        f"rings={'ON' if ring.get('enable', True) else 'OFF'}  "
        f"{ring.get('method', '')}  snr={ring.get('snr', '')}  "
        f"la={ring.get('la_size', '')}  sm={ring.get('sm_size', '')}",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _to_u8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max()) + 1e-6
    scaled = np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def save_history_entry(
    scan_dir: Path,
    kind: str,
    settings_dict: Dict[str, Any],
    images: Dict[str, np.ndarray],
    extra: str = "",
) -> Path:
    """
    images keys become filenames, e.g. {"before": arr, "after": arr, "align": arr}
    """
    import yaml
    from PIL import Image

    root = history_root(Path(scan_dir))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"{stamp}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)

    caption = settings_caption(settings_dict, kind, extra=extra)
    (folder / "settings.txt").write_text(caption + "\n", encoding="utf-8")
    with (folder / "settings.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"kind": kind, "extra": extra, "config": settings_dict}, f, sort_keys=False)

    for name, arr in images.items():
        if arr is None:
            continue
        u8 = _to_u8(arr)
        Image.fromarray(u8).save(folder / f"{name}.png")

    return folder


def list_history_gallery(scan_dir: Path, limit: int = 60) -> List[Tuple[str, str]]:
    """Newest-first Gradio gallery entries: (png_path, caption with settings)."""
    root = history_root(Path(scan_dir))
    if not root.is_dir():
        return []
    entries = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    out: List[Tuple[str, str]] = []
    for folder in entries[:limit]:
        caption_path = folder / "settings.txt"
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else folder.name
        after = folder / "after.png"
        before = folder / "before.png"
        align = folder / "align.png"
        diff = folder / "diff.png"
        if after.is_file():
            out.append((str(after), f"{folder.name}  AFTER\n{caption}"))
        if before.is_file():
            out.append((str(before), f"{folder.name}  BEFORE\n{caption}"))
        if diff.is_file():
            out.append((str(diff), f"{folder.name}  |DIFF|\n{caption}"))
        if align.is_file() and not after.is_file():
            out.append((str(align), f"{folder.name}\n{caption}"))
        # Ring-compare tiles (m_0_none.png etc.)
        for png in sorted(folder.glob("m_*.png")):
            out.append((str(png), f"{folder.name}\n{png.stem}\n{caption}"))
    return out


def before_cache_key(
    scan_dir: str,
    row: int,
    center: float,
    recon_type: str,
    method: str,
    apply_log: bool,
    filter_name: str,
) -> str:
    """Key ignores ring params — safe to reuse BEFORE when only tuning rings."""
    return (
        f"{Path(scan_dir).resolve()}|r{row}|c{center:.4f}|"
        f"{recon_type}|{method}|log{int(bool(apply_log))}|{filter_name}"
    )
