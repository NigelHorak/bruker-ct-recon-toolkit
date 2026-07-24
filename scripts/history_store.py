"""
Save every quick-align / preview / full-recon QC shot for side-by-side comparison.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def history_root(scan_dir: Path) -> Path:
    """History lives inside the scan folder: <scan>/algotom/history/."""
    scan_dir = Path(scan_dir)
    return scan_dir / "algotom" / "history"


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


def list_history_gallery(scan_dir: Path, limit: int = 60) -> List[Tuple[np.ndarray, str]]:
    """Newest-first Gradio gallery: (image_array, caption)."""
    entries = list_history_entries(scan_dir, limit=limit)
    return [(e["image"], e["caption"]) for e in entries]


def list_history_entries(scan_dir: Path, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Newest-first history cards with image + settings for restore-on-click.
    Each item: image (np), caption (str), settings (dict), kind (str), folder (str)
    """
    import yaml
    from PIL import Image

    root = history_root(Path(scan_dir))
    if not root.is_dir():
        return []
    folders = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    out: List[Dict[str, Any]] = []

    for folder in folders[:limit]:
        caption_path = folder / "settings.txt"
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else folder.name
        cfg: Dict[str, Any] = {}
        yaml_path = folder / "settings.yaml"
        if yaml_path.is_file():
            try:
                raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                cfg = raw.get("config") or {}
            except Exception:
                cfg = {}

        # Prefer after > align > before > first m_*.png
        candidates = [
            folder / "after.png",
            folder / "align.png",
            folder / "before.png",
            folder / "diff.png",
        ]
        candidates.extend(sorted(folder.glob("m_*.png")))
        img_path = next((p for p in candidates if p.is_file()), None)
        if img_path is None:
            continue
        try:
            arr = np.asarray(Image.open(img_path).convert("RGB"))
        except Exception:
            continue
        out.append(
            {
                "image": arr,
                "caption": f"{folder.name}\n{caption}",
                "settings": cfg,
                "kind": folder.name,
                "folder": str(folder),
                "png": str(img_path),
            }
        )
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
