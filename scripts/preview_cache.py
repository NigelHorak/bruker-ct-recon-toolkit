"""
Disk cache for recon previews under <scan>/algotom/previews/.
Same parameters => reuse PNG, no recompute (log only).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def previews_root(scan_dir: Path) -> Path:
    return Path(scan_dir) / "algotom" / "previews"


def preview_params_key(
    scan_dir: Path,
    *,
    kind: str,
    row: int,
    pixel_shift: float,
    ring_enable: bool,
    ring_method: str,
    snr: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    recon_type: str = "FBP",
    filter_name: str = "hann",
    apply_log: bool = True,
    bh_strength: float = 0.0,
    extra: str = "",
) -> Dict[str, Any]:
    return {
        "sample": Path(scan_dir).name,
        "kind": str(kind),
        "row": int(row),
        "pixel_shift": round(float(pixel_shift or 0.0), 4),
        "ring_enable": bool(ring_enable),
        "ring_method": str(ring_method),
        "snr": round(float(snr), 4),
        "la_size": int(la_size),
        "sm_size": int(sm_size),
        "drop_ratio": round(float(drop_ratio), 4),
        "dim": int(dim),
        "recon_type": str(recon_type).upper(),
        "filter_name": str(filter_name),
        "apply_log": bool(apply_log),
        "bh_strength": round(float(bh_strength or 0.0), 4),
        "extra": str(extra or ""),
    }


def params_hash(params: Dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def format_params_log(params: Dict[str, Any]) -> str:
    ring = "off"
    if params.get("ring_enable") and params.get("ring_method") != "none":
        ring = (
            f"{params.get('ring_method')} snr={params.get('snr')} "
            f"la={params.get('la_size')} sm={params.get('sm_size')}"
        )
    return (
        f"settings Center shift: {params.get('pixel_shift'):+.4f}, "
        f"ring correction: {ring}, "
        f"row={params.get('row')}, sample={params.get('sample')}, "
        f"type={params.get('recon_type')}, log={params.get('apply_log')}, "
        f"BH={params.get('bh_strength')}"
    )


def preview_folder(scan_dir: Path, params: Dict[str, Any]) -> Path:
    return previews_root(scan_dir) / f"{params.get('kind', 'recon')}_{params_hash(params)}"


def load_cached_preview(
    scan_dir: Path, params: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, Path]]:
    """Return (image, folder) if cache hit."""
    from PIL import Image

    folder = preview_folder(scan_dir, params)
    png = folder / "recon.png"
    meta = folder / "params.json"
    if not png.is_file() or not meta.is_file():
        return None
    try:
        arr = np.asarray(Image.open(png).convert("RGB"))
        return arr, folder
    except Exception:
        return None


def save_cached_preview(
    scan_dir: Path,
    params: Dict[str, Any],
    image: np.ndarray,
) -> Path:
    import yaml
    from PIL import Image

    folder = preview_folder(scan_dir, params)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    with (folder / "params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    x = np.asarray(image)
    if x.dtype != np.uint8:
        lo, hi = np.percentile(x.astype(np.float64), (1, 99))
        if hi <= lo:
            hi = lo + 1e-6
        x = (np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if x.ndim == 2:
        Image.fromarray(x).save(folder / "recon.png")
    else:
        Image.fromarray(x).save(folder / "recon.png")
    return folder
