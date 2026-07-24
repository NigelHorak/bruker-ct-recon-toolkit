"""
Shared Bruker recon engine used by the GUI and CLI.
"""
from __future__ import annotations

import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from parse_bruker_log import BrukerLogMeta, estimate_angles_deg, parse_bruker_log

ProgressCb = Optional[Callable[[str], None]]

RING_METHODS = (
    "remove_all_stripe",
    "remove_stripe_based_sorting",
    "remove_large_stripe",
    "remove_dead_stripe",
    "none",
)

RECON_METHODS = (
    "FBP_CUDA",
    "SIRT_CUDA",
    "SART_CUDA",
    "CGLS_CUDA",
    "FBP",
    "SIRT",
    "SART",
    "CGLS",
)

FILTER_NAMES = ("hann", "ram-lak", "shepp-logan", "cosine", "hamming")


@dataclass
class Settings:
    # Ring
    ring_enable: bool = True
    ring_method: str = "remove_all_stripe"
    snr: float = 3.0
    la_size: int = 51
    sm_size: int = 21
    drop_ratio: float = 0.1
    dim: int = 1
    # Recon
    method: str = "FBP_CUDA"
    filter_name: str = "hann"
    apply_log: bool = True
    num_iter: int = 100
    chunk_size: int = 32
    center_mode: str = "auto"  # auto | manual
    center: Optional[float] = None
    # IO / preview
    preview_row: Optional[int] = None  # None = mid
    output_dir: str = ""
    save_preview: bool = True

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "ring": {
                "enable": self.ring_enable,
                "method": self.ring_method,
                "snr": self.snr,
                "la_size": self.la_size,
                "sm_size": self.sm_size,
                "drop_ratio": self.drop_ratio,
                "dim": self.dim,
            },
            "recon": {
                "method": self.method,
                "filter_name": self.filter_name,
                "apply_log": self.apply_log,
                "num_iter": self.num_iter,
                "chunk_size": self.chunk_size,
                "center_mode": self.center_mode,
                "center": self.center,
            },
            "io": {"save_preview": self.save_preview},
            "paths": {"output_dir": self.output_dir},
            "preview": {"row": self.preview_row},
        }

    @staticmethod
    def from_config_dict(cfg: Dict[str, Any]) -> "Settings":
        ring = cfg.get("ring", {}) or {}
        recon = cfg.get("recon", {}) or {}
        io_cfg = cfg.get("io", {}) or {}
        paths = cfg.get("paths", {}) or {}
        preview = cfg.get("preview", {}) or {}
        center = recon.get("center", None)
        center_mode = recon.get("center_mode") or ("manual" if center is not None else "auto")
        return Settings(
            ring_enable=bool(ring.get("enable", True)),
            ring_method=str(ring.get("method", "remove_all_stripe")),
            snr=float(ring.get("snr", 3.0)),
            la_size=int(ring.get("la_size", 51)),
            sm_size=int(ring.get("sm_size", 21)),
            drop_ratio=float(ring.get("drop_ratio", 0.1)),
            dim=int(ring.get("dim", 1)),
            method=str(recon.get("method", "FBP_CUDA")),
            filter_name=str(recon.get("filter_name", "hann")),
            apply_log=bool(recon.get("apply_log", True)),
            num_iter=int(recon.get("num_iter", 100)),
            chunk_size=int(recon.get("chunk_size", 32)),
            center_mode=str(center_mode),
            center=None if center is None else float(center),
            preview_row=preview.get("row", None),
            output_dir=str(paths.get("output_dir") or ""),
            save_preview=bool(io_cfg.get("save_preview", True)),
        )


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_settings(path: Path) -> Settings:
    return Settings.from_config_dict(load_yaml(path))


def _log(msg: str, progress: ProgressCb = None) -> None:
    print(msg)
    if progress:
        progress(msg)


def find_log_file(scan_dir: Path) -> Path:
    logs = sorted(scan_dir.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No .log file found in {scan_dir}")
    for p in logs:
        if "recon" not in p.stem.lower():
            return p
    return logs[0]


def _proj_index(path: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", path.stem)
    idx = int(m.group(1)) if m else -1
    return idx, path.name.lower()


def list_projections(scan_dir: Path, prefix: str = "") -> List[Path]:
    files: List[Path] = []
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(scan_dir.glob(pat))
    skip = ("_rec", "rec_", "dark", "flat", "ref", "white", "bkg", "background", "arc")
    filtered: List[Path] = []
    for p in files:
        name = p.name.lower()
        if any(tok in name for tok in skip):
            continue
        if prefix and not p.name.startswith(prefix):
            if prefix.rstrip("_") and not p.name.startswith(prefix.rstrip("_")):
                continue
        filtered.append(p)
    if not filtered:
        filtered = [p for p in files if re.search(r"\d+$", p.stem)]
    filtered = sorted(set(filtered), key=_proj_index)
    filtered = [p for p in filtered if _proj_index(p)[0] >= 0]
    if not filtered:
        raise FileNotFoundError(f"No projection TIFFs found in {scan_dir}")
    return filtered


def imread(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except ImportError:
        from algotom.io import loadersaver as losa

        return np.asarray(losa.load_image(str(path)), dtype=np.float32)


def probe_shape(paths: List[Path]) -> Tuple[int, int]:
    arr = imread(paths[0])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got {arr.shape} from {paths[0]}")
    return int(arr.shape[0]), int(arr.shape[1])


def load_stack(paths: List[Path], progress: ProgressCb = None) -> np.ndarray:
    imgs = []
    n = len(paths)
    for i, p in enumerate(paths):
        if i % 50 == 0 or i == n - 1:
            _log(f"Loading projection {i + 1}/{n}: {p.name}", progress)
        imgs.append(imread(p))
    return np.stack(imgs, axis=0)


def load_sinogram_row(paths: List[Path], row: int, progress: ProgressCb = None) -> np.ndarray:
    n = len(paths)
    first = imread(paths[0])
    height, width = first.shape
    if row < 0 or row >= height:
        raise ValueError(f"row {row} out of range 0..{height - 1}")
    sino = np.empty((n, width), dtype=np.float32)
    sino[0] = first[row, :]
    for i, p in enumerate(paths[1:], start=1):
        if i % 50 == 0 or i == n - 1:
            _log(f"Loading row {row} from proj {i + 1}/{n}", progress)
        sino[i] = imread(p)[row, :]
    return sino


def default_output_dir(scan_dir: Path, configured: str, preview: bool) -> Path:
    if configured:
        out = Path(configured)
        return out if out.is_absolute() else scan_dir / out
    suffix = "_algotom_preview" if preview else "_algotom_recon"
    return scan_dir.parent / f"{scan_dir.name}{suffix}"


def save_qc_png(path: Path, img: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def apply_ring_removal(sino: np.ndarray, settings: Settings) -> np.ndarray:
    if (not settings.ring_enable) or settings.ring_method == "none":
        return sino
    import algotom.prep.removal as remo

    method = settings.ring_method
    snr, la, sm = settings.snr, settings.la_size, settings.sm_size
    drop, dim = settings.drop_ratio, settings.dim
    if method == "remove_all_stripe":
        return remo.remove_all_stripe(sino, snr=snr, la_size=la, sm_size=sm, drop_ratio=drop, dim=dim)
    if method == "remove_stripe_based_sorting":
        return remo.remove_stripe_based_sorting(sino, size=sm, dim=dim)
    if method == "remove_large_stripe":
        return remo.remove_large_stripe(sino, snr=snr, size=la, drop_ratio=drop)
    if method == "remove_dead_stripe":
        return remo.remove_dead_stripe(sino, snr=snr, size=la, residual=False)
    raise ValueError(f"Unknown ring method: {method}")


def find_center(sino: np.ndarray, settings: Settings, width: int) -> float:
    if settings.center_mode == "manual" and settings.center is not None:
        return float(settings.center)
    import algotom.prep.calculation as calc

    try:
        return float(calc.find_center_vo(sino))
    except Exception:
        return (width - 1) / 2.0


def reconstruct_sinogram(
    sino: np.ndarray,
    center: float,
    thetas: np.ndarray,
    settings: Settings,
) -> np.ndarray:
    import algotom.rec.reconstruction as rec

    kwargs: Dict[str, Any] = {
        "angles": thetas,
        "method": settings.method,
        "filter_name": settings.filter_name,
        "apply_log": settings.apply_log,
    }
    # num_iter used by iterative Astra methods
    if "SIRT" in settings.method.upper() or "SART" in settings.method.upper() or "CGLS" in settings.method.upper():
        kwargs["num_iter"] = settings.num_iter
    try:
        img = rec.astra_reconstruction(sino, center, **kwargs)
    except Exception:
        if settings.method.upper().endswith("CUDA"):
            kwargs["method"] = "FBP"
            img = rec.astra_reconstruction(sino, center, **kwargs)
        else:
            raise
    return np.asarray(img, dtype=np.float32)


def _norm_display(img: np.ndarray) -> np.ndarray:
    """Percentile-scale float image to uint8 for Gradio display."""
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max()) + 1e-6
    scaled = np.clip((img - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


@dataclass
class PreviewResult:
    out_dir: Path
    center: float
    row: int
    height: int
    width: int
    n_projections: int
    img_raw: np.ndarray
    img_corr: np.ndarray
    display_raw: np.ndarray
    display_corr: np.ndarray
    settings: Settings
    meta: BrukerLogMeta
    message: str


@dataclass
class FullResult:
    out_dir: Path
    center: float
    height: int
    width: int
    n_projections: int
    settings: Settings
    message: str


def _prepare_scan(scan_dir: Path, progress: ProgressCb = None):
    scan_dir = Path(scan_dir).resolve()
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan folder not found: {scan_dir}")
    log_path = find_log_file(scan_dir)
    meta = parse_bruker_log(log_path)
    _log(f"Log: {log_path}", progress)
    proj_paths = list_projections(scan_dir, prefix=meta.filename_prefix)
    height, width = probe_shape(proj_paths)
    _log(f"Projections: {len(proj_paths)}  detector: {height}x{width}", progress)
    return scan_dir, log_path, meta, proj_paths, height, width


def run_preview(
    scan_dir: Path,
    settings: Settings,
    out_dir: Optional[Path] = None,
    progress: ProgressCb = None,
) -> PreviewResult:
    from algotom.io import loadersaver as losa

    scan_dir, log_path, meta, proj_paths, height, width = _prepare_scan(scan_dir, progress)
    mid = height // 2
    row = mid if settings.preview_row is None else int(settings.preview_row)
    if row < 0 or row >= height:
        raise ValueError(f"preview row {row} out of range 0..{height - 1}")

    out = Path(out_dir) if out_dir else default_output_dir(scan_dir, settings.output_dir, True)
    qc = out / "qc"
    if qc.exists():
        shutil.rmtree(qc)
    qc.mkdir(parents=True, exist_ok=True)

    _log(f"Preview row {row}/{height - 1}", progress)
    sino = load_sinogram_row(proj_paths, row, progress)
    n_angles = sino.shape[0]
    thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))
    center = find_center(sino, settings, width)
    _log(f"Center of rotation: {center:.3f}", progress)

    img_raw = reconstruct_sinogram(sino.copy(), center, thetas, settings)
    try:
        sino_corr = apply_ring_removal(sino.copy(), settings)
    except Exception as exc:
        _log(f"Ring removal failed: {exc}", progress)
        sino_corr = sino
    img_corr = reconstruct_sinogram(sino_corr, center, thetas, settings)

    losa.save_image(str(qc / "preview_raw.tif"), img_raw)
    losa.save_image(str(qc / "preview_corrected.tif"), img_corr)
    if settings.save_preview:
        save_qc_png(qc / "before.png", img_raw, f"BEFORE rings  row={row}")
        save_qc_png(qc / "after.png", img_corr, f"AFTER rings  row={row}  c={center:.2f}")

    used = deepcopy(settings)
    used.center = center
    used.center_mode = settings.center_mode
    used.preview_row = row
    run_info = {
        "mode": "preview",
        "preview_row": row,
        "scan_dir": str(scan_dir),
        "log_file": str(log_path),
        "n_projections": n_angles,
        "volume_shape_hw": [height, width],
        "center": center,
        "config": used.to_config_dict(),
        "meta_summary": meta.to_dict(),
    }
    # shrink raw dump
    run_info["meta_summary"].pop("raw", None)
    save_yaml(out / "run_config.yaml", run_info)

    msg = f"Preview OK. Center={center:.3f}. Saved under {out}"
    _log(msg, progress)
    return PreviewResult(
        out_dir=out,
        center=center,
        row=row,
        height=height,
        width=width,
        n_projections=n_angles,
        img_raw=img_raw,
        img_corr=img_corr,
        display_raw=_norm_display(img_raw),
        display_corr=_norm_display(img_corr),
        settings=used,
        meta=meta,
        message=msg,
    )


def run_full(
    scan_dir: Path,
    settings: Settings,
    out_dir: Optional[Path] = None,
    progress: ProgressCb = None,
) -> FullResult:
    from algotom.io import loadersaver as losa

    scan_dir, log_path, meta, proj_paths, height, width = _prepare_scan(scan_dir, progress)
    mid = height // 2
    out = Path(out_dir) if out_dir else default_output_dir(scan_dir, settings.output_dir, False)
    slices_dir = out / "slices"
    qc = out / "qc"
    if slices_dir.exists():
        shutil.rmtree(slices_dir)
    if qc.exists():
        shutil.rmtree(qc)
    slices_dir.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)

    _log("Loading full projection stack...", progress)
    projections = load_stack(proj_paths, progress)
    n_angles = projections.shape[0]
    thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))
    center = find_center(projections[:, mid, :], settings, width)
    _log(f"Center of rotation: {center:.3f}", progress)

    chunk = max(1, int(settings.chunk_size))
    mid_img = None
    n_done = 0
    for row0 in range(0, height, chunk):
        row1 = min(row0 + chunk, height)
        for r in range(row0, row1):
            sino = projections[:, r, :].copy()
            try:
                sino = apply_ring_removal(sino, settings)
            except Exception as exc:
                _log(f"Ring fail row {r}: {exc}", progress)
            img = reconstruct_sinogram(sino, center, thetas, settings)
            losa.save_image(str(slices_dir / f"recon_{r:05d}.tif"), img)
            if r == mid:
                mid_img = img
            n_done += 1
        _log(f"Reconstructed rows {row0}..{row1 - 1} ({n_done}/{height})", progress)

    if mid_img is not None and settings.save_preview:
        losa.save_image(str(qc / "preview_corrected.tif"), mid_img)
        save_qc_png(qc / "after.png", mid_img, f"FULL mid row={mid} c={center:.2f}")

    used = deepcopy(settings)
    used.center = center
    run_info = {
        "mode": "full",
        "scan_dir": str(scan_dir),
        "log_file": str(log_path),
        "n_projections": n_angles,
        "volume_shape_hw": [height, width],
        "center": center,
        "config": used.to_config_dict(),
    }
    save_yaml(out / "run_config.yaml", run_info)
    msg = f"Full recon done. Slices: {slices_dir}"
    _log(msg, progress)
    return FullResult(
        out_dir=out,
        center=center,
        height=height,
        width=width,
        n_projections=n_angles,
        settings=used,
        message=msg,
    )


def probe_scan_info(scan_dir: str) -> Tuple[str, int, int, int]:
    """Return (status_text, height, width, mid_row) for GUI folder load."""
    scan_dir_p, log_path, meta, proj_paths, height, width = _prepare_scan(Path(scan_dir))
    mid = height // 2
    text = (
        f"OK — {len(proj_paths)} projections | detector {height}×{width} | "
        f"log={log_path.name} | prefix={meta.filename_prefix!r} | "
        f"rot_step={meta.rotation_step_deg}° | mid_row={mid}"
    )
    return text, height, width, mid
