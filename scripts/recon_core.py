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

RECON_TYPES = ("FBP", "FDK")

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

FILTER_NAMES = ("hann", "ram-lak", "shepp-logan", "cosine", "hamming", "ramlak")


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
    recon_type: str = "FBP"  # FBP (slice-wise) | FDK (cone-beam 3D)
    method: str = "FBP_CUDA"
    filter_name: str = "hann"
    apply_log: bool = True
    num_iter: int = 100
    chunk_size: int = 32
    center_mode: str = "auto"  # auto | manual
    center: Optional[float] = None
    # NRecon-style postalignment: added to auto/manual base COR (can be fractional)
    pixel_shift: float = 0.0
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
                "recon_type": self.recon_type,
                "method": self.method,
                "filter_name": self.filter_name,
                "apply_log": self.apply_log,
                "num_iter": self.num_iter,
                "chunk_size": self.chunk_size,
                "center_mode": self.center_mode,
                "center": self.center,
                "pixel_shift": self.pixel_shift,
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
        recon_type = str(recon.get("recon_type", "FBP")).upper()
        if recon_type not in ("FBP", "FDK"):
            recon_type = "FBP"
        return Settings(
            ring_enable=bool(ring.get("enable", True)),
            ring_method=str(ring.get("method", "remove_all_stripe")),
            snr=float(ring.get("snr", 3.0)),
            la_size=int(ring.get("la_size", 51)),
            sm_size=int(ring.get("sm_size", 21)),
            drop_ratio=float(ring.get("drop_ratio", 0.1)),
            dim=int(ring.get("dim", 1)),
            recon_type=recon_type,
            method=str(recon.get("method", "FBP_CUDA")),
            filter_name=str(recon.get("filter_name", "hann")),
            apply_log=bool(recon.get("apply_log", True)),
            num_iter=int(recon.get("num_iter", 100)),
            chunk_size=int(recon.get("chunk_size", 32)),
            center_mode=str(center_mode),
            center=None if center is None else float(center),
            pixel_shift=float(recon.get("pixel_shift", 0.0) or 0.0),
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
    """Return effective COR = base (auto/manual) + pixel_shift."""
    _, _, effective = resolve_center(sino, settings, width)
    return effective


def resolve_center(sino: np.ndarray, settings: Settings, width: int) -> Tuple[float, float, float]:
    """
    Returns (base_center, pixel_shift, effective_center).
    pixel_shift is the NRecon-style postalignment nudge (can be fractional).
    """
    if settings.center_mode == "manual" and settings.center is not None:
        base = float(settings.center)
    else:
        import algotom.prep.calculation as calc

        try:
            base = float(calc.find_center_vo(sino))
        except Exception:
            base = (width - 1) / 2.0
    shift = float(settings.pixel_shift or 0.0)
    return base, shift, base + shift


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


def cone_geometry_from_log(meta: BrukerLogMeta) -> Dict[str, float]:
    """Build Astra cone distances (in detector-pixel units) from Bruker .log fields."""
    sod_mm = float(meta.object_to_source_mm)
    sdd_mm = float(meta.camera_to_source_mm)
    if sod_mm <= 0 or sdd_mm <= 0:
        raise ValueError(
            "FDK needs Object to Source (mm) and Camera to Source (mm) in the .log file."
        )
    odd_mm = sdd_mm - sod_mm
    if odd_mm <= 0:
        raise ValueError(f"Invalid geometry: Camera to Source ({sdd_mm}) must be > Object to Source ({sod_mm}).")
    pix_obj_mm = float(meta.image_pixel_size_um) / 1000.0
    if pix_obj_mm <= 0:
        raise ValueError("FDK needs Image Pixel Size (um) in the .log file.")
    mag = sdd_mm / sod_mm
    pix_det_mm = pix_obj_mm * mag
    return {
        "sod_mm": sod_mm,
        "odd_mm": odd_mm,
        "sdd_mm": sdd_mm,
        "mag": mag,
        "pix_obj_mm": pix_obj_mm,
        "pix_det_mm": pix_det_mm,
        "sod_pix": sod_mm / pix_det_mm,
        "odd_pix": odd_mm / pix_det_mm,
    }


def prepare_projections_for_fdk(
    projections: np.ndarray,
    center: float,
    settings: Settings,
    apply_rings: bool,
    progress: ProgressCb = None,
) -> np.ndarray:
    """projections: (n_angles, rows, cols) → logged, COR-shifted, optionally ring-cleaned."""
    data = np.asarray(projections, dtype=np.float32).copy()
    if settings.apply_log:
        data = np.maximum(data, 1e-6)
        data = -np.log(data)

    # Sub-pixel shift so geometric mid matches effective COR (NRecon postalignment)
    _, _, cols = data.shape
    shift = ((cols - 1) / 2.0) - float(center)
    if abs(shift) > 1e-6:
        _log(f"FDK COR shift: {shift:.3f} px", progress)
        try:
            from scipy.ndimage import shift as ndi_shift

            data = ndi_shift(data, shift=(0.0, 0.0, shift), order=1, mode="nearest")
        except Exception:
            data = np.roll(data, int(round(shift)), axis=2)

    if apply_rings and settings.ring_enable and settings.ring_method != "none":
        n_rows = data.shape[1]
        _log(f"Applying ring removal to {n_rows} detector rows (FDK)...", progress)
        for r in range(n_rows):
            try:
                data[:, r, :] = apply_ring_removal(data[:, r, :], settings)
            except Exception as exc:
                _log(f"Ring fail row {r}: {exc}", progress)
            if r % 100 == 0 or r == n_rows - 1:
                _log(f"  ring rows {r + 1}/{n_rows}", progress)
    return data


def reconstruct_fdk_volume(
    projections: np.ndarray,
    thetas: np.ndarray,
    meta: BrukerLogMeta,
    settings: Settings,
    progress: ProgressCb = None,
) -> np.ndarray:
    """
    Cone-beam FDK via Astra.
    Input projections already prepared (log + COR shift + optional rings),
    shape (n_angles, det_rows, det_cols).
    Returns volume shaped (det_rows, det_cols, det_cols) roughly (z, y, x).
    """
    import astra

    n_angles, det_rows, det_cols = projections.shape
    geom = cone_geometry_from_log(meta)
    _log(
        f"FDK geometry: SOD={geom['sod_mm']:.3f} mm  ODD={geom['odd_mm']:.3f} mm  "
        f"mag={geom['mag']:.3f}  pix_det={geom['pix_det_mm']*1000:.3f} um",
        progress,
    )

    proj_geom = astra.create_proj_geom(
        "cone",
        1.0,
        1.0,
        det_rows,
        det_cols,
        thetas,
        geom["sod_pix"],
        geom["odd_pix"],
    )
    vol_geom = astra.create_vol_geom(det_cols, det_cols, det_rows)

    proj_id = astra.data3d.create("-sino", proj_geom, projections)
    vol_id = astra.data3d.create("-vol", vol_geom)
    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = vol_id
    ftype = settings.filter_name.lower().replace("_", "-")
    if ftype in ("ramlak", "ram-lak"):
        ftype = "ram-lak"
    cfg["option"] = {"FilterType": ftype}
    try:
        alg_id = astra.algorithm.create(cfg)
    except Exception:
        cfg.pop("option", None)
        alg_id = astra.algorithm.create(cfg)
    try:
        _log("Running FDK_CUDA...", progress)
        astra.algorithm.run(alg_id)
        vol = astra.data3d.get(vol_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    return np.asarray(vol, dtype=np.float32)


def extract_fdk_slice(volume: np.ndarray, row: int) -> np.ndarray:
    """Pick axial slice nearest to detector row index from FDK volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D FDK volume, got shape {volume.shape}")
    z = volume.shape[0]
    idx = int(np.clip(row, 0, z - 1))
    return np.asarray(volume[idx], dtype=np.float32)


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
    base_center: float
    pixel_shift: float
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
    before_reused: bool = False
    history_dir: Optional[Path] = None
    before_key: str = ""


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
    cached_before: Optional[np.ndarray] = None,
    cached_before_key: str = "",
) -> PreviewResult:
    """
    Preview mid (or chosen) slice.

    Efficiency:
    - If rings are OFF: only ONE reconstruction (before == after).
    - If cached_before matches geometry/align key: reuse BEFORE, only rebuild AFTER.
    - Otherwise builds BEFORE once, then AFTER with rings.
    """
    from algotom.io import loadersaver as losa

    from history_store import before_cache_key, save_history_entry

    scan_dir, log_path, meta, proj_paths, height, width = _prepare_scan(scan_dir, progress)
    mid = height // 2
    row = mid if settings.preview_row is None else int(settings.preview_row)
    if row < 0 or row >= height:
        raise ValueError(f"preview row {row} out of range 0..{height - 1}")

    out = Path(out_dir) if out_dir else default_output_dir(scan_dir, settings.output_dir, True)
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    recon_type = (settings.recon_type or "FBP").upper()
    rings_on = bool(settings.ring_enable) and settings.ring_method != "none"
    _log(f"Preview mode={recon_type}  row={row}/{height - 1}  rings={'ON' if rings_on else 'OFF'}", progress)

    before_reused = False
    projections = None
    sino = None

    if recon_type == "FDK":
        _log("FDK preview loads the full projection stack...", progress)
        projections = load_stack(proj_paths, progress)
        n_angles = projections.shape[0]
        thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))
        base_c, shift_c, center = resolve_center(projections[:, mid, :], settings, width)
        key = before_cache_key(
            str(scan_dir), row, center, recon_type, "FDK_CUDA", settings.apply_log, settings.filter_name
        )
        _log(f"COR base={base_c:.3f}  shift={shift_c:+.3f}  effective={center:.3f}", progress)

        if cached_before is not None and cached_before_key == key:
            img_raw = np.asarray(cached_before, dtype=np.float32)
            before_reused = True
            _log("Reusing cached BEFORE (skipped duplicate FDK)", progress)
        else:
            raw_prep = prepare_projections_for_fdk(
                projections, center, settings, apply_rings=False, progress=progress
            )
            vol_raw = reconstruct_fdk_volume(raw_prep, thetas, meta, settings, progress)
            img_raw = extract_fdk_slice(vol_raw, row)

        if rings_on:
            corr_prep = prepare_projections_for_fdk(
                projections, center, settings, apply_rings=True, progress=progress
            )
            vol_corr = reconstruct_fdk_volume(corr_prep, thetas, meta, settings, progress)
            img_corr = extract_fdk_slice(vol_corr, row)
        else:
            img_corr = img_raw
            _log("Rings OFF — single reconstruction only", progress)
    else:
        sino = load_sinogram_row(proj_paths, row, progress)
        n_angles = sino.shape[0]
        thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))
        base_c, shift_c, center = resolve_center(sino, settings, width)
        key = before_cache_key(
            str(scan_dir), row, center, recon_type, settings.method, settings.apply_log, settings.filter_name
        )
        _log(f"COR base={base_c:.3f}  shift={shift_c:+.3f}  effective={center:.3f}", progress)

        if cached_before is not None and cached_before_key == key:
            img_raw = np.asarray(cached_before, dtype=np.float32)
            before_reused = True
            _log("Reusing cached BEFORE (skipped duplicate FBP)", progress)
        else:
            img_raw = reconstruct_sinogram(sino.copy(), center, thetas, settings)

        if rings_on:
            try:
                sino_corr = apply_ring_removal(sino.copy(), settings)
            except Exception as exc:
                _log(f"Ring removal failed: {exc}", progress)
                sino_corr = sino
            img_corr = reconstruct_sinogram(sino_corr, center, thetas, settings)
        else:
            img_corr = img_raw
            _log("Rings OFF — single reconstruction only", progress)

    losa.save_image(str(qc / "preview_raw.tif"), img_raw)
    losa.save_image(str(qc / "preview_corrected.tif"), img_corr)
    if settings.save_preview:
        save_qc_png(qc / "before.png", img_raw, f"BEFORE  {recon_type} row={row}")
        save_qc_png(
            qc / "after.png",
            img_corr,
            f"AFTER  {recon_type} row={row}  c={center:.2f}  shift={shift_c:+.2f}",
        )

    used = deepcopy(settings)
    used.center = center
    used.pixel_shift = shift_c
    used.preview_row = row
    run_info = {
        "mode": "preview",
        "recon_type": recon_type,
        "preview_row": row,
        "before_reused": before_reused,
        "scan_dir": str(scan_dir),
        "log_file": str(log_path),
        "n_projections": n_angles,
        "volume_shape_hw": [height, width],
        "base_center": base_c,
        "pixel_shift": shift_c,
        "center": center,
        "config": used.to_config_dict(),
    }
    save_yaml(out / "run_config.yaml", run_info)

    hist = save_history_entry(
        scan_dir,
        kind="preview",
        settings_dict=used.to_config_dict(),
        images={"before": img_raw, "after": img_corr},
        extra=f"before_reused={before_reused}  row={row}",
    )

    msg = (
        f"Preview OK ({recon_type}). base={base_c:.3f} shift={shift_c:+.3f} "
        f"effective={center:.3f}. before_reused={before_reused}. history={hist.name}"
    )
    _log(msg, progress)
    return PreviewResult(
        out_dir=out,
        center=center,
        base_center=base_c,
        pixel_shift=shift_c,
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
        before_reused=before_reused,
        history_dir=hist,
        before_key=key,
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

    recon_type = (settings.recon_type or "FBP").upper()
    _log(f"Full recon mode={recon_type}", progress)
    _log("Loading full projection stack...", progress)
    projections = load_stack(proj_paths, progress)
    n_angles = projections.shape[0]
    thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))
    center = find_center(projections[:, mid, :], settings, width)
    base_c, shift_c, _ = resolve_center(projections[:, mid, :], settings, width)
    _log(
        f"COR base={base_c:.3f}  pixel_shift={shift_c:.3f}  effective={center:.3f}",
        progress,
    )

    mid_img = None
    if recon_type == "FDK":
        prep = prepare_projections_for_fdk(
            projections, center, settings, apply_rings=True, progress=progress
        )
        volume = reconstruct_fdk_volume(prep, thetas, meta, settings, progress)
        # Save each z-slice
        n_slices = volume.shape[0]
        for r in range(n_slices):
            img = np.asarray(volume[r], dtype=np.float32)
            losa.save_image(str(slices_dir / f"recon_{r:05d}.tif"), img)
            if r == mid or (r == n_slices // 2 and mid_img is None):
                mid_img = img
            if r % 50 == 0 or r == n_slices - 1:
                _log(f"Saving FDK slices {r + 1}/{n_slices}", progress)
        if mid_img is None:
            mid_img = extract_fdk_slice(volume, mid)
    else:
        chunk = max(1, int(settings.chunk_size))
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
        save_qc_png(qc / "after.png", mid_img, f"FULL {recon_type} mid c={center:.2f}")

    used = deepcopy(settings)
    used.center = center
    run_info = {
        "mode": "full",
        "recon_type": recon_type,
        "scan_dir": str(scan_dir),
        "log_file": str(log_path),
        "n_projections": n_angles,
        "volume_shape_hw": [height, width],
        "center": center,
        "config": used.to_config_dict(),
    }
    save_yaml(out / "run_config.yaml", run_info)
    msg = f"Full recon done ({recon_type}). Slices: {slices_dir}"
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


@dataclass
class AlignCache:
    """Cached mid-row sinogram for sub-second alignment nudges."""

    scan_dir: str
    row: int
    sino: Any  # np.ndarray
    thetas: Any  # np.ndarray
    width: int
    height: int
    base_center: float
    log_postalignment: float
    n_projections: int


def _sharpness_score(img: np.ndarray) -> float:
    """Higher = sharper. Used to auto-pick pixel shift."""
    x = np.asarray(img, dtype=np.float64)
    # robust normalize
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return 0.0
    x = (x - lo) / (hi - lo)
    # Laplacian variance
    lap = (
        -4.0 * x
        + np.roll(x, 1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, 1, 1)
        + np.roll(x, -1, 1)
    )
    return float(lap.var())


def prepare_align_cache(
    scan_dir: Path,
    preview_row: Optional[int] = None,
    progress: ProgressCb = None,
) -> AlignCache:
    """
    One-time load of a single detector row for fast alignment.
    After this, quick_align_preview() only re-runs FBP (no disk I/O).
    """
    scan_dir, log_path, meta, proj_paths, height, width = _prepare_scan(Path(scan_dir), progress)
    mid = height // 2
    row = mid if preview_row is None or preview_row < 0 else int(preview_row)
    if row < 0 or row >= height:
        raise ValueError(f"align row {row} out of range 0..{height - 1}")

    _log(f"Caching row {row} for quick alignment (one-time load)...", progress)
    sino = load_sinogram_row(proj_paths, row, progress)
    n_angles = sino.shape[0]
    thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, n_angles), dtype=np.float64))

    # Base COR with zero shift
    tmp = Settings(center_mode="auto", pixel_shift=0.0)
    base, _, _ = resolve_center(sino, tmp, width)
    post = float(meta.postalignment or 0.0)
    _log(
        f"Align cache ready: base_COR={base:.3f}  log_Postalignment={post:.3f}  "
        f"n={n_angles}  row={row}",
        progress,
    )
    return AlignCache(
        scan_dir=str(Path(scan_dir).resolve()),
        row=row,
        sino=sino,
        thetas=thetas,
        width=width,
        height=height,
        base_center=base,
        log_postalignment=post,
        n_projections=n_angles,
    )


def quick_align_preview(
    cache: AlignCache,
    pixel_shift: float,
    apply_log: bool = True,
    save_history: bool = True,
) -> Tuple[np.ndarray, str, float, float]:
    """Fast FBP preview from cached sinogram (no rings, no disk)."""
    shift = float(pixel_shift or 0.0)
    center = float(cache.base_center) + shift
    settings = Settings(
        recon_type="FBP",
        method="FBP_CUDA",
        filter_name="hann",
        apply_log=bool(apply_log),
        ring_enable=False,
        ring_method="none",
        center_mode="manual",
        center=center,
        pixel_shift=shift,
    )
    try:
        img = reconstruct_sinogram(np.asarray(cache.sino, dtype=np.float32).copy(), center, cache.thetas, settings)
    except Exception:
        settings.method = "FBP"
        img = reconstruct_sinogram(np.asarray(cache.sino, dtype=np.float32).copy(), center, cache.thetas, settings)

    if save_history:
        from history_store import save_history_entry

        save_history_entry(
            Path(cache.scan_dir),
            kind="align",
            settings_dict=settings.to_config_dict(),
            images={"align": img},
            extra=f"row={cache.row}  base={cache.base_center:.3f}",
        )

    msg = (
        f"QUICK ALIGN | base={cache.base_center:.3f}  shift={shift:+.3f}  "
        f"effective={center:.3f}  row={cache.row}  (cached, FBP, no rings)"
    )
    return _norm_display(img), msg, float(cache.base_center), float(center)


def auto_tune_pixel_shift(
    cache: AlignCache,
    search: float = 2.0,
    step: float = 0.25,
    apply_log: bool = True,
    progress: ProgressCb = None,
) -> Tuple[float, np.ndarray, str, float, float]:
    """
    Try shifts around 0 (and favor log postalignment) — pick sharpest slice.
    Typically a few seconds on GPU.
    """
    candidates = list(np.arange(-search, search + 1e-9, step))
    # Always include log postalignment and 0
    for extra in (0.0, float(cache.log_postalignment), -float(cache.log_postalignment)):
        if all(abs(extra - c) > 1e-9 for c in candidates):
            candidates.append(extra)
    candidates = sorted(set(round(float(c), 3) for c in candidates))

    best_shift = float(cache.log_postalignment or 0.0)
    best_score = -1.0
    best_img = None
    _log(f"Auto-tuning pixel shift over {len(candidates)} trials...", progress)
    for i, shift in enumerate(candidates):
        center = cache.base_center + float(shift)
        settings = Settings(
            method="FBP_CUDA",
            apply_log=apply_log,
            ring_enable=False,
            center_mode="manual",
            center=center,
            pixel_shift=0.0,
        )
        try:
            img = reconstruct_sinogram(
                np.asarray(cache.sino, dtype=np.float32).copy(), center, cache.thetas, settings
            )
        except Exception:
            settings.method = "FBP"
            img = reconstruct_sinogram(
                np.asarray(cache.sino, dtype=np.float32).copy(), center, cache.thetas, settings
            )
        score = _sharpness_score(img)
        if score > best_score:
            best_score = score
            best_shift = float(shift)
            best_img = _norm_display(img)
        if i % 4 == 0:
            _log(f"  trial {i + 1}/{len(candidates)} shift={shift:+.2f} score={score:.4g}", progress)

    if best_img is None:
        best_img, msg, base, eff = quick_align_preview(
            cache, best_shift, apply_log=apply_log, save_history=True
        )
    else:
        from history_store import save_history_entry

        settings = Settings(
            recon_type="FBP",
            method="FBP_CUDA",
            apply_log=apply_log,
            ring_enable=False,
            center_mode="manual",
            center=cache.base_center + best_shift,
            pixel_shift=best_shift,
        )
        save_history_entry(
            Path(cache.scan_dir),
            kind="align_autotune",
            settings_dict=settings.to_config_dict(),
            images={"align": best_img},
            extra=f"score={best_score:.4g}  row={cache.row}",
        )
        msg = (
            f"QUICK ALIGN | base={cache.base_center:.3f}  shift={best_shift:+.3f}  "
            f"effective={cache.base_center + best_shift:.3f}  row={cache.row}"
        )
        base = float(cache.base_center)
        eff = float(cache.base_center + best_shift)
    out = (
        f"AUTO-TUNE done | best_shift={best_shift:+.3f}  score={best_score:.4g}  "
        f"(log suggested {cache.log_postalignment:+.3f})\n{msg}"
    )
    _log(out, progress)
    return best_shift, best_img, out, base, eff


def probe_scan_info(scan_dir: str) -> Tuple[str, int, int, int, float]:
    """Return (status_text, height, width, mid_row, log_postalignment)."""
    scan_dir_p, log_path, meta, proj_paths, height, width = _prepare_scan(Path(scan_dir))
    mid = height // 2
    post = float(meta.postalignment or 0.0)
    geom_txt = ""
    try:
        g = cone_geometry_from_log(meta)
        geom_txt = (
            f" | SOD={g['sod_mm']:.2f}mm ODD={g['odd_mm']:.2f}mm "
            f"pix={meta.image_pixel_size_um:.3f}um (FDK ready)"
        )
    except Exception as exc:
        geom_txt = f" | FDK geometry incomplete: {exc}"
    post_txt = f" | NRecon Postalignment={post:+.3f}"
    text = (
        f"OK — {len(proj_paths)} projections | detector {height}×{width} | "
        f"log={log_path.name} | prefix={meta.filename_prefix!r} | "
        f"rot_step={meta.rotation_step_deg}° | mid_row={mid}{post_txt}{geom_txt}"
    )
    return text, height, width, mid, post
