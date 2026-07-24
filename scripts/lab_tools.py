"""
Lab helpers: preflight, multi-row align check, ring-method bake-off.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from parse_bruker_log import estimate_angles_deg, parse_bruker_log
from qc_metrics import ring_score, sharpness_score
from recon_core import (
    RING_METHODS,
    ProgressCb,
    Settings,
    _log,
    _norm_display,
    _prepare_scan,
    apply_ring_removal,
    default_output_dir,
    find_log_file,
    list_projections,
    load_sinogram_row,
    probe_shape,
    reconstruct_sinogram,
    resolve_center,
)

Progress = ProgressCb


@dataclass
class PreflightReport:
    ok: bool
    lines: List[str]
    suggested_out: str

    @property
    def text(self) -> str:
        head = "PREFLIGHT PASS" if self.ok else "PREFLIGHT WARNINGS"
        return head + "\n" + "\n".join(self.lines) + f"\nSuggested output: {self.suggested_out}"


def versioned_full_output_dir(scan_dir: Path, settings: Settings) -> Path:
    """Never overwrite a previous full recon — always stamp a new folder under algotom/."""
    if settings.output_dir:
        base = default_output_dir(Path(scan_dir), settings.output_dir, preview=False)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base.parent / f"{base.name}_{stamp}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(scan_dir) / "algotom" / f"recon_{stamp}"


def _probe_nvidia() -> str:
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            errors="ignore",
            timeout=8,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return ""


def preflight_scan(scan_dir: str, settings: Settings, progress: Progress = None) -> PreflightReport:
    scan_dir_p = Path((scan_dir or "").strip().strip('"'))
    lines: List[str] = []
    ok = True

    if not scan_dir_p.is_dir():
        return PreflightReport(False, [f"Scan folder missing: {scan_dir_p}"], "")

    try:
        log_path = find_log_file(scan_dir_p)
        meta = parse_bruker_log(log_path)
        lines.append(f"Log OK: {log_path.name}")
    except Exception as exc:
        return PreflightReport(False, [f"Log problem: {exc}"], "")

    try:
        paths = list_projections(scan_dir_p, prefix=meta.filename_prefix)
        h, w = probe_shape(paths)
        lines.append(f"Projections found: {len(paths)}  detector {h}x{w}")
    except Exception as exc:
        return PreflightReport(False, [f"Projection problem: {exc}"], "")

    if meta.number_of_files and abs(int(meta.number_of_files) - len(paths)) > 2:
        ok = False
        lines.append(
            f"WARN: log NumberOfFiles={meta.number_of_files} but found {len(paths)} TIFFs"
        )
    else:
        lines.append("Projection count matches log (or log count missing)")

    bytes_est = len(paths) * h * w * 4
    gb = bytes_est / (1024**3)
    lines.append(f"Est. RAM for full stack: ~{gb:.1f} GB (float32)")
    if gb > 90:
        ok = False
        lines.append("WARN: stack may exceed ~90 GB — consider a smaller scan / later ROI tools")

    rtype = (settings.recon_type or "FBP").upper()
    lines.append(f"Requested mode: {rtype}  method={settings.method}")
    if rtype == "FDK":
        missing = []
        if not meta.object_to_source_mm:
            missing.append("Object to Source")
        if not meta.camera_to_source_mm:
            missing.append("Camera to Source")
        if not meta.image_pixel_size_um:
            missing.append("Image Pixel Size")
        if missing:
            ok = False
            lines.append("WARN FDK missing log fields: " + ", ".join(missing))
        else:
            lines.append("FDK geometry fields present in log")
        lines.append("NOTE: FDK preview/full loads the entire stack (GPU + RAM heavy)")

    gpu = _probe_nvidia()
    if gpu:
        lines.append(f"GPU: {gpu}")
    else:
        if "CUDA" in (settings.method or "").upper() or rtype == "FDK":
            ok = False
            lines.append("WARN: nvidia-smi not found — CUDA/FDK may fall back or fail")
        else:
            lines.append("GPU: nvidia-smi not found (CPU FBP may still work)")

    out = versioned_full_output_dir(scan_dir_p, settings)
    lines.append(f"Full recon will write a NEW folder (no overwrite): {out.name}")
    lines.append(
        f"Rings: {'ON' if settings.ring_enable and settings.ring_method != 'none' else 'OFF'} "
        f"({settings.ring_method})  shift={settings.pixel_shift:+.3f}"
    )
    _log(lines[-1], progress)
    return PreflightReport(ok=ok, lines=lines, suggested_out=str(out))


def validate_alignment_rows(
    scan_dir: Path,
    pixel_shift: float,
    apply_log: bool = True,
    progress: Progress = None,
) -> Tuple[np.ndarray, str, float]:
    """Check current shift on 3 rows. Returns montage, report, recommended shift."""
    scan_dir, _, meta, proj_paths, height, width = _prepare_scan(Path(scan_dir), progress)
    mid = height // 2
    rows = sorted({max(0, mid - height // 4), mid, min(height - 1, mid + height // 4)})
    shift0 = float(pixel_shift or 0.0)
    candidates = [round(shift0 + d, 3) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    best_by_row: Dict[int, Tuple[float, float, np.ndarray]] = {}

    for row in rows:
        _log(f"Align check row {row}...", progress)
        sino = load_sinogram_row(proj_paths, row, progress)
        thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, sino.shape[0]), dtype=np.float64))
        base, _, _ = resolve_center(sino, Settings(center_mode="auto", pixel_shift=0.0), width)
        best_s, best_score, best_img = shift0, -1.0, None
        for sh in candidates:
            center = base + sh
            settings = Settings(
                recon_type="FBP",
                method="FBP_CUDA",
                filter_name="hann",
                apply_log=bool(apply_log),
                ring_enable=False,
                center_mode="manual",
                center=center,
                pixel_shift=sh,
            )
            try:
                img = reconstruct_sinogram(sino.copy(), center, thetas, settings)
            except Exception:
                settings.method = "FBP"
                img = reconstruct_sinogram(sino.copy(), center, thetas, settings)
            sc = sharpness_score(img)
            if sc > best_score:
                best_s, best_score, best_img = sh, sc, img
        assert best_img is not None
        best_by_row[row] = (best_s, best_score, best_img)

    shifts = [best_by_row[r][0] for r in rows]
    spread = max(shifts) - min(shifts)
    recommended = float(np.median(shifts))
    if spread <= 0.5:
        confidence = "HIGH — rows agree"
    elif spread <= 1.0:
        confidence = "MEDIUM — mild disagreement"
    else:
        confidence = "LOW — check geometry / try auto-tune per region"

    tiles = [_norm_display(best_by_row[r][2]) for r in rows]
    montage = np.concatenate(tiles, axis=1)
    msg = (
        f"MULTI-ROW ALIGN | current={shift0:+.3f}  recommended={recommended:+.3f}  "
        f"per-row best={shifts}  spread={spread:.2f} px  confidence={confidence}"
    )
    _log(msg, progress)

    from history_store import save_history_entry

    save_history_entry(
        Path(scan_dir),
        kind="align_check",
        settings_dict=Settings(pixel_shift=recommended).to_config_dict(),
        images={"align": montage},
        extra=msg,
    )
    return montage, msg, recommended


def compare_ring_methods(
    scan_dir: Path,
    settings: Settings,
    progress: Progress = None,
) -> Tuple[List[Tuple[np.ndarray, str]], str, Settings]:
    """Fast FBP bake-off on one row. Returns gallery, report, winning settings."""
    scan_dir, _, meta, proj_paths, height, width = _prepare_scan(Path(scan_dir), progress)
    mid = height // 2
    row = mid if settings.preview_row is None or settings.preview_row < 0 else int(settings.preview_row)
    _log(f"Ring compare on row {row} (FBP only)...", progress)
    sino = load_sinogram_row(proj_paths, row, progress)
    thetas = np.deg2rad(np.asarray(estimate_angles_deg(meta, sino.shape[0]), dtype=np.float64))
    base, shift, center = resolve_center(sino, settings, width)

    methods = ["none"] + [m for m in RING_METHODS if m != "none"]
    results: List[Tuple[str, float, float, np.ndarray]] = []
    for method in methods:
        s = deepcopy(settings)
        s.recon_type = "FBP"
        if "CUDA" not in (s.method or "").upper():
            s.method = "FBP_CUDA"
        s.ring_enable = method != "none"
        s.ring_method = method
        work = sino.copy()
        if method != "none":
            try:
                work = apply_ring_removal(work, s)
            except Exception as exc:
                _log(f"{method} failed: {exc}", progress)
                continue
        try:
            img = reconstruct_sinogram(work, center, thetas, s)
        except Exception:
            s.method = "FBP"
            img = reconstruct_sinogram(work, center, thetas, s)
        rs = ring_score(img)
        sh = sharpness_score(img)
        results.append((method, rs, sh, img))
        _log(f"  {method}: ring={rs:.4g} sharp={sh:.4g}", progress)

    if not results:
        raise RuntimeError("Ring compare produced no results")

    ranked = sorted(results, key=lambda t: (t[1], -t[2]))
    winner = ranked[0][0]
    lines = [
        f"RING COMPARE row={row} center={center:.3f} shift={shift:+.3f}",
        f"Winner (lowest ring score): {winner}",
    ]
    gallery: List[Tuple[np.ndarray, str]] = []
    for method, rs, sh, img in results:
        mark = " *" if method == winner else ""
        cap = f"{method}{mark}\nring={rs:.4g}  sharp={sh:.4g}"
        lines.append(f"  {method}: ring={rs:.4g} sharp={sh:.4g}")
        gallery.append((_norm_display(img), cap))

    from history_store import save_history_entry

    save_history_entry(
        Path(scan_dir),
        kind="ring_compare",
        settings_dict=settings.to_config_dict(),
        images={f"m_{i}_{method}": img for i, (method, _, _, img) in enumerate(results)},
        extra=f"winner={winner}",
    )

    win_settings = deepcopy(settings)
    if winner == "none":
        win_settings.ring_enable = False
        win_settings.ring_method = "none"
    else:
        win_settings.ring_enable = True
        win_settings.ring_method = winner
    return gallery, "\n".join(lines), win_settings
