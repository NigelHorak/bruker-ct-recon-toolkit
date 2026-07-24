"""
NRecon-style parameter sweeps: generate options, user picks the best.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qc_metrics import ring_score, sharpness_score
from recon_core import (
    AlignCache,
    ProgressCb,
    Settings,
    _log,
    _norm_display,
    _sharpness_score,
    apply_ring_removal,
    quick_align_preview,
    reconstruct_sinogram,
)


@dataclass
class Candidate:
    label: str
    image: np.ndarray
    payload: Dict[str, Any]  # values to apply when user clicks Use


def _arange_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Step must be > 0")
    if stop < start:
        start, stop = stop, start
    vals = []
    x = float(start)
    # inclusive end within one step tolerance
    while x <= stop + abs(step) * 1e-9:
        vals.append(round(x, 4))
        x += step
    # de-dupe
    out: List[float] = []
    for v in vals:
        if not out or abs(out[-1] - v) > 1e-9:
            out.append(v)
    if len(out) > 40:
        raise ValueError(f"Too many options ({len(out)}). Widen the step or narrow the range (max 40).")
    if not out:
        raise ValueError("No values in that range.")
    return out


def sweep_alignment(
    cache: AlignCache,
    start: float,
    stop: float,
    step: float,
    apply_log: bool = True,
    progress: ProgressCb = None,
) -> List[Candidate]:
    from preview_cache import (
        format_params_log,
        load_cached_preview,
        preview_params_key,
        save_cached_preview,
    )

    values = _arange_inclusive(float(start), float(stop), float(step))
    _log(f"Alignment sweep: {len(values)} trials from {values[0]} to {values[-1]}", progress)
    out: List[Candidate] = []
    scan = Path(cache.scan_dir)
    for i, shift in enumerate(values):
        params = preview_params_key(
            scan,
            kind="align",
            row=int(cache.row),
            pixel_shift=float(shift),
            ring_enable=False,
            ring_method="none",
            snr=0.0,
            la_size=0,
            sm_size=0,
            drop_ratio=0.0,
            dim=1,
            apply_log=bool(apply_log),
        )
        hit = load_cached_preview(scan, params)
        if hit is not None:
            img, _folder = hit
            _log(f"recon already exists ({format_params_log(params)})", progress)
            base = float(cache.base_center)
            eff = base + float(shift)
        else:
            img, _msg, base, eff = quick_align_preview(
                cache, float(shift), apply_log=bool(apply_log), save_history=False
            )
            save_cached_preview(scan, params, img)
            _log(f"  [{i + 1}/{len(values)}] shift={shift:+.2f} saved", progress)
        score = _sharpness_score(img)
        out.append(
            Candidate(
                label=f"shift {shift:+.2f}\nsharp={score:.3g}",
                image=img,
                payload={"pixel_shift": float(shift), "base": float(base), "effective": float(eff)},
            )
        )
    return out


def sweep_ring_recipes(
    cache: AlignCache,
    pixel_shift: float,
    snr: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    apply_log: bool = True,
    progress: ProgressCb = None,
) -> List[Candidate]:
    """Fixed set of ring recipes at the current alignment (user picks)."""
    methods = [
        ("Off (no cleanup)", "none"),
        ("All rings", "remove_all_stripe"),
        ("Fine rings", "remove_stripe_based_sorting"),
        ("Large rings", "remove_large_stripe"),
    ]
    shift = float(pixel_shift or 0.0)
    center = float(cache.base_center) + shift
    thetas = cache.thetas
    sino = np.asarray(cache.sino, dtype=np.float32)
    out: List[Candidate] = []
    scan = Path(cache.scan_dir)
    from preview_cache import (
        format_params_log,
        load_cached_preview,
        preview_params_key,
        save_cached_preview,
    )

    _log(f"Ring recipes at shift={shift:+.3f} ({len(methods)} options)", progress)
    for label, method in methods:
        la = int(la_size) if int(la_size) % 2 == 1 else int(la_size) + 1
        sm = int(sm_size) if int(sm_size) % 2 == 1 else int(sm_size) + 1
        params = preview_params_key(
            scan,
            kind="ring",
            row=int(cache.row),
            pixel_shift=shift,
            ring_enable=method != "none",
            ring_method=method,
            snr=float(snr),
            la_size=la,
            sm_size=sm,
            drop_ratio=float(drop_ratio),
            dim=int(dim),
            apply_log=bool(apply_log),
        )
        hit = load_cached_preview(scan, params)
        if hit is not None:
            disp, _folder = hit
            _log(f"recon already exists ({format_params_log(params)})", progress)
            img_for_score = disp
        else:
            s = Settings(
                recon_type="FBP",
                method="FBP_CUDA",
                filter_name="hann",
                apply_log=bool(apply_log),
                ring_enable=method != "none",
                ring_method=method,
                snr=float(snr),
                la_size=la,
                sm_size=sm,
                drop_ratio=float(drop_ratio),
                dim=int(dim),
                center_mode="manual",
                center=center,
                pixel_shift=shift,
            )
            work = sino.copy()
            if method != "none":
                try:
                    work = apply_ring_removal(work, s)
                except Exception as exc:
                    _log(f"  {method} failed: {exc}", progress)
                    continue
            try:
                img = reconstruct_sinogram(work, center, thetas, s)
            except Exception:
                s.method = "FBP"
                img = reconstruct_sinogram(work, center, thetas, s)
            disp = _norm_display(img)
            save_cached_preview(scan, params, disp)
            img_for_score = img
            _log(f"  {label}: saved", progress)
        rs = ring_score(img_for_score)
        sh = sharpness_score(img_for_score)
        out.append(
            Candidate(
                label=f"{label}\nring={rs:.3g} sharp={sh:.3g}",
                image=disp,
                payload={
                    "ring_method": method,
                    "ring_enable": method != "none",
                    "snr": float(snr),
                    "la_size": la,
                    "sm_size": sm,
                    "drop_ratio": float(drop_ratio),
                    "dim": int(dim),
                },
            )
        )
    return out


def sweep_ring_strength(
    cache: AlignCache,
    pixel_shift: float,
    method: str,
    snr_start: float,
    snr_stop: float,
    snr_step: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    apply_log: bool = True,
    progress: ProgressCb = None,
) -> List[Candidate]:
    values = _arange_inclusive(float(snr_start), float(snr_stop), float(snr_step))
    shift = float(pixel_shift or 0.0)
    center = float(cache.base_center) + shift
    out: List[Candidate] = []
    _log(f"Ring strength sweep method={method} snr={values}", progress)
    for snr in values:
        cands = sweep_ring_recipes(
            cache,
            pixel_shift=shift,
            snr=snr,
            la_size=la_size,
            sm_size=sm_size,
            drop_ratio=drop_ratio,
            dim=dim,
            apply_log=apply_log,
            progress=None,
        )
        # pick matching method from recipes
        match = next((c for c in cands if c.payload.get("ring_method") == method), None)
        if match is None and cands:
            match = cands[0]
        if match is None:
            continue
        match.label = f"snr={snr:.1f}\n{match.label}"
        match.payload["snr"] = float(snr)
        out.append(match)
        _log(f"  snr={snr:.1f} done", progress)
    return out


def apply_simple_beam_hardening(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Simple display/post BH-style remap (not full physics).
    strength 0 = unchanged; higher = more correction curve.
    """
    x = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return _norm_display(x)
    n = np.clip((x - lo) / (hi - lo), 0, 1)
    s = max(0.0, float(strength))
    # gamma-like + soft polynomial lift of darks
    gamma = 1.0 / (1.0 + 0.35 * s)
    y = n**gamma
    y = y + s * 0.08 * (y * (1.0 - y))
    y = np.clip(y, 0, 1)
    return (y * 255.0).astype(np.uint8)


def sweep_beam_hardening(
    base_img: np.ndarray,
    start: float,
    stop: float,
    step: float,
    progress: ProgressCb = None,
) -> List[Candidate]:
    values = _arange_inclusive(float(start), float(stop), float(step))
    _log(f"Beam-hardening sweep: {values}", progress)
    out: List[Candidate] = []
    for v in values:
        img = apply_simple_beam_hardening(base_img, v)
        out.append(
            Candidate(
                label=f"BH strength {v:.2f}",
                image=img,
                payload={"bh_strength": float(v)},
            )
        )
    return out


def candidates_to_gallery(cands: List[Candidate]) -> List[Tuple[np.ndarray, str]]:
    return [(c.image, c.label) for c in cands]
