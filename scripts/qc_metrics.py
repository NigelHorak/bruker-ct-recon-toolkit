"""
Ring / comparison QC helpers for preview slices.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float64)


def ring_score(img: np.ndarray) -> float:
    """
    Higher = more ring-like concentric structure.
    Polar angular variance of a high-pass residual (no SciPy).
    """
    x = _to_float(img)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return 0.0
    x = (x - lo) / (hi - lo)
    nbr = (
        x
        + np.roll(x, 1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, 1, 1)
        + np.roll(x, -1, 1)
    ) / 5.0
    hp = x - nbr

    h, w = hp.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    n_r = max(24, min(h, w) // 8)
    n_t = 72
    rs = np.linspace(0.08 * min(h, w) / 2.0, 0.92 * min(h, w) / 2.0, n_r)
    ts = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    yy = cy + np.outer(rs, np.sin(ts))
    xx = cx + np.outer(rs, np.cos(ts))
    yi = np.clip(np.rint(yy).astype(int), 0, h - 1)
    xi = np.clip(np.rint(xx).astype(int), 0, w - 1)
    samples = hp[yi, xi]
    return float(samples.var(axis=1).mean())


def sharpness_score(img: np.ndarray) -> float:
    x = _to_float(img)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return 0.0
    x = (x - lo) / (hi - lo)
    lap = (
        -4.0 * x
        + np.roll(x, 1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, 1, 1)
        + np.roll(x, -1, 1)
    )
    return float(lap.var())


def difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Absolute difference, shared percentile window → uint8."""
    a = _to_float(before)
    b = _to_float(after)
    d = np.abs(a - b)
    lo, hi = np.percentile(d, (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((d - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def compare_pair(before: np.ndarray, after: np.ndarray) -> Dict[str, float]:
    rb = ring_score(before)
    ra = ring_score(after)
    sb = sharpness_score(before)
    sa = sharpness_score(after)
    reduction = 0.0 if rb <= 1e-12 else 100.0 * (rb - ra) / rb
    return {
        "ring_before": rb,
        "ring_after": ra,
        "ring_reduction_pct": reduction,
        "sharp_before": sb,
        "sharp_after": sa,
        "sharp_change_pct": 0.0 if sb <= 1e-12 else 100.0 * (sa - sb) / sb,
    }


def qc_warning(metrics: Dict[str, float]) -> str:
    notes = []
    if metrics["ring_reduction_pct"] < 5.0:
        notes.append("rings barely reduced — try stronger preset / different method")
    if metrics["ring_reduction_pct"] > 40.0 and metrics["sharp_change_pct"] < -15.0:
        notes.append("possible overcorrection (rings down but sharpness dropped)")
    if metrics["ring_after"] > metrics["ring_before"] * 1.05:
        notes.append("AFTER looks ringier than BEFORE — check method / params")
    return "; ".join(notes) if notes else "OK"


def format_qc_line(metrics: Dict[str, float]) -> str:
    warn = qc_warning(metrics)
    return (
        f"QC rings {metrics['ring_before']:.4g}→{metrics['ring_after']:.4g} "
        f"({metrics['ring_reduction_pct']:+.1f}%)  "
        f"sharp {metrics['sharp_change_pct']:+.1f}%  [{warn}]"
    )


def shared_norm_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Same window/level for fair before/after comparison."""
    stack = np.concatenate([_to_float(a).ravel(), _to_float(b).ravel()])
    lo, hi = np.percentile(stack, (1, 99))
    if hi <= lo:
        hi = lo + 1e-6

    def _n(x: np.ndarray) -> np.ndarray:
        s = np.clip((_to_float(x) - lo) / (hi - lo), 0, 1)
        return (s * 255).astype(np.uint8)

    return _n(a), _n(b)
