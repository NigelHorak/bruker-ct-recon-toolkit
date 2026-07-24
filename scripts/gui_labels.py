"""
Plain-language labels + help text for the partner GUI.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Slider range for alignment (nudges are clamped to this)
SHIFT_MIN = -20.0
SHIFT_MAX = 20.0

RING_METHOD_UI: List[Tuple[str, str]] = [
    ("All rings (recommended)", "remove_all_stripe"),
    ("Fine rings only", "remove_stripe_based_sorting"),
    ("Large rings only", "remove_large_stripe"),
    ("Dead / stuck detector lines", "remove_dead_stripe"),
    ("Off (no ring cleanup)", "none"),
]

RING_METHOD_TO_CODE: Dict[str, str] = {label: code for label, code in RING_METHOD_UI}
RING_CODE_TO_LABEL: Dict[str, str] = {code: label for label, code in RING_METHOD_UI}

FILTER_UI: List[Tuple[str, str]] = [
    ("Balanced (default)", "hann"),
    ("Sharper (more noise)", "ram-lak"),
    ("Smooth (less noise)", "shepp-logan"),
    ("Soft edges", "cosine"),
    ("Slightly soft", "hamming"),
]
FILTER_TO_CODE: Dict[str, str] = {label: code for label, code in FILTER_UI}
FILTER_CODE_TO_LABEL: Dict[str, str] = {code: label for label, code in FILTER_UI}
# ramlak alias
FILTER_CODE_TO_LABEL["ramlak"] = "Sharper (more noise)"

SPEED_UI: List[Tuple[str, str]] = [
    ("Fast preview / recon", "FBP"),
    ("Careful 3D (slower, uses full geometry)", "FDK"),
]
SPEED_TO_CODE: Dict[str, str] = {label: code for label, code in SPEED_UI}
SPEED_CODE_TO_LABEL: Dict[str, str] = {code: label for label, code in SPEED_UI}

INFO = {
    "shift": "Nudge how the rotation center lines up (same idea as NRecon Postalignment). "
    "Move until edges look sharp, not double.",
    "auto_tune": "Tries many alignment shifts and keeps the sharpest mid-slice "
    "(our sharpness score, not an Algotom built-in). Then quickly tries a few ring cleaners "
    "and applies the one with the lowest ring score.",
    "multi_row": "Checks top, middle, and bottom slices agree on the same shift.",
    "ring_method": "Which ring-cleanup recipe to use. Start with All rings.",
    "snr": "How strongly a ring must stand out before it is removed. "
    "Lower = more aggressive cleanup (can blur real edges).",
    "la_size": "How wide large rings are allowed to be (in pixels). Bigger catches thicker rings.",
    "sm_size": "How fine the small-ring cleanup is. Bigger = smoother, can soften detail.",
    "drop_ratio": "How aggressively outlier stripes are dropped. Higher = stronger cleanup.",
    "dim": "Usually leave at 1. Use 2 only if rings look wrong with 1.",
    "speed": "Fast = rebuild one slice at a time (great for tuning). "
    "Careful 3D = true cone-beam volume (needs more GPU/RAM, slower).",
    "filter": "Reconstruction filter. Balanced is fine for almost everything.",
    "apply_log": "Converts detector brightness into the numbers CT math expects. Leave ON unless you know otherwise.",
    "preset": "Saved recipes (YAML files in config/presets). Apply loads them; Save writes a new one.",
    "history": "Click a thumbnail to reload those settings and show that image.",
    "viewer": "Scroll/pinch zoom in the browser; click the expand control if shown for a larger view.",
}


def ring_label(code: str) -> str:
    return RING_CODE_TO_LABEL.get(code, RING_CODE_TO_LABEL["remove_all_stripe"])


def filter_label(code: str) -> str:
    return FILTER_CODE_TO_LABEL.get(code, FILTER_CODE_TO_LABEL["hann"])


def speed_label(code: str) -> str:
    return SPEED_CODE_TO_LABEL.get(str(code).upper(), SPEED_CODE_TO_LABEL["FBP"])
