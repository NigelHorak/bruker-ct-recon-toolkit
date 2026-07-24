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
    "shift": "How far to nudge the rotation center (same idea as NRecon Postalignment). "
    "Wrong shift = blurry or double edges. Right shift = sharp edges.",
    "single_align": "Type one shift value and click Test. The Reconstruction pane shows that result.",
    "range_align": "Builds several shifts from → to in steps so you can compare them with the arrows/slider.",
    "nudge": "Tiny left/right tweaks (±0.5 or ±1 pixel) after you already have a close shift. "
    "Same as typing a slightly different number in Single mode — optional fine polish.",
    "centers": "Base = auto/log center. Effective = base + the shift you are viewing. Read-only.",
    "ring_method": "Which ring-cleanup recipe to use. These names map to Algotom stripe filters. Start with All rings.",
    "ring_recipes": "Runs a few Algotom ring recipes at your current alignment so you can pick the cleanest look.",
    "ring_strength": "Keeps the current recipe and sweeps its strength gate (SNR) across a range.",
    "snr": "How strongly a ring must stand out before it is removed (Algotom snr). "
    "Lower = more aggressive cleanup (can blur real edges).",
    "la_size": "Algotom large-ring width (pixels, odd). Bigger catches thicker rings.",
    "sm_size": "Algotom fine smoothing size (odd). Bigger = smoother, can soften detail.",
    "drop_ratio": "Algotom drop_ratio — how aggressively outlier stripes are dropped.",
    "dim": "Algotom stripe axis. Usually leave at 1.",
    "speed": "Fast = rebuild one slice at a time (great for tuning). "
    "Careful 3D = true cone-beam volume (needs more GPU/RAM, slower).",
    "filter": "Reconstruction filter (Algotom/Astra). Balanced is fine for almost everything.",
    "apply_log": "Converts detector brightness into the numbers CT math expects. Leave ON unless you know otherwise.",
    "preset": "Saved recipes (YAML files in config/presets). Apply loads them; Save writes a new one.",
    "bh": "Algotom beam_hardening_correction on the sinogram. "
    "q is the curve strength (try ~0.01–0.2). n must be > 1 (often ~2).",
    "folder": "Paste a path or click Browse to pick the scan folder (not a single file).",
    "slice": "Detector row to reconstruct. Middle is usually fastest and representative.",
    "viewer": "Scroll to zoom, drag to pan, double-click to zoom in. Scale bar uses Image Pixel Size from the Bruker .log.",
}


def ring_label(code: str) -> str:
    return RING_CODE_TO_LABEL.get(code, RING_CODE_TO_LABEL["remove_all_stripe"])


def filter_label(code: str) -> str:
    return FILTER_CODE_TO_LABEL.get(code, FILTER_CODE_TO_LABEL["hann"])


def speed_label(code: str) -> str:
    return SPEED_CODE_TO_LABEL.get(str(code).upper(), SPEED_CODE_TO_LABEL["FBP"])
