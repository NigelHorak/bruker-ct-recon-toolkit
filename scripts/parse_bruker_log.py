"""
Parse Bruker / SkyScan-style acquisition .log files (INI-like key=value text).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class BrukerLogMeta:
    log_path: str
    filename_prefix: str = ""
    number_of_files: int = 0
    number_of_rows: int = 0
    number_of_columns: int = 0
    rotation_step_deg: float = 0.0
    use_360_rotation: bool = False
    image_pixel_size_um: float = 0.0
    object_to_source_mm: float = 0.0
    camera_to_source_mm: float = 0.0
    source_voltage_kv: float = 0.0
    source_current_ua: float = 0.0
    flat_field_correction: bool = False
    postalignment: float = 0.0
    postalignment_applied: bool = False
    raw: Dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.raw is None:
            self.raw = {}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _truthy(value: str) -> bool:
    v = value.strip().upper()
    return v in {"YES", "ON", "TRUE", "1", "Y"}


def _find_key(raw: Dict[str, str], *candidates: str) -> Optional[str]:
    # Exact match first (case-insensitive)
    lower_map = {k.lower().strip(): v for k, v in raw.items()}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # Fuzzy: startswith
    for cand in candidates:
        key = cand.lower().strip()
        for k, v in lower_map.items():
            if k.startswith(key):
                return v
    return None


def _as_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.replace(",", ""))
    return float(m.group(0)) if m else default


def _as_int(value: Optional[str], default: int = 0) -> int:
    if value is None:
        return default
    m = re.search(r"-?\d+", value.replace(",", ""))
    return int(m.group(0)) if m else default


def parse_bruker_log(log_path: str | Path) -> BrukerLogMeta:
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")

    text = path.read_text(encoding="latin-1", errors="replace")
    raw: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[") or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        raw[key.strip()] = val.strip()

    meta = BrukerLogMeta(
        log_path=str(path.resolve()),
        filename_prefix=_find_key(raw, "Filename Prefix") or "",
        number_of_files=_as_int(_find_key(raw, "Number Of Files", "Number of Files")),
        number_of_rows=_as_int(_find_key(raw, "Number Of Rows", "Number of Rows")),
        number_of_columns=_as_int(_find_key(raw, "Number Of Columns", "Number of Columns")),
        rotation_step_deg=_as_float(_find_key(raw, "Rotation Step (deg)", "Rotation Step")),
        use_360_rotation=_truthy(_find_key(raw, "Use 360 Rotation") or "NO"),
        image_pixel_size_um=_as_float(
            _find_key(raw, "Image Pixel Size (um)", "Scaled Image Pixel Size (um)", "Pixel Size (um)")
        ),
        object_to_source_mm=_as_float(_find_key(raw, "Object to Source (mm)")),
        camera_to_source_mm=_as_float(_find_key(raw, "Camera to Source (mm)")),
        source_voltage_kv=_as_float(_find_key(raw, "Source Voltage (kV)")),
        source_current_ua=_as_float(_find_key(raw, "Source Current (uA)")),
        flat_field_correction=_truthy(_find_key(raw, "Flat Field Correction") or "OFF"),
        postalignment=_as_float(_find_key(raw, "Postalignment"), 0.0),
        postalignment_applied=_truthy(_find_key(raw, "Postalignment Applied") or "0"),
        raw=raw,
    )
    return meta


def estimate_angles_deg(meta: BrukerLogMeta, n_proj: int) -> list[float]:
    """Return projection angles in degrees from log rotation step / 180-360 flag."""
    step = meta.rotation_step_deg
    if step <= 0:
        span = 360.0 if meta.use_360_rotation else 180.0
        step = span / max(n_proj, 1)
    angles = [i * step for i in range(n_proj)]
    return angles


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_bruker_log.py <path-to.log>")
        sys.exit(1)
    m = parse_bruker_log(sys.argv[1])
    print(json.dumps(m.to_dict(), indent=2, default=str))
