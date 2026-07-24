"""
Bruker TIFF + .log reconstruction with Algotom ring removal and Astra FBP_CUDA.

Usage:
  python reconstruct_bruker.py --scan-dir D:\\data\\MyScan
  python reconstruct_bruker.py --scan-dir D:\\data\\MyScan --out-dir D:\\out --config config\\default.yaml
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running as scripts\reconstruct_bruker.py
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_bruker_log import estimate_angles_deg, parse_bruker_log  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Re-run setup.ps1") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def find_log_file(scan_dir: Path) -> Path:
    logs = sorted(scan_dir.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No .log file found in {scan_dir}")
    # Prefer non-recon logs if multiple
    for p in logs:
        if "recon" not in p.stem.lower():
            return p
    return logs[0]


def _proj_index(path: Path) -> Tuple[int, str]:
    """Sort key: trailing digits in stem, then name."""
    m = re.search(r"(\d+)$", path.stem)
    idx = int(m.group(1)) if m else -1
    return idx, path.name.lower()


def list_projections(scan_dir: Path, prefix: str = "") -> List[Path]:
    files = []
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(scan_dir.glob(pat))
    # Drop obvious non-projection names
    skip_tokens = ("_rec", "rec_", "dark", "flat", "ref", "white", "bkg", "background", "arc")
    filtered: List[Path] = []
    for p in files:
        name = p.name.lower()
        if any(tok in name for tok in skip_tokens):
            continue
        if prefix and not p.name.startswith(prefix):
            # Still allow if prefix empty or fuzzy
            if prefix.rstrip("_") and not p.name.startswith(prefix.rstrip("_")):
                continue
        filtered.append(p)

    if not filtered:
        # Fallback: all tifs with a trailing index
        for p in files:
            if re.search(r"\d+$", p.stem):
                filtered.append(p)

    filtered = sorted(set(filtered), key=_proj_index)
    # Keep only those with numeric suffix
    filtered = [p for p in filtered if _proj_index(p)[0] >= 0]
    if not filtered:
        raise FileNotFoundError(f"No projection TIFFs found in {scan_dir}")
    return filtered


def load_projection_stack(paths: List[Path]) -> np.ndarray:
    """Load projections as float32 array shaped (n_angles, height, width)."""
    try:
        import tifffile
    except ImportError:
        from algotom.io import loadersaver as losa

        imgs = [np.asarray(losa.load_image(str(p)), dtype=np.float32) for p in paths]
        return np.stack(imgs, axis=0)

    imgs = []
    for i, p in enumerate(paths):
        if i % 50 == 0 or i == len(paths) - 1:
            print(f"  loading projection {i + 1}/{len(paths)}: {p.name}")
        arr = tifffile.imread(str(p))
        imgs.append(np.asarray(arr, dtype=np.float32))
    return np.stack(imgs, axis=0)


def default_output_dir(scan_dir: Path, configured: str) -> Path:
    if configured:
        out = Path(configured)
        if not out.is_absolute():
            out = scan_dir / out
        return out
    return scan_dir.parent / f"{scan_dir.name}_algotom_recon"


def reconstruct(
    scan_dir: Path,
    out_dir: Optional[Path],
    config_path: Path,
) -> Path:
    import algotom.prep.calculation as calc
    import algotom.prep.removal as remo
    import algotom.rec.reconstruction as rec
    from algotom.io import loadersaver as losa

    cfg = _load_yaml(config_path)
    ring_cfg = cfg.get("ring", {})
    recon_cfg = cfg.get("recon", {})
    io_cfg = cfg.get("io", {})
    paths_cfg = cfg.get("paths", {})

    scan_dir = scan_dir.resolve()
    log_path = find_log_file(scan_dir)
    meta = parse_bruker_log(log_path)
    print(f"Log: {log_path}")
    print(
        f"  prefix={meta.filename_prefix!r}  n_files={meta.number_of_files}  "
        f"rows={meta.number_of_rows}  cols={meta.number_of_columns}  "
        f"rot_step={meta.rotation_step_deg}  360={meta.use_360_rotation}"
    )

    proj_paths = list_projections(scan_dir, prefix=meta.filename_prefix)
    if meta.number_of_files and len(proj_paths) != meta.number_of_files:
        print(
            f"WARNING: found {len(proj_paths)} projections, log says {meta.number_of_files}"
        )
    print(f"Projections: {len(proj_paths)} (first={proj_paths[0].name}, last={proj_paths[-1].name})")

    print("Loading projection stack into memory...")
    projections = load_projection_stack(proj_paths)
    n_angles, height, width = projections.shape
    print(f"Stack shape (angles, H, W) = {projections.shape}")

    angles_deg = estimate_angles_deg(meta, n_angles)
    thetas = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

    # Mid-row sinogram for center finding: shape (angles, width)
    mid = height // 2
    sino_mid = projections[:, mid, :]

    center_cfg = recon_cfg.get("center", None)
    if center_cfg is None:
        print("Finding center of rotation (Algotom)...")
        try:
            center = float(calc.find_center_vo(sino_mid))
        except Exception as exc:
            center = (width - 1) / 2.0
            print(f"  auto center failed ({exc}); using geometric mid = {center:.3f}")
    else:
        center = float(center_cfg)
    print(f"Center of rotation: {center:.3f}")

    out = out_dir or default_output_dir(scan_dir, paths_cfg.get("output_dir") or "")
    slices_dir = out / "slices"
    qc_dir = out / "qc"
    if out.exists():
        print(f"Output exists — cleaning slices/qc under {out}")
        if slices_dir.exists():
            shutil.rmtree(slices_dir)
        if qc_dir.exists():
            shutil.rmtree(qc_dir)
    slices_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    snr = float(ring_cfg.get("snr", 3.0))
    la_size = int(ring_cfg.get("la_size", 51))
    sm_size = int(ring_cfg.get("sm_size", 21))
    chunk = int(recon_cfg.get("chunk_size", 32))
    method = str(recon_cfg.get("method", "FBP_CUDA"))
    apply_log = bool(recon_cfg.get("apply_log", True))
    filter_name = str(recon_cfg.get("filter_name", "hann"))

    print(
        f"Ring removal: remove_all_stripe(snr={snr}, la_size={la_size}, sm_size={sm_size})"
    )
    print(f"Recon: {method}  apply_log={apply_log}  chunk_size={chunk}")

    # Process row chunks: for each row build sinogram, remove stripes, reconstruct
    n_done = 0
    preview_img = None
    preview_row = mid

    for row0 in range(0, height, chunk):
        row1 = min(row0 + chunk, height)
        # sinograms chunk: (angles, n_rows, width) -> process each row
        for row in range(row0, row1):
            sino = projections[:, row, :].copy()
            try:
                sino = remo.remove_all_stripe(sino, snr=snr, la_size=la_size, sm_size=sm_size)
            except Exception as exc:
                print(f"  WARNING: ring removal failed on row {row}: {exc}")

            try:
                img = rec.astra_reconstruction(
                    sino,
                    center,
                    angles=thetas,
                    method=method,
                    filter_name=filter_name,
                    apply_log=apply_log,
                )
            except Exception as exc:
                if method.upper().endswith("CUDA"):
                    print(f"  CUDA recon failed ({exc}); falling back to FBP CPU for this row")
                    img = rec.astra_reconstruction(
                        sino,
                        center,
                        angles=thetas,
                        method="FBP",
                        filter_name=filter_name,
                        apply_log=apply_log,
                    )
                else:
                    raise

            img = np.asarray(img, dtype=np.float32)
            out_name = slices_dir / f"recon_{row:05d}.tif"
            losa.save_image(str(out_name), img)
            if row == preview_row:
                preview_img = img
            n_done += 1

        print(f"  reconstructed rows {row0}..{row1 - 1}  ({n_done}/{height})")

    if preview_img is not None and io_cfg.get("save_preview", True):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(preview_img, cmap="gray")
            ax.set_title(f"mid slice row={preview_row}  center={center:.2f}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(qc_dir / "mid_slice.png", dpi=150)
            plt.close(fig)
            print(f"QC preview: {qc_dir / 'mid_slice.png'}")
        except Exception as exc:
            print(f"WARNING: could not save preview PNG: {exc}")
            losa.save_image(str(qc_dir / "mid_slice.tif"), preview_img)

    run_info = {
        "scan_dir": str(scan_dir),
        "log_file": str(log_path),
        "n_projections": n_angles,
        "volume_shape_hw": [height, width],
        "center": center,
        "config": cfg,
        "meta_summary": {
            "filename_prefix": meta.filename_prefix,
            "rotation_step_deg": meta.rotation_step_deg,
            "use_360_rotation": meta.use_360_rotation,
            "image_pixel_size_um": meta.image_pixel_size_um,
            "object_to_source_mm": meta.object_to_source_mm,
            "camera_to_source_mm": meta.camera_to_source_mm,
            "flat_field_correction": meta.flat_field_correction,
        },
    }
    _save_yaml(out / "run_config.yaml", run_info)
    print(f"Done. Slices: {slices_dir}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bruker Algotom + Astra recon with ring removal")
    parser.add_argument("--scan-dir", required=True, type=Path, help="Folder with TIFF projections + .log")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config path",
    )
    args = parser.parse_args(argv)

    if not args.scan_dir.is_dir():
        print(f"ERROR: scan dir not found: {args.scan_dir}")
        return 1
    if not args.config.is_file():
        print(f"ERROR: config not found: {args.config}")
        return 1

    reconstruct(args.scan_dir, args.out_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
