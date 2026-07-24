"""
Bruker TIFF + .log reconstruction with Algotom ring removal and Astra FBP_CUDA.

Usage:
  python reconstruct_bruker.py --scan-dir D:\\data\\MyScan
  python reconstruct_bruker.py --scan-dir D:\\data\\MyScan --preview
  python reconstruct_bruker.py --scan-dir D:\\data\\MyScan --preview --preview-row 512
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    for p in logs:
        if "recon" not in p.stem.lower():
            return p
    return logs[0]


def _proj_index(path: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", path.stem)
    idx = int(m.group(1)) if m else -1
    return idx, path.name.lower()


def list_projections(scan_dir: Path, prefix: str = "") -> List[Path]:
    files = []
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(scan_dir.glob(pat))
    skip_tokens = ("_rec", "rec_", "dark", "flat", "ref", "white", "bkg", "background", "arc")
    filtered: List[Path] = []
    for p in files:
        name = p.name.lower()
        if any(tok in name for tok in skip_tokens):
            continue
        if prefix and not p.name.startswith(prefix):
            if prefix.rstrip("_") and not p.name.startswith(prefix.rstrip("_")):
                continue
        filtered.append(p)

    if not filtered:
        for p in files:
            if re.search(r"\d+$", p.stem):
                filtered.append(p)

    filtered = sorted(set(filtered), key=_proj_index)
    filtered = [p for p in filtered if _proj_index(p)[0] >= 0]
    if not filtered:
        raise FileNotFoundError(f"No projection TIFFs found in {scan_dir}")
    return filtered


def _imread(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except ImportError:
        from algotom.io import loadersaver as losa

        return np.asarray(losa.load_image(str(path)), dtype=np.float32)


def probe_projection_shape(paths: List[Path]) -> Tuple[int, int]:
    arr = _imread(paths[0])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D projection TIFF, got shape {arr.shape} from {paths[0]}")
    return int(arr.shape[0]), int(arr.shape[1])


def load_projection_stack(paths: List[Path]) -> np.ndarray:
    """Load projections as float32 array shaped (n_angles, height, width)."""
    imgs = []
    for i, p in enumerate(paths):
        if i % 50 == 0 or i == len(paths) - 1:
            print(f"  loading projection {i + 1}/{len(paths)}: {p.name}")
        imgs.append(_imread(p))
    return np.stack(imgs, axis=0)


def load_sinogram_row(paths: List[Path], row: int) -> np.ndarray:
    """
    Fast path: build one sinogram (n_angles, width) by keeping only `row`
    from each projection. Avoids holding the full 3D stack in RAM.
    """
    n = len(paths)
    first = _imread(paths[0])
    height, width = first.shape
    if row < 0 or row >= height:
        raise ValueError(f"preview row {row} out of range 0..{height - 1}")

    sino = np.empty((n, width), dtype=np.float32)
    sino[0] = first[row, :]
    for i, p in enumerate(paths[1:], start=1):
        if i % 50 == 0 or i == n - 1:
            print(f"  loading row {row} from projection {i + 1}/{n}: {p.name}")
        arr = _imread(p)
        sino[i] = arr[row, :]
    return sino


def default_output_dir(scan_dir: Path, configured: str, preview: bool) -> Path:
    if configured:
        out = Path(configured)
        if not out.is_absolute():
            out = scan_dir / out
        return out
    suffix = "_algotom_preview" if preview else "_algotom_recon"
    return scan_dir.parent / f"{scan_dir.name}{suffix}"


def _save_qc_png(path: Path, img: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"QC preview: {path}")
    except Exception as exc:
        print(f"WARNING: could not save preview PNG: {exc}")


def _reconstruct_sinogram(
    sino: np.ndarray,
    center: float,
    thetas: np.ndarray,
    method: str,
    filter_name: str,
    apply_log: bool,
) -> np.ndarray:
    import algotom.rec.reconstruction as rec

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
            print(f"  CUDA recon failed ({exc}); falling back to FBP CPU")
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
    return np.asarray(img, dtype=np.float32)


def reconstruct(
    scan_dir: Path,
    out_dir: Optional[Path],
    config_path: Path,
    preview: bool = False,
    preview_row: Optional[int] = None,
) -> Path:
    import algotom.prep.calculation as calc
    import algotom.prep.removal as remo
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
    if preview:
        print("MODE: PREVIEW (single slice — fast ring-tuning pass)")

    proj_paths = list_projections(scan_dir, prefix=meta.filename_prefix)
    if meta.number_of_files and len(proj_paths) != meta.number_of_files:
        print(
            f"WARNING: found {len(proj_paths)} projections, log says {meta.number_of_files}"
        )
    print(f"Projections: {len(proj_paths)} (first={proj_paths[0].name}, last={proj_paths[-1].name})")

    height, width = probe_projection_shape(proj_paths)
    if meta.number_of_rows and meta.number_of_rows != height:
        print(f"WARNING: log rows={meta.number_of_rows}, TIFF height={height}")
    if meta.number_of_columns and meta.number_of_columns != width:
        print(f"WARNING: log cols={meta.number_of_columns}, TIFF width={width}")

    mid = height // 2
    row = mid if preview_row is None else int(preview_row)
    if row < 0 or row >= height:
        raise ValueError(f"preview row {row} out of range 0..{height - 1}")

    snr = float(ring_cfg.get("snr", 3.0))
    la_size = int(ring_cfg.get("la_size", 51))
    sm_size = int(ring_cfg.get("sm_size", 21))
    chunk = int(recon_cfg.get("chunk_size", 32))
    method = str(recon_cfg.get("method", "FBP_CUDA"))
    apply_log = bool(recon_cfg.get("apply_log", True))
    filter_name = str(recon_cfg.get("filter_name", "hann"))
    center_cfg = recon_cfg.get("center", None)

    out = out_dir or default_output_dir(scan_dir, paths_cfg.get("output_dir") or "", preview)
    qc_dir = out / "qc"
    slices_dir = out / "slices"

    if preview:
        if qc_dir.exists():
            shutil.rmtree(qc_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out.exists():
            print(f"Output exists — cleaning slices/qc under {out}")
            if slices_dir.exists():
                shutil.rmtree(slices_dir)
            if qc_dir.exists():
                shutil.rmtree(qc_dir)
        slices_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Ring removal: remove_all_stripe(snr={snr}, la_size={la_size}, sm_size={sm_size})"
    )
    print(f"Recon: {method}  apply_log={apply_log}  chunk_size={chunk}")

    if preview:
        print(f"Loading single detector row {row}/{height - 1} from all projections...")
        sino = load_sinogram_row(proj_paths, row)
        n_angles, width = sino.shape
        angles_deg = estimate_angles_deg(meta, n_angles)
        thetas = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

        if center_cfg is None:
            print("Finding center of rotation (Algotom)...")
            try:
                center = float(calc.find_center_vo(sino))
            except Exception as exc:
                center = (width - 1) / 2.0
                print(f"  auto center failed ({exc}); using geometric mid = {center:.3f}")
        else:
            center = float(center_cfg)
        print(f"Center of rotation: {center:.3f}")

        try:
            sino_corr = remo.remove_all_stripe(sino, snr=snr, la_size=la_size, sm_size=sm_size)
        except Exception as exc:
            print(f"WARNING: ring removal failed: {exc}")
            sino_corr = sino

        img = _reconstruct_sinogram(sino_corr, center, thetas, method, filter_name, apply_log)
        losa.save_image(str(qc_dir / "preview_slice.tif"), img)
        if io_cfg.get("save_preview", True):
            _save_qc_png(
                qc_dir / "mid_slice.png",
                img,
                f"PREVIEW row={row}  center={center:.2f}",
            )

        run_info = {
            "mode": "preview",
            "preview_row": row,
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
        print(f"Preview done. Open: {qc_dir / 'mid_slice.png'}")
        print("When rings look good, run full recon without -Preview.")
        return out

    # --- Full volume path ---
    print("Loading projection stack into memory...")
    projections = load_projection_stack(proj_paths)
    n_angles, height, width = projections.shape
    print(f"Stack shape (angles, H, W) = {projections.shape}")

    angles_deg = estimate_angles_deg(meta, n_angles)
    thetas = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    sino_mid = projections[:, mid, :]

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

    n_done = 0
    preview_img = None
    for row0 in range(0, height, chunk):
        row1 = min(row0 + chunk, height)
        for r in range(row0, row1):
            sino = projections[:, r, :].copy()
            try:
                sino = remo.remove_all_stripe(sino, snr=snr, la_size=la_size, sm_size=sm_size)
            except Exception as exc:
                print(f"  WARNING: ring removal failed on row {r}: {exc}")

            img = _reconstruct_sinogram(sino, center, thetas, method, filter_name, apply_log)
            losa.save_image(str(slices_dir / f"recon_{r:05d}.tif"), img)
            if r == mid:
                preview_img = img
            n_done += 1
        print(f"  reconstructed rows {row0}..{row1 - 1}  ({n_done}/{height})")

    if preview_img is not None and io_cfg.get("save_preview", True):
        losa.save_image(str(qc_dir / "preview_slice.tif"), preview_img)
        _save_qc_png(qc_dir / "mid_slice.png", preview_img, f"mid slice row={mid}  center={center:.2f}")

    run_info = {
        "mode": "full",
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
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Fast mode: reconstruct only one detector row (default: mid) for ring tuning",
    )
    parser.add_argument(
        "--preview-row",
        type=int,
        default=None,
        help="Detector row index for --preview (default: middle row)",
    )
    args = parser.parse_args(argv)

    if not args.scan_dir.is_dir():
        print(f"ERROR: scan dir not found: {args.scan_dir}")
        return 1
    if not args.config.is_file():
        print(f"ERROR: config not found: {args.config}")
        return 1

    reconstruct(
        args.scan_dir,
        args.out_dir,
        args.config,
        preview=args.preview,
        preview_row=args.preview_row,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
