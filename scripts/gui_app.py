"""
Gradio GUI - NRecon-style: generate options in a range, user picks the best.
"""
from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recon_core import (  # noqa: E402
    AlignCache,
    Settings,
    load_settings,
    prepare_align_cache,
    probe_scan_info,
    quick_align_preview,
    run_full,
    run_preview,
    save_yaml,
)
from gui_style import GUI_CSS, build_theme  # noqa: E402
from gui_log import ProgressLog, log_exception, log_line, log_path, startup_banner  # noqa: E402
from gui_labels import (  # noqa: E402
    INFO,
    RING_METHOD_TO_CODE,
    RING_METHOD_UI,
    SHIFT_MAX,
    SHIFT_MIN,
    SPEED_TO_CODE,
    SPEED_UI,
    ring_label,
    speed_label,
)
from sweep_tools import (  # noqa: E402
    Candidate,
    apply_simple_beam_hardening,
    sweep_alignment,
    sweep_beam_hardening,
    sweep_ring_recipes,
    sweep_ring_strength,
)
from viewer_html import (  # noqa: E402
    VIEWER_HEAD,
    centers_html,
    empty_viewer_html,
    tip_html,
    viewer_html,
)

PRESET_DIR = ROOT / "config" / "presets"
DEFAULT_CFG = ROOT / "config" / "default.yaml"


def _help(text: str, tip_key: str = "") -> str:
    tip = tip_html(INFO[tip_key]) if tip_key and tip_key in INFO else ""
    return f"<div class='ct-help'>{text}{tip}</div>"


def _preset_choices() -> List[str]:
    names = ["default"]
    if PRESET_DIR.is_dir():
        names.extend(sorted(p.stem for p in PRESET_DIR.glob("*.yaml")))
    return names


def _load_preset(name: str) -> Settings:
    if name == "default":
        return load_settings(DEFAULT_CFG)
    path = PRESET_DIR / f"{name}.yaml"
    return load_settings(path) if path.is_file() else load_settings(DEFAULT_CFG)


def _clamp_shift(v: float) -> float:
    return float(max(SHIFT_MIN, min(SHIFT_MAX, round(float(v or 0.0), 3))))


def _settings_from_ui(
    ring_method_label: str,
    snr: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    speed_label_v: str,
    apply_log: bool,
    pixel_shift: float,
    preview_row: int,
    output_dir: str,
) -> Settings:
    code_ring = RING_METHOD_TO_CODE.get(str(ring_method_label), "remove_all_stripe")
    rtype = SPEED_TO_CODE.get(str(speed_label_v), "FBP")
    la, sm = int(la_size), int(sm_size)
    return Settings(
        ring_enable=code_ring != "none",
        ring_method=code_ring,
        snr=float(snr),
        la_size=la if la % 2 == 1 else la + 1,
        sm_size=sm if sm % 2 == 1 else sm + 1,
        drop_ratio=float(drop_ratio),
        dim=int(dim),
        recon_type=rtype,
        method="FBP_CUDA",
        filter_name="hann",
        apply_log=bool(apply_log),
        num_iter=100,
        chunk_size=32,
        center_mode="auto",
        center=None,
        pixel_shift=_clamp_shift(pixel_shift),
        preview_row=int(preview_row) if int(preview_row or -1) >= 0 else None,
        output_dir=(output_dir or "").strip(),
        save_preview=True,
    )


def ui_tuple(s: Settings, bh: float = 0.0) -> Tuple[Any, ...]:
    return (
        ring_label(s.ring_method if s.ring_enable else "none"),
        s.snr,
        s.la_size,
        s.sm_size,
        s.drop_ratio,
        s.dim,
        speed_label(s.recon_type),
        s.apply_log,
        _clamp_shift(s.pixel_shift or 0.0),
        0 if s.preview_row is None else int(s.preview_row),
        s.output_dir or "",
        float(bh or 0.0),
    )


def _cand_list(cands: List[Candidate]) -> List[Dict[str, Any]]:
    return [{"label": c.label, "image": c.image, "payload": c.payload} for c in cands]


def _one_stack(label: str, image, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"label": label, "image": image, "payload": payload}]


def _slider_for(n: int, idx: int = 0):
    import gradio as gr

    n = max(1, int(n or 1))
    i = max(0, min(int(idx or 0), n - 1))
    # Gradio requires minimum < maximum (so max at least 1 even for a single image)
    return gr.update(maximum=max(1, n - 1), value=i, interactive=n > 1)


def _view(img, pix_um: float) -> str:
    return viewer_html(img, float(pix_um or 0.0))


def _centers(payload: Dict[str, Any], fallback_shift: float = 0.0) -> str:
    shift = float(payload.get("pixel_shift", fallback_shift) or 0.0)
    base = float(payload.get("base", 0.0) or 0.0)
    eff = float(payload.get("effective", base + shift) or (base + shift))
    return centers_html(base, eff, shift)


def on_browse_folder(current: str):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askdirectory(title="Select Bruker scan folder")
        root.destroy()
        if path:
            log_line(f"Browse folder -> {path}")
            return path
        return current or ""
    except Exception as exc:
        log_exception("Browse folder", exc)
        return current or ""


def on_load(scan_dir: str, preview_row: float):
    scan_dir = (scan_dir or "").strip().strip('"')
    empty = (
        "",
        0,
        0.0,
        None,
        empty_viewer_html(),
        None,
        centers_html(0, 0, 0),
        [],
        0,
        {},
        _slider_for(1, 0),
        0.0,
        "Enter a scan folder path.",
    )
    if not scan_dir:
        return empty

    path = Path(scan_dir)
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
        try:
            from PIL import Image

            arr = np.asarray(Image.open(path).convert("RGB"))
            stack = _one_stack(path.name, arr, {})
            return (
                str(path),
                0,
                0.0,
                None,
                _view(arr, 0.0),
                arr,
                centers_html(0, 0, 0),
                stack,
                0,
                {},
                _slider_for(1, 0),
                0.0,
                f"Loaded {path.name}",
            )
        except Exception as exc:
            return (*empty[:-1], log_exception("Load", exc))

    progress = ProgressLog(f"LOAD {scan_dir}")
    try:
        text, height, width, mid, post = probe_scan_info(scan_dir)
        row_in = int(preview_row or 0)
        row = mid if row_in < 0 else min(max(0, row_in), height - 1)
        cache = prepare_align_cache(Path(scan_dir), preview_row=row, progress=progress)
        shift = _clamp_shift(cache.log_postalignment or post or 0.0)
        img, msg, base, eff = quick_align_preview(cache, shift, apply_log=True, save_history=True)
        pix = float(getattr(cache, "pixel_size_um", 0.0) or 0.0)
        if pix <= 0:
            try:
                from parse_bruker_log import parse_bruker_log
                from recon_core import find_log_file

                pix = float(parse_bruker_log(find_log_file(Path(scan_dir))).image_pixel_size_um or 0.0)
            except Exception:
                pix = 0.0
        payload = {"pixel_shift": float(shift), "base": float(base), "effective": float(eff)}
        stack = _one_stack(f"shift {shift:+.2f}", img, payload)
        status = (
            f"Loaded. Detector {height}x{width}. Slice {row} (middle={mid}).\n"
            f"Started at Bruker log alignment {shift:+.3f}.\n{msg}\n{progress.text()}"
        )
        return (
            text,
            row,
            shift,
            cache,
            _view(img, pix),
            img,
            _centers(payload),
            stack,
            0,
            payload,
            _slider_for(1, 0),
            pix,
            status,
        )
    except Exception as exc:
        return (*empty[:-1], log_exception("Load", exc))


def on_middle_slice(scan_dir: str, cache: Optional[AlignCache]):
    if cache is not None:
        mid = int(cache.height // 2)
        return mid, f"Slice set to middle ({mid}). Click Load scan to rebuild the cache for that slice."
    scan_dir = (scan_dir or "").strip().strip('"')
    if scan_dir and Path(scan_dir).is_dir():
        try:
            _t, _h, _w, mid, _p = probe_scan_info(scan_dir)
            return mid, f"Middle slice is {mid}. Click Load scan."
        except Exception as exc:
            return 0, log_exception("Middle slice", exc)
    return 0, "Load a scan first."


def _stack_result(stack, pix, status: str):
    if not stack:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            status,
        )
    item = stack[0]
    img = item["image"]
    pl = item["payload"]
    return (
        _view(img, pix),
        img,
        stack,
        0,
        pl,
        _slider_for(len(stack), 0),
        _centers(pl),
        status,
    )


def on_align_single(cache, single_shift, apply_log, pix):
    if cache is None:
        return empty_viewer_html(), None, [], 0, {}, _slider_for(1, 0), centers_html(0, 0, 0), "Load a scan first."
    progress = ProgressLog("Single alignment")
    try:
        shift = _clamp_shift(single_shift)
        img, msg, base, eff = quick_align_preview(cache, shift, apply_log=bool(apply_log), save_history=True)
        pl = {"pixel_shift": float(shift), "base": float(base), "effective": float(eff)}
        stack = _one_stack(f"shift {shift:+.2f}", img, pl)
        return _stack_result(stack, pix, f"Tested shift {shift:+.3f}.\n{msg}\n{progress.text()}")
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("Single align", exc),
        )


def on_align_sweep(cache, shift_from, shift_to, shift_step, apply_log, pix):
    if cache is None:
        return empty_viewer_html(), None, [], 0, {}, _slider_for(1, 0), centers_html(0, 0, 0), "Load a scan first."
    progress = ProgressLog("Alignment sweep")
    try:
        cands = sweep_alignment(
            cache, float(shift_from), float(shift_to), float(shift_step), bool(apply_log), progress
        )
        stack = _cand_list(cands)
        return _stack_result(
            stack,
            pix,
            f"{len(stack)} alignments ready - browse with arrows/slider, then Use this alignment.\n{progress.text()}",
        )
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("Alignment sweep", exc),
        )


def on_ring_recipes(cache, pixel_shift, snr, la_size, sm_size, drop_ratio, dim, apply_log, pix):
    if cache is None:
        return empty_viewer_html(), None, [], 0, {}, _slider_for(1, 0), centers_html(0, 0, 0), "Load a scan first."
    progress = ProgressLog("Ring recipes")
    try:
        cands = sweep_ring_recipes(
            cache,
            float(pixel_shift or 0.0),
            float(snr),
            int(la_size),
            int(sm_size),
            float(drop_ratio),
            int(dim),
            bool(apply_log),
            progress,
        )
        stack = _cand_list(cands)
        return _stack_result(
            stack,
            pix,
            f"{len(stack)} ring recipes ready - browse, then Use this ring setting.\n{progress.text()}",
        )
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("Ring recipes", exc),
        )


def on_ring_strength(
    cache, pixel_shift, ring_method_label, snr_from, snr_to, snr_step, la_size, sm_size, drop_ratio, dim, apply_log, pix
):
    if cache is None:
        return empty_viewer_html(), None, [], 0, {}, _slider_for(1, 0), centers_html(0, 0, 0), "Load a scan first."
    progress = ProgressLog("Ring strength range")
    try:
        method = RING_METHOD_TO_CODE.get(str(ring_method_label), "remove_all_stripe")
        if method == "none":
            return (
                empty_viewer_html(),
                None,
                [],
                0,
                {},
                _slider_for(1, 0),
                centers_html(0, 0, 0),
                "Pick a ring recipe other than Off before sweeping strength.",
            )
        cands = sweep_ring_strength(
            cache,
            float(pixel_shift or 0.0),
            method,
            float(snr_from),
            float(snr_to),
            float(snr_step),
            int(la_size),
            int(sm_size),
            float(drop_ratio),
            int(dim),
            bool(apply_log),
            progress,
        )
        stack = _cand_list(cands)
        return _stack_result(
            stack,
            pix,
            f"{len(stack)} strength options ready - browse, then Use this ring setting.\n{progress.text()}",
        )
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("Ring strength", exc),
        )


def on_bh_sweep(base_img, bh_from, bh_to, bh_step, pix):
    if base_img is None:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            "Load or preview a slice first.",
        )
    progress = ProgressLog("BH options")
    try:
        cands = sweep_beam_hardening(base_img, float(bh_from), float(bh_to), float(bh_step), progress)
        stack = _cand_list(cands)
        return _stack_result(
            stack,
            pix,
            f"{len(stack)} BH options ready - browse, then Use this BH.\n{progress.text()}",
        )
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("BH sweep", exc),
        )


def on_recon_index(idx, stack, pix):
    if not stack:
        return empty_viewer_html(), None, {}, 0, _slider_for(1, 0), centers_html(0, 0, 0), "No reconstructions yet."
    i = max(0, min(int(idx or 0), len(stack) - 1))
    item = stack[i]
    img = item["image"]
    pl = item["payload"]
    return (
        _view(img, pix),
        img,
        pl,
        i,
        _slider_for(len(stack), i),
        _centers(pl),
        f"Viewing {i + 1}/{len(stack)}: {item['label']}",
    )


def on_recon_prev(idx, stack, pix):
    return on_recon_index(max(0, int(idx or 0) - 1), stack, pix)


def on_recon_next(idx, stack, pix):
    n = len(stack) if stack else 1
    return on_recon_index(min(n - 1, int(idx or 0) + 1), stack, pix)


def on_use_alignment(payload, cache, pix):
    if not payload or "pixel_shift" not in payload:
        return (
            0.0,
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            "Browse to an alignment first, then Use this alignment.",
        )
    shift = _clamp_shift(payload["pixel_shift"])
    if cache is None:
        return (
            shift,
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, shift),
            f"Using shift {shift:+.3f}.",
        )
    img, msg, base, eff = quick_align_preview(cache, shift, apply_log=True, save_history=True)
    pl = {"pixel_shift": float(shift), "base": float(base), "effective": float(eff)}
    stack = _one_stack(f"shift {shift:+.2f}", img, pl)
    view, work, stack, idx, pl, slider, centers, status = _stack_result(
        stack, pix, f"Locked alignment {shift:+.3f}.\n{msg}"
    )
    return shift, view, work, stack, idx, pl, slider, centers, status


def on_nudge(cache, pixel_shift, delta, apply_log, pix):
    if cache is None:
        return (
            _clamp_shift(pixel_shift),
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            "Load a scan first.",
        )
    shift = _clamp_shift(float(pixel_shift or 0.0) + float(delta))
    log_line(f"NUDGE {delta:+.1f} -> {shift:+.3f}")
    try:
        img, msg, base, eff = quick_align_preview(cache, shift, apply_log=bool(apply_log), save_history=True)
        pl = {"pixel_shift": float(shift), "base": float(base), "effective": float(eff)}
        stack = _one_stack(f"shift {shift:+.2f}", img, pl)
        view, work, stack, idx, pl, slider, centers, status = _stack_result(stack, pix, msg)
        return shift, view, work, stack, idx, pl, slider, centers, status
    except Exception as exc:
        return (
            shift,
            empty_viewer_html(),
            None,
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, shift),
            log_exception("Nudge", exc),
        )


def on_use_ring(payload):
    if not payload or "ring_method" not in payload:
        return (
            ring_label("remove_all_stripe"),
            3.0,
            51,
            21,
            0.1,
            1,
            "Browse to a ring option first, then Use this ring setting.",
        )
    method = payload["ring_method"]
    return (
        ring_label(method),
        float(payload.get("snr", 3.0)),
        int(payload.get("la_size", 51)),
        int(payload.get("sm_size", 21)),
        float(payload.get("drop_ratio", 0.1)),
        int(payload.get("dim", 1)),
        f"Locked ring setting: {ring_label(method)}",
    )


def on_use_bh(payload, base_img, pix):
    if not payload or "bh_strength" not in payload:
        return 0.0, empty_viewer_html(), base_img, "Browse to a BH option first."
    s = float(payload["bh_strength"])
    img = apply_simple_beam_hardening(base_img, s) if base_img is not None else None
    return s, _view(img, pix), img, f"Locked BH strength {s:.2f}."


def on_preview(scan_dir, before_cache, before_key, pix, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog("PREVIEW")
    if not scan_dir:
        return (
            empty_viewer_html(),
            None,
            before_cache,
            before_key or "",
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            "Enter a scan folder.",
        )
    try:
        settings = _settings_from_ui(*ctrl)
        result = run_preview(
            Path(scan_dir),
            settings,
            progress=progress,
            cached_before=before_cache,
            cached_before_key=before_key or "",
        )
        pl = {
            "pixel_shift": float(result.pixel_shift),
            "base": float(result.base_center),
            "effective": float(result.center),
            "ring_method": settings.ring_method if settings.ring_enable else "none",
            "ring_enable": bool(settings.ring_enable),
        }
        stack = _one_stack(f"preview row={result.row}", result.display_corr, pl)
        view, work, stack, idx, pl, slider, centers, status = _stack_result(
            stack, pix, f"{result.message}\n{progress.text()}"
        )
        return view, work, result.img_raw, result.before_key, stack, idx, pl, slider, centers, status
    except Exception as exc:
        return (
            empty_viewer_html(),
            None,
            before_cache,
            before_key or "",
            [],
            0,
            {},
            _slider_for(1, 0),
            centers_html(0, 0, 0),
            log_exception("Preview", exc),
        )


def on_full(scan_dir, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog("FULL RECON")
    if not scan_dir:
        return "Enter a scan folder."
    try:
        settings = _settings_from_ui(*ctrl)
        result = run_full(Path(scan_dir), settings, progress=progress)
        return f"{result.message}\n{progress.text()}"
    except Exception as exc:
        return log_exception("Full recon", exc)


def on_apply_preset(name: str):
    try:
        s = _load_preset(name)
        return (*ui_tuple(s), f"Loaded preset {name}")
    except Exception as exc:
        return (*ui_tuple(load_settings(DEFAULT_CFG)), log_exception("Preset", exc))


def on_save_recipe(name: str, *ctrl):
    import gradio as gr

    try:
        settings = _settings_from_ui(*ctrl[:11])
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "").strip()) or (
            f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        out = PRESET_DIR / f"{safe}.yaml"
        save_yaml(out, settings.to_config_dict())
        return f"Saved config/presets/{safe}.yaml", gr.update(choices=_preset_choices(), value=safe)
    except Exception as exc:
        return log_exception("Save recipe", exc), gr.update()


def on_align_mode(mode: str):
    import gradio as gr

    single = mode == "Single value"
    return gr.update(visible=single), gr.update(visible=not single)


def on_ring_mode(mode: str):
    import gradio as gr

    recipes = mode == "Try recipes"
    return gr.update(visible=recipes), gr.update(visible=not recipes)


def build_app():
    import gradio as gr

    defaults = load_settings(DEFAULT_CFG)
    ring_choices = [x[0] for x in RING_METHOD_UI]
    speed_choices = [x[0] for x in SPEED_UI]

    def _dd(**kw):
        try:
            return gr.Dropdown(**kw, allow_custom_value=False)
        except TypeError:
            return gr.Dropdown(**kw)

    blocks_kwargs: Dict[str, Any] = {"title": "Bruker CT Algotom Toolkit"}
    sig = inspect.signature(gr.Blocks.__init__)
    if "fill_width" in sig.parameters:
        blocks_kwargs["fill_width"] = True
    if "theme" in sig.parameters:
        th = build_theme()
        if th is not None:
            blocks_kwargs["theme"] = th
    blocks_kwargs["css"] = GUI_CSS
    if "head" in sig.parameters:
        blocks_kwargs["head"] = VIEWER_HEAD

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            f"""
            <div class="ct-header">
              <div>
                <div class="ct-brand">Bruker CT <span>Algotom</span></div>
                <div class="ct-sub">Generate options → browse Reconstruction → lock it in
                {tip_html('Same idea as NRecon: you choose. Hover any teal i for help.')}</div>
              </div>
            </div>
            """
        )

        align_cache = gr.State(None)
        before_cache = gr.State(None)
        before_key = gr.State("")
        recon_stack = gr.State([])
        recon_idx = gr.State(0)
        pending = gr.State({})
        work_img = gr.State(None)
        pixel_um = gr.State(0.0)

        pixel_shift = gr.Number(value=0.0, visible=False)
        bh_strength = gr.Number(value=0.0, visible=False)
        apply_log = gr.Checkbox(value=True, visible=False)

        with gr.Row(elem_classes=["ct-panel", "ct-scan-row"]):
            scan_dir = gr.Textbox(
                label="Scan folder",
                placeholder=r"D:\Results\...\MyScan",
                scale=4,
                elem_classes=["ct-mono"],
            )
            btn_browse = gr.Button("Browse...", scale=1)
            preview_row = gr.Number(value=0, label="Slice number", precision=0, scale=1)
            btn_mid = gr.Button("Middle (fastest)", scale=1)
            btn_load = gr.Button("Load scan", variant="primary", scale=1)

        gr.HTML(_help("Paste a path or Browse to pick the folder (not a file).", "folder"))
        scan_info = gr.Textbox(label="Scan info", interactive=False, lines=2, elem_classes=["ct-mono"])

        with gr.Row():
            with gr.Column(scale=2, min_width=340, elem_classes=["ct-panel"]):
                with gr.Tabs():
                    with gr.Tab("Align"):
                        gr.HTML(_help("Line up the rotation center until edges look sharp.", "shift"))
                        align_mode = gr.Radio(
                            choices=["Single value", "Range"],
                            value="Range",
                            label="Mode",
                        )
                        with gr.Group(visible=False) as align_single_box:
                            gr.HTML(_help("Type one shift and test it.", "single_align"))
                            single_shift = gr.Number(value=0.0, label="Center shift (pixels)", precision=3)
                            btn_align_single = gr.Button("Test this alignment", variant="primary")
                        with gr.Group(visible=True) as align_range_box:
                            gr.HTML(_help("Build several shifts to compare.", "range_align"))
                            with gr.Row():
                                shift_from = gr.Number(value=-10.0, label="From", precision=2)
                                shift_to = gr.Number(value=10.0, label="To", precision=2)
                                shift_step = gr.Number(value=5.0, label="Step", precision=2)
                            btn_align_sweep = gr.Button("Generate alignment options", variant="primary")

                        gr.HTML(_help("Optional fine polish after you are close.", "nudge"))
                        with gr.Row(elem_classes=["ct-nudge"]):
                            btn_m1 = gr.Button("Nudge -1")
                            btn_m05 = gr.Button("Nudge -0.5")
                            btn_p05 = gr.Button("Nudge +0.5")
                            btn_p1 = gr.Button("Nudge +1")

                        centers_box = gr.HTML(centers_html(0, 0, 0))
                        gr.HTML(_help("Read-only numbers for the reconstruction you are viewing.", "centers"))
                        btn_use_align = gr.Button("Use this alignment", variant="secondary")

                    with gr.Tab("Rings"):
                        gr.HTML(
                            _help(
                                "Ring recipes come from Algotom stripe filters. "
                                "Try recipes, or sweep strength for the current recipe.",
                                "ring_method",
                            )
                        )
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            preset = _dd(choices=_preset_choices(), value="default", label="Preset")
                            btn_preset = gr.Button("Apply")
                        ring_method = _dd(
                            choices=ring_choices,
                            value=ring_label(defaults.ring_method),
                            label="Current ring recipe",
                        )
                        ring_mode = gr.Radio(
                            choices=["Try recipes", "Strength range"],
                            value="Try recipes",
                            label="Mode",
                        )
                        with gr.Group(visible=True) as ring_recipes_box:
                            gr.HTML(_help("Runs Off / All / Fine / Large at your current alignment.", "ring_recipes"))
                            btn_ring_recipes = gr.Button("Generate ring recipes", variant="primary")
                        with gr.Group(visible=False) as ring_strength_box:
                            gr.HTML(_help("Varies Algotom SNR (strength gate) for the current recipe.", "ring_strength"))
                            with gr.Row():
                                snr_from = gr.Number(value=1.5, label="SNR from", precision=2)
                                snr_to = gr.Number(value=6.0, label="SNR to", precision=2)
                                snr_step = gr.Number(value=1.5, label="Step", precision=2)
                            btn_ring_strength = gr.Button("Generate strength options", variant="primary")

                        with gr.Accordion("Fine knobs (Algotom parameters)", open=False):
                            snr = gr.Slider(1.0, 10.0, value=defaults.snr, step=0.1, label="Strength gate (snr)")
                            gr.HTML(tip_html(INFO["snr"]))
                            la_size = gr.Slider(3, 151, value=defaults.la_size, step=2, label="Large-ring width")
                            gr.HTML(tip_html(INFO["la_size"]))
                            sm_size = gr.Slider(3, 101, value=defaults.sm_size, step=2, label="Fine smoothing")
                            gr.HTML(tip_html(INFO["sm_size"]))
                            drop_ratio = gr.Slider(0.0, 0.5, value=defaults.drop_ratio, step=0.01, label="Cleanup amount")
                            gr.HTML(tip_html(INFO["drop_ratio"]))
                            dim = gr.Radio(choices=[1, 2], value=defaults.dim, label="Stripe axis")
                            gr.HTML(tip_html(INFO["dim"]))

                        btn_use_ring = gr.Button("Use this ring setting", variant="secondary")
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            recipe_name = gr.Textbox(label="Save recipe as", placeholder="my_sample")
                            btn_save = gr.Button("Save")

                    with gr.Tab("Beam hardening"):
                        gr.HTML(_help("Simple post curve on the finished slice (not full physics).", "bh"))
                        with gr.Row():
                            bh_from = gr.Number(value=0.0, label="From", precision=2)
                            bh_to = gr.Number(value=2.0, label="To", precision=2)
                            bh_step = gr.Number(value=0.5, label="Step", precision=2)
                        btn_bh_sweep = gr.Button("Generate BH options", variant="primary")
                        btn_use_bh = gr.Button("Use this BH", variant="secondary")

                    with gr.Tab("Run"):
                        gr.HTML(_help("Fast for tuning; Careful 3D for the final volume.", "speed"))
                        speed = gr.Radio(
                            choices=speed_choices, value=speed_label(defaults.recon_type), label="Speed"
                        )
                        output_dir = gr.Textbox(value="", label="Output override (optional)")
                        btn_preview = gr.Button("Preview this slice", variant="primary")
                        btn_full = gr.Button("Reconstruct full volume", variant="stop")

            with gr.Column(scale=4, min_width=520, elem_classes=["ct-panel", "ct-viewer-wrap"]):
                gr.HTML(_help("Reconstruction", "viewer"))
                main_view = gr.HTML(value=empty_viewer_html())
                with gr.Row(elem_classes=["ct-nudge"]):
                    btn_recon_prev = gr.Button("< Previous", scale=1)
                    recon_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=1,
                        label="Browse results",
                        interactive=False,
                        scale=3,
                    )
                    btn_recon_next = gr.Button("Next >", scale=1)

        status = gr.Textbox(label="Log", lines=7, value=startup_banner(), elem_classes=["ct-mono", "ct-panel"])

        recon_controls = [
            ring_method,
            snr,
            la_size,
            sm_size,
            drop_ratio,
            dim,
            speed,
            apply_log,
            pixel_shift,
            preview_row,
            output_dir,
        ]
        stack_outs = [main_view, work_img, recon_stack, recon_idx, pending, recon_slider, centers_box, status]
        use_align_outs = [
            pixel_shift,
            main_view,
            work_img,
            recon_stack,
            recon_idx,
            pending,
            recon_slider,
            centers_box,
            status,
        ]
        nav_outs = [main_view, work_img, pending, recon_idx, recon_slider, centers_box, status]
        preview_outs = [
            main_view,
            work_img,
            before_cache,
            before_key,
            recon_stack,
            recon_idx,
            pending,
            recon_slider,
            centers_box,
            status,
        ]

        align_mode.change(on_align_mode, inputs=[align_mode], outputs=[align_single_box, align_range_box])
        ring_mode.change(on_ring_mode, inputs=[ring_mode], outputs=[ring_recipes_box, ring_strength_box])

        btn_browse.click(on_browse_folder, inputs=[scan_dir], outputs=[scan_dir])
        btn_mid.click(on_middle_slice, inputs=[scan_dir, align_cache], outputs=[preview_row, status])
        btn_load.click(
            on_load,
            inputs=[scan_dir, preview_row],
            outputs=[
                scan_info,
                preview_row,
                pixel_shift,
                align_cache,
                main_view,
                work_img,
                centers_box,
                recon_stack,
                recon_idx,
                pending,
                recon_slider,
                pixel_um,
                status,
            ],
        )

        btn_align_single.click(
            on_align_single,
            inputs=[align_cache, single_shift, apply_log, pixel_um],
            outputs=stack_outs,
        )
        btn_align_sweep.click(
            on_align_sweep,
            inputs=[align_cache, shift_from, shift_to, shift_step, apply_log, pixel_um],
            outputs=stack_outs,
        )

        btn_m1.click(
            lambda c, s, a, p: on_nudge(c, s, -1.0, a, p),
            inputs=[align_cache, pixel_shift, apply_log, pixel_um],
            outputs=use_align_outs,
        )
        btn_m05.click(
            lambda c, s, a, p: on_nudge(c, s, -0.5, a, p),
            inputs=[align_cache, pixel_shift, apply_log, pixel_um],
            outputs=use_align_outs,
        )
        btn_p05.click(
            lambda c, s, a, p: on_nudge(c, s, 0.5, a, p),
            inputs=[align_cache, pixel_shift, apply_log, pixel_um],
            outputs=use_align_outs,
        )
        btn_p1.click(
            lambda c, s, a, p: on_nudge(c, s, 1.0, a, p),
            inputs=[align_cache, pixel_shift, apply_log, pixel_um],
            outputs=use_align_outs,
        )

        btn_use_align.click(
            on_use_alignment,
            inputs=[pending, align_cache, pixel_um],
            outputs=use_align_outs,
        )

        btn_ring_recipes.click(
            on_ring_recipes,
            inputs=[align_cache, pixel_shift, snr, la_size, sm_size, drop_ratio, dim, apply_log, pixel_um],
            outputs=stack_outs,
        )
        btn_ring_strength.click(
            on_ring_strength,
            inputs=[
                align_cache,
                pixel_shift,
                ring_method,
                snr_from,
                snr_to,
                snr_step,
                la_size,
                sm_size,
                drop_ratio,
                dim,
                apply_log,
                pixel_um,
            ],
            outputs=stack_outs,
        )
        btn_use_ring.click(
            on_use_ring,
            inputs=[pending],
            outputs=[ring_method, snr, la_size, sm_size, drop_ratio, dim, status],
        )

        btn_bh_sweep.click(
            on_bh_sweep,
            inputs=[work_img, bh_from, bh_to, bh_step, pixel_um],
            outputs=stack_outs,
        )
        btn_use_bh.click(
            on_use_bh,
            inputs=[pending, work_img, pixel_um],
            outputs=[bh_strength, main_view, work_img, status],
        )

        recon_slider.change(on_recon_index, inputs=[recon_slider, recon_stack, pixel_um], outputs=nav_outs)
        btn_recon_prev.click(on_recon_prev, inputs=[recon_idx, recon_stack, pixel_um], outputs=nav_outs)
        btn_recon_next.click(on_recon_next, inputs=[recon_idx, recon_stack, pixel_um], outputs=nav_outs)

        btn_preset.click(
            on_apply_preset,
            inputs=[preset],
            outputs=[
                ring_method,
                snr,
                la_size,
                sm_size,
                drop_ratio,
                dim,
                speed,
                apply_log,
                pixel_shift,
                preview_row,
                output_dir,
                bh_strength,
                status,
            ],
        )
        btn_save.click(on_save_recipe, inputs=[recipe_name, *recon_controls], outputs=[status, preset])

        btn_preview.click(
            on_preview,
            inputs=[scan_dir, before_cache, before_key, pixel_um, *recon_controls],
            outputs=preview_outs,
        )
        btn_full.click(on_full, inputs=[scan_dir, *recon_controls], outputs=[status])

    return demo


def _pids_listening_on_port(port: int) -> list[int]:
    """Windows: PIDs with a LISTENING socket on this port."""
    import subprocess

    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")
    except (OSError, subprocess.CalledProcessError):
        return []

    needle = f":{port}"
    pids: set[int] = set()
    for line in out.splitlines():
        if "LISTENING" not in line.upper() or needle not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        local = parts[1]
        if not (local.endswith(needle) or local.endswith(f"]{needle}")):
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return sorted(pids)


def _reclaim_port(port: int) -> None:
    """End any previous process holding the toolkit port so startup always works."""
    import subprocess
    import time

    pids = [p for p in _pids_listening_on_port(port) if p != os.getpid()]
    if not pids:
        return
    print(f"Port {port} in use by PID(s) {pids} - closing previous toolkit instance...")
    for pid in pids:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
    for _ in range(20):
        if not _pids_listening_on_port(port):
            break
        time.sleep(0.25)


def _gradio_allowed_paths() -> list[str]:
    """Let Gradio read history/QC on any local drive (lab PCs often use D:\\Results)."""
    paths = [str(ROOT.resolve())]
    if os.name == "nt":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            if os.path.isdir(root):
                paths.append(root)
    else:
        paths.append("/")
    return paths


def main() -> None:
    port = 7860
    _reclaim_port(port)
    print(f"Opening GUI at http://127.0.0.1:{port}")
    print(f"Activity log file: {log_path()}")
    demo = build_app()
    launch_kwargs: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "inbrowser": True,
        "show_error": True,
    }
    if "allowed_paths" in inspect.signature(demo.launch).parameters:
        launch_kwargs["allowed_paths"] = _gradio_allowed_paths()
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
