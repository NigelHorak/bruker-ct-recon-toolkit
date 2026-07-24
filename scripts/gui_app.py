"""
Gradio GUI for Bruker CT Algotom toolkit.
Designed for fast alignment (~30s): cache mid-row once, then sub-second nudges.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recon_core import (  # noqa: E402
    FILTER_NAMES,
    RECON_METHODS,
    RECON_TYPES,
    RING_METHODS,
    AlignCache,
    Settings,
    auto_tune_pixel_shift,
    load_settings,
    prepare_align_cache,
    probe_scan_info,
    quick_align_preview,
    run_full,
    run_preview,
    save_yaml,
)
from lab_tools import (  # noqa: E402
    compare_ring_methods,
    preflight_scan,
    validate_alignment_rows,
)
from gui_style import GUI_CSS, build_theme  # noqa: E402
from gui_log import (  # noqa: E402
    ProgressLog,
    log_exception,
    log_line,
    log_path,
    safe_history,
    startup_banner,
)

PRESET_DIR = ROOT / "config" / "presets"
DEFAULT_CFG = ROOT / "config" / "default.yaml"


def _preset_choices() -> List[str]:
    names = ["default"]
    if PRESET_DIR.is_dir():
        names.extend(sorted(p.stem for p in PRESET_DIR.glob("*.yaml")))
    return names


def _load_preset(name: str) -> Settings:
    if name == "default":
        return load_settings(DEFAULT_CFG)
    path = PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        return load_settings(DEFAULT_CFG)
    return load_settings(path)


def _settings_from_ui(*ctrl) -> Settings:
    expected = 18
    if len(ctrl) != expected:
        raise ValueError(
            f"UI sent {len(ctrl)} control values, expected {expected}. "
            "Restart the toolkit (close black window, Start Toolkit.bat again)."
        )
    (
        ring_enable,
        ring_method,
        snr,
        la_size,
        sm_size,
        drop_ratio,
        dim,
        recon_type,
        method,
        filter_name,
        apply_log,
        num_iter,
        chunk_size,
        center_mode,
        center_value,
        pixel_shift,
        preview_row,
        output_dir,
    ) = ctrl
    center_mode_l = str(center_mode or "auto").lower()
    rtype = str(recon_type or "FBP").upper()
    if rtype not in ("FBP", "FDK"):
        rtype = "FBP"
    return Settings(
        ring_enable=bool(ring_enable),
        ring_method=str(ring_method or "remove_all_stripe"),
        snr=float(snr),
        la_size=int(la_size) if int(la_size) % 2 == 1 else int(la_size) + 1,
        sm_size=int(sm_size) if int(sm_size) % 2 == 1 else int(sm_size) + 1,
        drop_ratio=float(drop_ratio),
        dim=int(dim),
        recon_type=rtype,
        method=str(method or "FBP_CUDA"),
        filter_name=str(filter_name or "hann"),
        apply_log=bool(apply_log),
        num_iter=int(num_iter),
        chunk_size=int(chunk_size),
        center_mode=center_mode_l,
        center=float(center_value) if center_mode_l == "manual" else None,
        pixel_shift=float(pixel_shift or 0.0),
        preview_row=int(preview_row) if int(preview_row or -1) >= 0 else None,
        output_dir=(output_dir or "").strip() if output_dir is not None else "",
        save_preview=True,
    )


def ui_settings_tuple(s: Settings) -> Tuple[Any, ...]:
    return (
        s.ring_enable,
        s.ring_method,
        s.snr,
        s.la_size,
        s.sm_size,
        s.drop_ratio,
        s.dim,
        s.recon_type,
        s.method,
        s.filter_name,
        s.apply_log,
        s.num_iter,
        s.chunk_size,
        s.center_mode,
        0.0 if s.center is None else float(s.center),
        float(s.pixel_shift or 0.0),
        -1 if s.preview_row is None else int(s.preview_row),
        s.output_dir or "",
    )


def _nudge(shift: float, delta: float) -> float:
    return float(round(float(shift or 0.0) + delta, 3))


def _fmt_block(title: str, body: str, progress: Optional[ProgressLog] = None) -> str:
    parts = [title, body.strip()]
    if progress is not None:
        parts.append("--- progress ---")
        parts.append(progress.text())
    parts.append(f"(log file: {log_path()})")
    return "\n".join(parts)


def on_load_and_cache(scan_dir: str, preview_row: float):
    """Load folder, pull NRecon postalignment, cache mid-row for fast alignment."""
    scan_dir = (scan_dir or "").strip().strip('"')
    empty = []
    if not scan_dir:
        msg = _fmt_block("LOAD", "Enter a scan folder path first.")
        log_line("LOAD blocked: empty path")
        return "Enter a scan folder path.", 0, 0, -1, 0.0, None, None, 0.0, 0.0, empty, msg

    progress = ProgressLog(f"LOAD start: {scan_dir}")
    try:
        text, height, width, mid, post = probe_scan_info(scan_dir)
        progress(f"probe OK: {height}x{width} mid={mid} post={post}")
        row = mid if int(preview_row or -1) < 0 else int(preview_row)
        cache = prepare_align_cache(Path(scan_dir), preview_row=row, progress=progress)
        start_shift = float(cache.log_postalignment or post or 0.0)
        img, msg, base, eff = quick_align_preview(
            cache, start_shift, apply_log=True, save_history=True
        )
        gallery = safe_history(scan_dir)
        status = _fmt_block(
            "LOAD OK",
            f"{text}\nAlign cache ready. shift={start_shift:+.3f} from log.\n{msg}",
            progress,
        )
        return text, height, width, row, start_shift, cache, img, base, eff, gallery, status
    except Exception as exc:
        err = log_exception("Load scan", exc)
        return f"ERROR: {exc}", 0, 0, -1, 0.0, None, None, 0.0, 0.0, [], err


def on_quick_align(cache: Optional[AlignCache], pixel_shift: float, apply_log: bool):
    if cache is None:
        msg = _fmt_block("QUICK ALIGN", "Click Load scan first (builds align cache).")
        return None, 0.0, 0.0, float(pixel_shift or 0.0), [], msg
    progress = ProgressLog(f"QUICK ALIGN shift={float(pixel_shift or 0.0):+.3f}")
    try:
        img, msg, base, eff = quick_align_preview(
            cache, float(pixel_shift or 0.0), apply_log=bool(apply_log), save_history=True
        )
        gallery = safe_history(cache.scan_dir)
        status = _fmt_block("QUICK ALIGN OK", msg, progress)
        return img, base, eff, float(pixel_shift or 0.0), gallery, status
    except Exception as exc:
        return None, 0.0, 0.0, float(pixel_shift or 0.0), [], log_exception("Quick align", exc)


def on_nudge(cache, pixel_shift, apply_log, delta):
    new_shift = _nudge(pixel_shift, delta)
    log_line(f"NUDGE {delta:+.1f} -> {new_shift:+.3f}")
    img, base, eff, _, gallery, status = on_quick_align(cache, new_shift, apply_log)
    return new_shift, new_shift, img, base, eff, gallery, status


def on_auto_tune(cache: Optional[AlignCache], apply_log: bool):
    if cache is None:
        msg = _fmt_block("AUTO-TUNE", "Click Load scan first.")
        return 0.0, 0.0, None, 0.0, 0.0, [], msg
    progress = ProgressLog("AUTO-TUNE start")
    try:
        best, img, msg, base, eff = auto_tune_pixel_shift(
            cache, search=2.0, step=0.25, apply_log=bool(apply_log), progress=progress
        )
        gallery = safe_history(cache.scan_dir)
        status = _fmt_block("AUTO-TUNE OK", msg, progress)
        return best, best, img, base, eff, gallery, status
    except Exception as exc:
        return 0.0, 0.0, None, 0.0, 0.0, [], log_exception("Auto-tune", exc)


def on_full_preview(scan_dir: str, before_cache, before_key, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"FULL PREVIEW start: {scan_dir or '(empty)'}")
    if not scan_dir:
        return None, None, None, before_cache, before_key or "", [], _fmt_block(
            "PREVIEW", "Enter a scan folder path first."
        )
    try:
        progress(f"controls received: {len(ctrl)} values")
        settings = _settings_from_ui(*ctrl)
        progress(
            f"settings: type={settings.recon_type} method={settings.method} "
            f"rings={settings.ring_enable}/{settings.ring_method} shift={settings.pixel_shift:+.3f}"
        )
        result = run_preview(
            Path(scan_dir),
            settings,
            progress=progress,
            cached_before=before_cache,
            cached_before_key=before_key or "",
        )
        gallery = safe_history(scan_dir)
        status = _fmt_block(
            "PREVIEW OK",
            f"{result.message}\n"
            f"base={result.base_center:.3f} shift={result.pixel_shift:+.3f} "
            f"eff={result.center:.3f} | reused_before={result.before_reused}",
            progress,
        )
        return (
            result.display_raw,
            result.display_corr,
            result.display_diff,
            result.img_raw,
            result.before_key,
            gallery,
            status,
        )
    except Exception as exc:
        return (
            None,
            None,
            None,
            before_cache,
            before_key or "",
            [],
            log_exception("Full Preview", exc) + "\n\n" + progress.text(),
        )


def on_preflight(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"PREFLIGHT: {scan_dir or '(empty)'}")
    try:
        settings = _settings_from_ui(*ctrl)
        report = preflight_scan(scan_dir, settings, progress=progress)
        return _fmt_block("PREFLIGHT", report.text, progress)
    except Exception as exc:
        return log_exception("Preflight", exc)


def on_full(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"FULL RECON start: {scan_dir or '(empty)'}")
    if not scan_dir:
        return _fmt_block("FULL RECON", "Enter a scan folder path first.")
    try:
        settings = _settings_from_ui(*ctrl)
        report = preflight_scan(scan_dir, settings, progress=progress)
        result = run_full(Path(scan_dir), settings, progress=progress)
        return _fmt_block("FULL RECON OK", f"{report.text}\n\n{result.message}", progress)
    except Exception as exc:
        return log_exception("Full reconstruction", exc) + "\n\n" + progress.text()


def on_align_check(scan_dir: str, pixel_shift: float, apply_log: bool):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        msg = _fmt_block("ALIGN CHECK", "Enter a scan folder first.")
        return None, float(pixel_shift or 0.0), float(pixel_shift or 0.0), [], msg
    progress = ProgressLog("MULTI-ROW ALIGN CHECK")
    try:
        montage, msg, recommended = validate_alignment_rows(
            Path(scan_dir), float(pixel_shift or 0.0), apply_log=bool(apply_log), progress=progress
        )
        gallery = safe_history(scan_dir)
        status = _fmt_block("ALIGN CHECK OK", msg, progress)
        return montage, recommended, recommended, gallery, status
    except Exception as exc:
        return (
            None,
            float(pixel_shift or 0.0),
            float(pixel_shift or 0.0),
            [],
            log_exception("Multi-row align check", exc),
        )


def on_ring_compare(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"RING COMPARE: {scan_dir or '(empty)'}")
    if not scan_dir:
        return [], True, "remove_all_stripe", [], _fmt_block("RING COMPARE", "Enter a scan folder first.")
    try:
        settings = _settings_from_ui(*ctrl)
        gallery_tiles, report, win = compare_ring_methods(
            Path(scan_dir), settings, progress=progress
        )
        hist = safe_history(scan_dir)
        status = _fmt_block("RING COMPARE OK", report, progress)
        return gallery_tiles, win.ring_enable, win.ring_method, hist, status
    except Exception as exc:
        return [], True, "remove_all_stripe", [], log_exception("Ring compare", exc)


def on_save_recipe(recipe_name: str, *ctrl):
    import gradio as gr

    try:
        name = (recipe_name or "").strip()
        if not name:
            name = f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        settings = _settings_from_ui(*ctrl)
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        out = PRESET_DIR / f"{safe}.yaml"
        save_yaml(out, settings.to_config_dict())
        log_line(f"Saved recipe {out}")
        return _fmt_block("SAVE RECIPE OK", str(out)), gr.update(choices=_preset_choices(), value=safe)
    except Exception as exc:
        return log_exception("Save recipe", exc), gr.update()


def on_apply_preset(name: str):
    try:
        s = _load_preset(name)
        log_line(f"Applied preset: {name}")
        return (*ui_settings_tuple(s), _fmt_block("PRESET", f"Loaded: {name}"))
    except Exception as exc:
        # Keep current controls untouched by returning zeros-ish defaults + error
        s = load_settings(DEFAULT_CFG)
        return (*ui_settings_tuple(s), log_exception("Apply preset", exc))


def on_refresh_history(scan_dir: str):
    scan_dir = (scan_dir or "").strip().strip('"')
    try:
        gallery = safe_history(scan_dir)
        msg = _fmt_block("HISTORY", f"{len(gallery)} image(s) for {scan_dir or '(no folder)'}")
        return gallery, msg
    except Exception as exc:
        return [], log_exception("Refresh history", exc)

def build_app():
    import gradio as gr
    import inspect

    defaults = load_settings(DEFAULT_CFG)
    theme = build_theme()

    blocks_kwargs = {
        "title": "Bruker CT Algotom Toolkit",
        "theme": theme,
        "css": GUI_CSS,
    }
    if "fill_width" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["fill_width"] = True

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <div class="ct-header">
              <div>
                <div class="ct-brand">Bruker CT <span>Algotom</span></div>
                <div class="ct-tagline">
                  Align fast, kill rings, compare QC — load once, tune on the right, run when ready.
                </div>
              </div>
              <div class="ct-steps">1 Load · 2 Align · 3 Rings · 4 Preview · 5 Full</div>
            </div>
            """
        )

        align_cache = gr.State(None)
        before_cache = gr.State(None)
        before_key = gr.State("")
        height_state = gr.Number(value=0, visible=False)
        width_state = gr.Number(value=0, visible=False)

        # --- top: scan strip ---
        with gr.Row(elem_classes=["ct-panel"]):
            with gr.Column(scale=5):
                scan_dir = gr.Textbox(
                    label="Scan folder",
                    placeholder=r"D:\Data\MySample_scan   (TIFF projections + .log)",
                    elem_classes=["ct-mono"],
                )
            with gr.Column(scale=1, min_width=140):
                preview_row = gr.Number(value=-1, label="Row (-1=mid)", precision=0)
            with gr.Column(scale=1, min_width=160):
                btn_load = gr.Button("Load scan", variant="primary")

        scan_info = gr.Textbox(
            label="Scan",
            interactive=False,
            lines=2,
            elem_classes=["ct-mono"],
        )

        # Always-visible activity log (errors + progress land here)
        status = gr.Textbox(
            label="Activity / QC log (also printed in the black console + toolkit_gui.log)",
            lines=10,
            value=startup_banner(),
            elem_classes=["ct-mono", "ct-panel"],
        )

        # --- main: controls rail | viewer ---
        with gr.Row():
            # LEFT: controls
            with gr.Column(scale=2, min_width=340, elem_classes=["ct-panel", "ct-rail", "ct-compact"]):
                with gr.Tabs():
                    with gr.Tab("Align"):
                        gr.Markdown("Cached mid-row FBP — nudges stay fast.")
                        apply_log_align = gr.Checkbox(value=True, label="Log transform (align)")
                        pixel_shift = gr.Slider(
                            -5.0, 5.0, value=0.0, step=0.05, label="Pixel shift (postalignment)"
                        )
                        pixel_shift_num = gr.Number(value=0.0, label="Exact shift", precision=3)
                        with gr.Row():
                            btn_m05 = gr.Button("-0.5")
                            btn_m01 = gr.Button("-0.1")
                            btn_quick = gr.Button("Refresh")
                            btn_p01 = gr.Button("+0.1")
                            btn_p05 = gr.Button("+0.5")
                        with gr.Row():
                            btn_auto = gr.Button("Auto-tune", variant="primary")
                            btn_align_check = gr.Button("Multi-row check")
                            btn_reset = gr.Button("Reset log")
                        with gr.Row():
                            base_center_out = gr.Number(0, label="Base COR", interactive=False)
                            effective_center_out = gr.Number(0, label="Effective COR", interactive=False)

                    with gr.Tab("Rings"):
                        with gr.Row():
                            preset = gr.Dropdown(
                                choices=_preset_choices(), value="default", label="Preset"
                            )
                            btn_preset = gr.Button("Apply")
                        with gr.Row():
                            recipe_name = gr.Textbox(label="Save as", placeholder="my_sample")
                            btn_save = gr.Button("Save")
                        ring_enable = gr.Checkbox(value=defaults.ring_enable, label="Enable ring removal")
                        ring_method = gr.Dropdown(
                            choices=list(RING_METHODS), value=defaults.ring_method, label="Method"
                        )
                        snr = gr.Slider(1.0, 10.0, value=defaults.snr, step=0.1, label="snr")
                        la_size = gr.Slider(3, 151, value=defaults.la_size, step=2, label="la_size")
                        sm_size = gr.Slider(3, 101, value=defaults.sm_size, step=2, label="sm_size")
                        drop_ratio = gr.Slider(
                            0.0, 0.5, value=defaults.drop_ratio, step=0.01, label="drop_ratio"
                        )
                        dim = gr.Radio(choices=[1, 2], value=defaults.dim, label="dim")
                        btn_ring_cmp = gr.Button("Compare all methods (FBP)", variant="secondary")

                    with gr.Tab("Algorithm"):
                        recon_type = gr.Radio(
                            choices=list(RECON_TYPES), value=defaults.recon_type, label="Family"
                        )
                        method = gr.Dropdown(
                            choices=list(RECON_METHODS), value=defaults.method, label="FBP method"
                        )
                        filter_name = gr.Dropdown(
                            choices=list(FILTER_NAMES), value=defaults.filter_name, label="Filter"
                        )
                        apply_log = gr.Checkbox(value=defaults.apply_log, label="Apply log")
                        # Keep these OUT of a closed accordion so Gradio always sends values.
                        num_iter = gr.Slider(
                            1, 500, value=defaults.num_iter, step=1, label="Iterations"
                        )
                        chunk_size = gr.Slider(
                            1, 128, value=defaults.chunk_size, step=1, label="Chunk size"
                        )
                        center_mode = gr.Radio(
                            choices=["auto", "manual"], value="auto", label="Base COR mode"
                        )
                        center_value = gr.Number(value=0.0, label="Manual base COR", precision=3)
                        output_dir = gr.Textbox(value="", label="Output override")

                    with gr.Tab("Run"):
                        gr.Markdown(
                            "Preview reuses BEFORE when only rings change.  \n"
                            "Full recon always writes a **new timestamped** folder."
                        )
                        btn_preview = gr.Button("Full Preview", variant="primary")
                        btn_preflight = gr.Button("Preflight check")
                        btn_full = gr.Button("Run full reconstruction", variant="stop")

            # RIGHT: persistent viewer
            with gr.Column(scale=3, min_width=480, elem_classes=["ct-panel", "ct-viewer"]):
                with gr.Tabs():
                    with gr.Tab("Align view"):
                        align_img = gr.Image(
                            label="Quick align (FBP, no rings)",
                            type="numpy",
                            height=460,
                        )
                    with gr.Tab("Rings QC"):
                        with gr.Row():
                            img_before = gr.Image(label="BEFORE", type="numpy", height=360)
                            img_after = gr.Image(label="AFTER", type="numpy", height=360)
                        img_diff = gr.Image(label="|AFTER - BEFORE|", type="numpy", height=280)
                    with gr.Tab("Method bake-off"):
                        compare_gallery = gr.Gallery(
                            label="Ring methods",
                            columns=3,
                            height=480,
                        )
                    with gr.Tab("History"):
                        btn_hist = gr.Button("Refresh history")
                        history_gallery = gr.Gallery(
                            label="Saved runs (settings in captions)",
                            columns=3,
                            height=480,
                        )

        controls = [
            ring_enable,
            ring_method,
            snr,
            la_size,
            sm_size,
            drop_ratio,
            dim,
            recon_type,
            method,
            filter_name,
            apply_log,
            num_iter,
            chunk_size,
            center_mode,
            center_value,
            pixel_shift,
            preview_row,
            output_dir,
        ]

        # --- wiring (every action updates Activity log) ---
        pixel_shift.release(lambda v: float(v), inputs=pixel_shift, outputs=pixel_shift_num)
        pixel_shift_num.change(lambda v: float(v or 0.0), inputs=pixel_shift_num, outputs=pixel_shift)

        btn_load.click(
            on_load_and_cache,
            inputs=[scan_dir, preview_row],
            outputs=[
                scan_info,
                height_state,
                width_state,
                preview_row,
                pixel_shift,
                align_cache,
                align_img,
                base_center_out,
                effective_center_out,
                history_gallery,
                status,
            ],
        ).then(lambda v: float(v or 0.0), inputs=pixel_shift, outputs=pixel_shift_num)

        btn_quick.click(
            on_quick_align,
            inputs=[align_cache, pixel_shift, apply_log_align],
            outputs=[
                align_img,
                base_center_out,
                effective_center_out,
                pixel_shift_num,
                history_gallery,
                status,
            ],
        )
        for btn, delta in ((btn_m05, -0.5), (btn_m01, -0.1), (btn_p01, 0.1), (btn_p05, 0.5)):
            btn.click(
                lambda c, s, a, d=delta: on_nudge(c, s, a, d),
                inputs=[align_cache, pixel_shift, apply_log_align],
                outputs=[
                    pixel_shift,
                    pixel_shift_num,
                    align_img,
                    base_center_out,
                    effective_center_out,
                    history_gallery,
                    status,
                ],
            )
        btn_auto.click(
            on_auto_tune,
            inputs=[align_cache, apply_log_align],
            outputs=[
                pixel_shift,
                pixel_shift_num,
                align_img,
                base_center_out,
                effective_center_out,
                history_gallery,
                status,
            ],
        )
        btn_align_check.click(
            on_align_check,
            inputs=[scan_dir, pixel_shift, apply_log_align],
            outputs=[align_img, pixel_shift, pixel_shift_num, history_gallery, status],
        )

        def _reset_shift(cache: Optional[AlignCache]):
            try:
                val = float(cache.log_postalignment) if cache is not None else 0.0
                img, base, eff, _, gallery, st = on_quick_align(cache, val, True)
                return val, val, img, base, eff, gallery, st
            except Exception as exc:
                return 0.0, 0.0, None, 0.0, 0.0, [], log_exception("Reset shift", exc)

        btn_reset.click(
            _reset_shift,
            inputs=[align_cache],
            outputs=[
                pixel_shift,
                pixel_shift_num,
                align_img,
                base_center_out,
                effective_center_out,
                history_gallery,
                status,
            ],
        )

        btn_preset.click(on_apply_preset, inputs=[preset], outputs=[*controls, status])
        btn_preview.click(
            on_full_preview,
            inputs=[scan_dir, before_cache, before_key, *controls],
            outputs=[
                img_before,
                img_after,
                img_diff,
                before_cache,
                before_key,
                history_gallery,
                status,
            ],
        )
        btn_ring_cmp.click(
            on_ring_compare,
            inputs=[scan_dir, *controls],
            outputs=[compare_gallery, ring_enable, ring_method, history_gallery, status],
        )
        btn_preflight.click(on_preflight, inputs=[scan_dir, *controls], outputs=[status])
        btn_full.click(on_full, inputs=[scan_dir, *controls], outputs=[status])
        btn_save.click(on_save_recipe, inputs=[recipe_name, *controls], outputs=[status, preset])
        btn_hist.click(on_refresh_history, inputs=[scan_dir], outputs=[history_gallery, status])

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
        # Match "127.0.0.1:7860" / "0.0.0.0:7860" / "[::]:7860", not ":78600"
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
    # Brief wait so Windows releases the socket
    for _ in range(20):
        if not _pids_listening_on_port(port):
            break
        time.sleep(0.25)


def main() -> None:
    port = 7860
    _reclaim_port(port)
    print(f"Opening GUI at http://127.0.0.1:{port}")
    print(f"Activity log file: {log_path()}")
    demo = build_app()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
