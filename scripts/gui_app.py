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
from history_store import list_history_gallery  # noqa: E402
from lab_tools import (  # noqa: E402
    compare_ring_methods,
    preflight_scan,
    validate_alignment_rows,
)
from gui_style import GUI_CSS, build_theme  # noqa: E402

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


def _settings_from_ui(
    ring_enable: bool,
    ring_method: str,
    snr: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    recon_type: str,
    method: str,
    filter_name: str,
    apply_log: bool,
    num_iter: int,
    chunk_size: int,
    center_mode: str,
    center_value: float,
    pixel_shift: float,
    preview_row: int,
    output_dir: str,
) -> Settings:
    center_mode_l = str(center_mode).lower()
    rtype = str(recon_type).upper()
    if rtype not in ("FBP", "FDK"):
        rtype = "FBP"
    return Settings(
        ring_enable=bool(ring_enable),
        ring_method=ring_method,
        snr=float(snr),
        la_size=int(la_size) if int(la_size) % 2 == 1 else int(la_size) + 1,
        sm_size=int(sm_size) if int(sm_size) % 2 == 1 else int(sm_size) + 1,
        drop_ratio=float(drop_ratio),
        dim=int(dim),
        recon_type=rtype,
        method=method,
        filter_name=filter_name,
        apply_log=bool(apply_log),
        num_iter=int(num_iter),
        chunk_size=int(chunk_size),
        center_mode=center_mode_l,
        center=float(center_value) if center_mode_l == "manual" else None,
        pixel_shift=float(pixel_shift or 0.0),
        preview_row=int(preview_row) if int(preview_row) >= 0 else None,
        output_dir=(output_dir or "").strip(),
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


def on_load_and_cache(scan_dir: str, preview_row: float):
    """Load folder, pull NRecon postalignment, cache mid-row for fast alignment."""
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        empty = []
        return "Enter a scan folder path.", 0, 0, -1, 0.0, None, None, "No folder yet.", 0.0, 0.0, empty
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        text, height, width, mid, post = probe_scan_info(scan_dir)
        row = mid if int(preview_row) < 0 else int(preview_row)
        cache = prepare_align_cache(Path(scan_dir), preview_row=row, progress=progress)
        start_shift = float(cache.log_postalignment or post or 0.0)
        img, msg, base, eff = quick_align_preview(cache, start_shift, apply_log=True, save_history=True)
        status = (
            f"{text}\n"
            f"Align cache ready. Starting shift={start_shift:+.3f} from log.\n"
            f"{msg}\n" + "\n".join(logs[-6:])
        )
        gallery = list_history_gallery(Path(scan_dir))
        return text, height, width, row, start_shift, cache, img, status, base, eff, gallery
    except Exception as exc:
        return f"ERROR: {exc}", 0, 0, -1, 0.0, None, None, f"LOAD FAILED: {exc}", 0.0, 0.0, []


def on_quick_align(cache: Optional[AlignCache], pixel_shift: float, apply_log: bool):
    if cache is None:
        return None, "Click Load folder first (builds align cache).", 0.0, 0.0, float(pixel_shift or 0.0), []
    try:
        img, msg, base, eff = quick_align_preview(
            cache, float(pixel_shift or 0.0), apply_log=bool(apply_log), save_history=True
        )
        gallery = list_history_gallery(Path(cache.scan_dir))
        return img, msg, base, eff, float(pixel_shift or 0.0), gallery
    except Exception as exc:
        return None, f"QUICK ALIGN FAILED: {exc}", 0.0, 0.0, float(pixel_shift or 0.0), []


def on_nudge(cache, pixel_shift, apply_log, delta):
    new_shift = _nudge(pixel_shift, delta)
    img, msg, base, eff, _, gallery = on_quick_align(cache, new_shift, apply_log)
    return new_shift, new_shift, img, msg, base, eff, gallery


def on_auto_tune(cache: Optional[AlignCache], apply_log: bool):
    if cache is None:
        return 0.0, 0.0, None, "Click Load folder first.", 0.0, 0.0, []
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        best, img, msg, base, eff = auto_tune_pixel_shift(
            cache, search=2.0, step=0.25, apply_log=bool(apply_log), progress=progress
        )
        gallery = list_history_gallery(Path(cache.scan_dir))
        return best, best, img, msg + "\n" + "\n".join(logs[-8:]), base, eff, gallery
    except Exception as exc:
        return 0.0, 0.0, None, f"AUTO-TUNE FAILED: {exc}", 0.0, 0.0, []


def on_full_preview(scan_dir: str, before_cache, before_key, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        settings = _settings_from_ui(*ctrl)
        result = run_preview(
            Path(scan_dir),
            settings,
            progress=progress,
            cached_before=before_cache,
            cached_before_key=before_key or "",
        )
        status = (
            f"{result.message}\n"
            f"base={result.base_center:.3f} shift={result.pixel_shift:+.3f} "
            f"eff={result.center:.3f} | reused_before={result.before_reused}\n"
            + "\n".join(logs[-8:])
        )
        gallery = list_history_gallery(Path(scan_dir))
        return (
            result.display_raw,
            result.display_corr,
            result.display_diff,
            status,
            result.img_raw,
            result.before_key,
            gallery,
        )
    except Exception as exc:
        return None, None, None, f"PREVIEW FAILED: {exc}\n" + "\n".join(logs[-12:]), before_cache, before_key, []


def on_preflight(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    try:
        settings = _settings_from_ui(*ctrl)
        report = preflight_scan(scan_dir, settings)
        return report.text
    except Exception as exc:
        return f"PREFLIGHT FAILED: {exc}"


def on_full(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        settings = _settings_from_ui(*ctrl)
        # Soft gate: always show preflight notes in the log first
        report = preflight_scan(scan_dir, settings, progress=progress)
        result = run_full(Path(scan_dir), settings, progress=progress)
        return f"{report.text}\n\n{result.message}\n" + "\n".join(logs[-20:])
    except Exception as exc:
        return f"FULL RECON FAILED: {exc}\n" + "\n".join(logs[-20:])


def on_align_check(scan_dir: str, pixel_shift: float, apply_log: bool):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        return None, "Enter a scan folder first.", float(pixel_shift or 0.0), float(pixel_shift or 0.0), []
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        montage, msg, recommended = validate_alignment_rows(
            Path(scan_dir), float(pixel_shift or 0.0), apply_log=bool(apply_log), progress=progress
        )
        gallery = list_history_gallery(Path(scan_dir))
        return montage, msg + "\n" + "\n".join(logs[-8:]), recommended, recommended, gallery
    except Exception as exc:
        return None, f"ALIGN CHECK FAILED: {exc}", float(pixel_shift or 0.0), float(pixel_shift or 0.0), []


def on_ring_compare(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        settings = _settings_from_ui(*ctrl)
        gallery_tiles, report, win = compare_ring_methods(Path(scan_dir), settings, progress=progress)
        hist = list_history_gallery(Path(scan_dir))
        # Apply winning ring method into UI controls
        return (
            gallery_tiles,
            report + "\n" + "\n".join(logs[-10:]),
            win.ring_enable,
            win.ring_method,
            hist,
        )
    except Exception as exc:
        return [], f"RING COMPARE FAILED: {exc}", True, "remove_all_stripe", []


def on_save_recipe(recipe_name: str, *ctrl):
    name = (recipe_name or "").strip()
    if not name:
        name = f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    settings = _settings_from_ui(*ctrl)
    out = PRESET_DIR / f"{safe}.yaml"
    save_yaml(out, settings.to_config_dict())
    import gradio as gr

    return f"Saved recipe: {out}", gr.update(choices=_preset_choices())


def on_apply_preset(name: str):
    s = _load_preset(name)
    return (*ui_settings_tuple(s), f"Loaded preset: {name}")


def on_refresh_history(scan_dir: str):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        return []
    return list_history_gallery(Path(scan_dir))


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
                btn_load = gr.Button("Load scan", variant="primary", size="lg")

        scan_info = gr.Textbox(
            label="Scan",
            interactive=False,
            lines=2,
            elem_classes=["ct-mono"],
            max_lines=3,
        )

        # --- main: controls rail | viewer ---
        with gr.Row(equal_height=False):
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
                            btn_m05 = gr.Button("-0.5", size="sm")
                            btn_m01 = gr.Button("-0.1", size="sm")
                            btn_quick = gr.Button("Refresh", size="sm")
                            btn_p01 = gr.Button("+0.1", size="sm")
                            btn_p05 = gr.Button("+0.5", size="sm")
                        with gr.Row():
                            btn_auto = gr.Button("Auto-tune", variant="primary")
                            btn_align_check = gr.Button("Multi-row check")
                            btn_reset = gr.Button("Reset log")
                        with gr.Row():
                            base_center_out = gr.Number(0, label="Base COR", interactive=False)
                            effective_center_out = gr.Number(0, label="Effective COR", interactive=False)
                        align_status = gr.Textbox(
                            label="Align log", lines=4, elem_classes=["ct-mono"], max_lines=6
                        )

                    with gr.Tab("Rings"):
                        with gr.Row():
                            preset = gr.Dropdown(
                                choices=_preset_choices(), value="default", label="Preset", scale=2
                            )
                            btn_preset = gr.Button("Apply", scale=1)
                        with gr.Row():
                            recipe_name = gr.Textbox(label="Save as", placeholder="my_sample", scale=2)
                            btn_save = gr.Button("Save", scale=1)
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
                        with gr.Accordion("Advanced", open=False):
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
                with gr.Tabs() as viewer_tabs:
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
                            object_fit="contain",
                            preview=True,
                        )
                    with gr.Tab("History"):
                        btn_hist = gr.Button("Refresh history", size="sm")
                        history_gallery = gr.Gallery(
                            label="Saved runs (settings in captions)",
                            columns=3,
                            height=480,
                            object_fit="contain",
                            preview=True,
                        )

        status = gr.Textbox(
            label="Activity / QC log",
            lines=6,
            max_lines=12,
            elem_classes=["ct-mono", "ct-panel"],
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

        # --- wiring ---
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
                align_status,
                base_center_out,
                effective_center_out,
                history_gallery,
            ],
        ).then(lambda v: float(v or 0.0), inputs=pixel_shift, outputs=pixel_shift_num)

        btn_quick.click(
            on_quick_align,
            inputs=[align_cache, pixel_shift, apply_log_align],
            outputs=[
                align_img,
                align_status,
                base_center_out,
                effective_center_out,
                pixel_shift_num,
                history_gallery,
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
                    align_status,
                    base_center_out,
                    effective_center_out,
                    history_gallery,
                ],
            )
        btn_auto.click(
            on_auto_tune,
            inputs=[align_cache, apply_log_align],
            outputs=[
                pixel_shift,
                pixel_shift_num,
                align_img,
                align_status,
                base_center_out,
                effective_center_out,
                history_gallery,
            ],
        )
        btn_align_check.click(
            on_align_check,
            inputs=[scan_dir, pixel_shift, apply_log_align],
            outputs=[align_img, align_status, pixel_shift, pixel_shift_num, history_gallery],
        )

        def _reset_shift(cache: Optional[AlignCache]):
            val = float(cache.log_postalignment) if cache is not None else 0.0
            img, msg, base, eff, _, gallery = on_quick_align(cache, val, True)
            return val, val, img, msg, base, eff, gallery

        btn_reset.click(
            _reset_shift,
            inputs=[align_cache],
            outputs=[
                pixel_shift,
                pixel_shift_num,
                align_img,
                align_status,
                base_center_out,
                effective_center_out,
                history_gallery,
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
                status,
                before_cache,
                before_key,
                history_gallery,
            ],
        )
        btn_ring_cmp.click(
            on_ring_compare,
            inputs=[scan_dir, *controls],
            outputs=[compare_gallery, status, ring_enable, ring_method, history_gallery],
        )
        btn_preflight.click(on_preflight, inputs=[scan_dir, *controls], outputs=[status])
        btn_full.click(on_full, inputs=[scan_dir, *controls], outputs=[status])
        btn_save.click(on_save_recipe, inputs=[recipe_name, *controls], outputs=[status, preset])
        btn_hist.click(on_refresh_history, inputs=[scan_dir], outputs=[history_gallery])

        # silence unused warning if Gradio version lacks tab select
        _ = viewer_tabs

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
    demo = build_app()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
