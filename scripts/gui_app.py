"""
Gradio GUI for Bruker CT Algotom toolkit — partner-friendly workstation UI.
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
    auto_tune_pixel_shift,
    load_settings,
    prepare_align_cache,
    probe_scan_info,
    quick_align_preview,
    run_full,
    run_preview,
    save_yaml,
)
from history_store import list_history_entries  # noqa: E402
from lab_tools import compare_ring_methods, validate_alignment_rows  # noqa: E402
from gui_style import GUI_CSS, build_theme  # noqa: E402
from gui_log import (  # noqa: E402
    ProgressLog,
    log_exception,
    log_line,
    log_path,
    startup_banner,
)
from gui_labels import (  # noqa: E402
    FILTER_TO_CODE,
    FILTER_UI,
    INFO,
    RING_METHOD_TO_CODE,
    RING_METHOD_UI,
    SHIFT_MAX,
    SHIFT_MIN,
    SPEED_TO_CODE,
    SPEED_UI,
    filter_label,
    ring_label,
    speed_label,
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


def _clamp_shift(v: float) -> float:
    return float(max(SHIFT_MIN, min(SHIFT_MAX, round(float(v or 0.0), 3))))


def _settings_from_ui(
    ring_enable: bool,
    ring_method_label: str,
    snr: float,
    la_size: int,
    sm_size: int,
    drop_ratio: float,
    dim: int,
    speed_label_v: str,
    filter_label_v: str,
    apply_log: bool,
    pixel_shift: float,
    preview_row: int,
    output_dir: str,
) -> Settings:
    code_ring = RING_METHOD_TO_CODE.get(str(ring_method_label), "remove_all_stripe")
    if code_ring == "none":
        ring_enable = False
    rtype = SPEED_TO_CODE.get(str(speed_label_v), "FBP")
    fcode = FILTER_TO_CODE.get(str(filter_label_v), "hann")
    la = int(la_size)
    sm = int(sm_size)
    return Settings(
        ring_enable=bool(ring_enable) and code_ring != "none",
        ring_method=code_ring,
        snr=float(snr),
        la_size=la if la % 2 == 1 else la + 1,
        sm_size=sm if sm % 2 == 1 else sm + 1,
        drop_ratio=float(drop_ratio),
        dim=int(dim),
        recon_type=rtype,
        method="FBP_CUDA",
        filter_name=fcode,
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


def ui_settings_tuple(s: Settings) -> Tuple[Any, ...]:
    return (
        bool(s.ring_enable) and s.ring_method != "none",
        ring_label(s.ring_method),
        s.snr,
        s.la_size,
        s.sm_size,
        s.drop_ratio,
        s.dim,
        speed_label(s.recon_type),
        filter_label(s.filter_name),
        s.apply_log,
        _clamp_shift(s.pixel_shift or 0.0),
        -1 if s.preview_row is None else int(s.preview_row),
        s.output_dir or "",
    )


def _nudge(shift: float, delta: float) -> float:
    return _clamp_shift(float(shift or 0.0) + delta)


def _gallery_and_entries(scan_dir: str):
    entries = []
    try:
        if scan_dir:
            entries = list_history_entries(Path(scan_dir))
    except Exception as exc:
        log_exception("history entries", exc)
        entries = []
    gallery = [(e["image"], e["caption"]) for e in entries]
    return gallery, entries


def _pick_view(choice: str, align, before, after, diff):
    c = (choice or "Align check").lower()
    if "before" in c:
        return before if before is not None else align
    if "after" in c or "cleaned" in c:
        return after if after is not None else align
    if "diff" in c or "changed" in c:
        return diff if diff is not None else (after if after is not None else align)
    return align


def on_load_and_cache(scan_dir: str, preview_row: float, view_choice: str):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        msg = "Enter a scan folder path first."
        log_line("LOAD blocked: empty path")
        return "Enter a scan folder path.", -1, 0.0, None, None, None, None, None, 0.0, 0.0, None, [], [], msg

    path = Path(scan_dir)
    if path.is_file() and path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }:
        try:
            from PIL import Image

            arr = np.asarray(Image.open(path).convert("RGB"))
            status = f"Loaded.\n{path.name}"
            return (
                str(path),
                -1,
                0.0,
                None,
                arr,
                None,
                None,
                None,
                0.0,
                0.0,
                arr,
                [],
                [],
                status,
            )
        except Exception as exc:
            return (
                f"ERROR: {exc}",
                -1,
                0.0,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0.0,
                None,
                [],
                [],
                log_exception("Load scan", exc),
            )

    progress = ProgressLog(f"LOAD: {scan_dir}")
    try:
        text, height, width, mid, post = probe_scan_info(scan_dir)
        row = mid if int(preview_row or -1) < 0 else int(preview_row)
        cache = prepare_align_cache(Path(scan_dir), preview_row=row, progress=progress)
        start_shift = _clamp_shift(cache.log_postalignment or post or 0.0)
        img, msg, base, eff = quick_align_preview(
            cache, start_shift, apply_log=True, save_history=True
        )
        gallery, entries = _gallery_and_entries(scan_dir)
        status = f"Loaded.\n{text}\nStarting alignment from Bruker log value {start_shift:+.3f}.\n{msg}\n{progress.text()}"
        view = _pick_view(view_choice, img, None, None, None)
        return text, row, start_shift, cache, img, None, None, None, base, eff, view, gallery, entries, status
    except Exception as exc:
        return (
            f"ERROR: {exc}",
            -1,
            0.0,
            None,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
            None,
            [],
            [],
            log_exception("Load scan", exc),
        )


def on_quick_align(cache, pixel_shift, apply_log, view_choice, before, after, diff):
    if cache is None:
        return None, 0.0, 0.0, _clamp_shift(pixel_shift), None, [], [], "Click Load scan first."
    shift = _clamp_shift(pixel_shift)
    progress = ProgressLog(f"ALIGN refresh shift={shift:+.3f}")
    try:
        img, msg, base, eff = quick_align_preview(
            cache, shift, apply_log=bool(apply_log), save_history=True
        )
        gallery, entries = _gallery_and_entries(cache.scan_dir)
        view = _pick_view(view_choice, img, before, after, diff)
        return img, base, eff, shift, view, gallery, entries, f"{msg}\n{progress.text()}"
    except Exception as exc:
        return None, 0.0, 0.0, shift, None, [], [], log_exception("Align refresh", exc)


def on_nudge(cache, pixel_shift, apply_log, delta, view_choice, before, after, diff):
    new_shift = _nudge(pixel_shift, delta)
    log_line(f"NUDGE {delta:+.1f} -> {new_shift:+.3f}")
    img, base, eff, shift, view, gallery, entries, status = on_quick_align(
        cache, new_shift, apply_log, view_choice, before, after, diff
    )
    return shift, shift, img, base, eff, view, gallery, entries, status


# After Auto-find best: try these ring cleaners (off + 3 recipes)
QUICK_RING_METHODS = [
    "none",
    "remove_all_stripe",
    "remove_stripe_based_sorting",
    "remove_large_stripe",
]


def on_auto_tune(cache, apply_log, view_choice, before, after, diff, snr, la_size, sm_size, drop_ratio, dim):
    if cache is None:
        return (
            0.0,
            0.0,
            None,
            0.0,
            0.0,
            None,
            None,
            [],
            [],
            [],
            True,
            ring_label("remove_all_stripe"),
            "Click Load scan first.",
        )
    progress = ProgressLog("AUTO-FIND: trying many shifts, then a short ring bake-off")
    try:
        best, img, msg, base, eff = auto_tune_pixel_shift(
            cache, search=5.0, step=0.25, apply_log=bool(apply_log), progress=progress
        )
        best = _clamp_shift(best)

        # Short ring bake-off with the new alignment
        settings = Settings(
            ring_enable=True,
            ring_method="remove_all_stripe",
            snr=float(snr),
            la_size=int(la_size) if int(la_size) % 2 == 1 else int(la_size) + 1,
            sm_size=int(sm_size) if int(sm_size) % 2 == 1 else int(sm_size) + 1,
            drop_ratio=float(drop_ratio),
            dim=int(dim),
            recon_type="FBP",
            method="FBP_CUDA",
            filter_name="hann",
            apply_log=bool(apply_log),
            pixel_shift=best,
            preview_row=int(cache.row),
            center_mode="auto",
        )
        tiles, report, win, win_img = compare_ring_methods(
            Path(cache.scan_dir),
            settings,
            progress=progress,
            methods=list(QUICK_RING_METHODS),
        )
        gallery, entries = _gallery_and_entries(cache.scan_dir)
        view = _pick_view(view_choice or "After cleanup", img, img, win_img, diff)
        status = (
            f"{msg}\n\nThen tried {len(QUICK_RING_METHODS)} ring cleaners:\n{report}\n"
            f"{progress.text()}"
        )
        return (
            best,
            best,
            img,
            base,
            eff,
            view,
            win_img,
            tiles,
            gallery,
            entries,
            win.ring_enable,
            ring_label(win.ring_method),
            status,
        )
    except Exception as exc:
        return (
            0.0,
            0.0,
            None,
            0.0,
            0.0,
            None,
            None,
            [],
            [],
            [],
            True,
            ring_label("remove_all_stripe"),
            log_exception("Auto-find best", exc),
        )


def on_align_check(scan_dir, pixel_shift, apply_log, view_choice, before, after, diff):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        return None, _clamp_shift(pixel_shift), _clamp_shift(pixel_shift), None, [], [], "Enter a scan folder first."
    progress = ProgressLog("Checking top / middle / bottom agreement")
    try:
        montage, msg, recommended = validate_alignment_rows(
            Path(scan_dir), _clamp_shift(pixel_shift), apply_log=bool(apply_log), progress=progress
        )
        recommended = _clamp_shift(recommended)
        gallery, entries = _gallery_and_entries(scan_dir)
        view = _pick_view(view_choice, montage, before, after, diff)
        return montage, recommended, recommended, view, gallery, entries, f"{msg}\n{progress.text()}"
    except Exception as exc:
        return None, _clamp_shift(pixel_shift), _clamp_shift(pixel_shift), None, [], [], log_exception(
            "Multi-row check", exc
        )


def on_full_preview(scan_dir, before_cache, before_key, view_choice, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"PREVIEW: {scan_dir or '(empty)'}")
    if not scan_dir:
        return None, None, None, before_cache, before_key or "", None, [], [], "Enter a scan folder first."
    try:
        settings = _settings_from_ui(*ctrl)
        result = run_preview(
            Path(scan_dir),
            settings,
            progress=progress,
            cached_before=before_cache,
            cached_before_key=before_key or "",
        )
        gallery, entries = _gallery_and_entries(scan_dir)
        view = _pick_view(
            view_choice or "After cleanup",
            None,
            result.display_raw,
            result.display_corr,
            result.display_diff,
        )
        status = f"{result.message}\n{progress.text()}"
        return (
            result.display_raw,
            result.display_corr,
            result.display_diff,
            result.img_raw,
            result.before_key,
            view,
            gallery,
            entries,
            status,
        )
    except Exception as exc:
        return (
            None,
            None,
            None,
            before_cache,
            before_key or "",
            None,
            [],
            [],
            log_exception("Preview", exc) + "\n" + progress.text(),
        )


def on_full(scan_dir, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog(f"FULL RECON: {scan_dir or '(empty)'}")
    if not scan_dir:
        return "Enter a scan folder first."
    try:
        settings = _settings_from_ui(*ctrl)
        result = run_full(Path(scan_dir), settings, progress=progress)
        return f"{result.message}\nSaved under the scan's algotom folder.\n{progress.text()}"
    except Exception as exc:
        return log_exception("Full reconstruction", exc) + "\n" + progress.text()


def on_ring_compare(scan_dir, view_choice, align_img, before, after, diff, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog("Trying every ring cleaner on one slice")
    if not scan_dir:
        return [], True, ring_label("remove_all_stripe"), None, [], [], "Enter a scan folder first."
    try:
        settings = _settings_from_ui(*ctrl)
        tiles, report, win, win_img = compare_ring_methods(
            Path(scan_dir), settings, progress=progress
        )
        gallery, entries = _gallery_and_entries(scan_dir)
        view = _pick_view(view_choice, win_img, before, after, diff)
        return (
            tiles,
            win.ring_enable,
            ring_label(win.ring_method),
            view,
            gallery,
            entries,
            f"{report}\n{progress.text()}",
        )
    except Exception as exc:
        return [], True, ring_label("remove_all_stripe"), None, [], [], log_exception("Ring compare", exc)


def on_save_recipe(recipe_name: str, *ctrl):
    import gradio as gr

    try:
        name = (recipe_name or "").strip() or f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        settings = _settings_from_ui(*ctrl)
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        out = PRESET_DIR / f"{safe}.yaml"
        save_yaml(out, settings.to_config_dict())
        log_line(f"Saved recipe {out}")
        return f"Saved recipe to config/presets/{safe}.yaml", gr.update(choices=_preset_choices(), value=safe)
    except Exception as exc:
        return log_exception("Save recipe", exc), gr.update()


def on_apply_preset(name: str):
    try:
        s = _load_preset(name)
        log_line(f"Applied preset: {name}")
        return (*ui_settings_tuple(s), f"Loaded preset: {name} (from config/presets)")
    except Exception as exc:
        s = load_settings(DEFAULT_CFG)
        return (*ui_settings_tuple(s), log_exception("Apply preset", exc))


def on_view_choice(choice, align, before, after, diff):
    return _pick_view(choice, align, before, after, diff)


def on_history_select(evt, entries):
    """Click a history thumbnail -> show image + restore its settings into the form."""
    import gradio as gr

    try:
        if not entries:
            return (None, *ui_settings_tuple(load_settings(DEFAULT_CFG)), "No history yet.")
        idx = int(getattr(evt, "index", 0) or 0)
        if idx < 0 or idx >= len(entries):
            return (None, *ui_settings_tuple(load_settings(DEFAULT_CFG)), "Bad history index.")
        item = entries[idx]
        cfg = item.get("settings") or {}
        s = Settings.from_config_dict(cfg) if cfg else load_settings(DEFAULT_CFG)
        vals = ui_settings_tuple(s)
        return (item.get("image"), *vals, f"Restored settings from:\n{item.get('caption', '')}")
    except Exception as exc:
        return (None, *ui_settings_tuple(load_settings(DEFAULT_CFG)), log_exception("History click", exc))


def build_app():
    import gradio as gr

    defaults = load_settings(DEFAULT_CFG)
    ring_choices = [x[0] for x in RING_METHOD_UI]
    filter_choices = [x[0] for x in FILTER_UI]
    speed_choices = [x[0] for x in SPEED_UI]

    def _dropdown(**kwargs):
        # Gradio versions differ on allow_custom_value
        try:
            return gr.Dropdown(**kwargs, allow_custom_value=False)
        except TypeError:
            kwargs.pop("allow_custom_value", None)
            return gr.Dropdown(**kwargs)

    blocks_kwargs: Dict[str, Any] = {"title": "Bruker CT Algotom Toolkit"}
    sig = inspect.signature(gr.Blocks.__init__)
    if "fill_width" in sig.parameters:
        blocks_kwargs["fill_width"] = True
    # Gradio <6 still accepts theme/css on Blocks
    if "theme" in sig.parameters:
        theme = build_theme()
        if theme is not None:
            blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = GUI_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <div class="ct-header">
              <div>
                <div class="ct-brand">Bruker CT <span>Algotom</span></div>
                <div class="ct-sub">Ring cleanup &amp; alignment for Bruker / SkyScan scans</div>
              </div>
            </div>
            """
        )

        align_cache = gr.State(None)
        before_cache = gr.State(None)
        before_key = gr.State("")
        align_img_state = gr.State(None)
        before_state = gr.State(None)
        after_state = gr.State(None)
        diff_state = gr.State(None)
        history_entries = gr.State([])

        with gr.Row(elem_classes=["ct-panel"]):
            scan_dir = gr.Textbox(
                label="Scan folder",
                placeholder=r"D:\Results\...\MyScan   (projection TIFFs + .log)",
                scale=5,
                elem_classes=["ct-mono"],
            )
            preview_row = gr.Number(
                value=-1,
                label="Slice row",
                info="Use -1 for the middle slice.",
                precision=0,
                scale=1,
            )
            btn_load = gr.Button("Load scan", variant="primary", scale=1)

        scan_info = gr.Textbox(label="Scan info", interactive=False, lines=2, elem_classes=["ct-mono"])

        with gr.Row(equal_height=False):
            # -------- controls --------
            with gr.Column(scale=2, min_width=360, elem_classes=["ct-panel"]):
                with gr.Tabs():
                    with gr.Tab("Align"):
                        apply_log_align = gr.Checkbox(
                            value=True,
                            label="Use CT intensity conversion",
                            info=INFO["apply_log"],
                        )
                        pixel_shift = gr.Slider(
                            SHIFT_MIN,
                            SHIFT_MAX,
                            value=0.0,
                            step=0.05,
                            label="Fine alignment (pixels)",
                            info=INFO["shift"],
                        )
                        pixel_shift_num = gr.Number(
                            value=0.0, label="Exact value", precision=3, info=INFO["shift"]
                        )
                        with gr.Row(elem_classes=["ct-nudge"]):
                            btn_m05 = gr.Button("-0.5")
                            btn_m01 = gr.Button("-0.1")
                            btn_quick = gr.Button("Refresh view")
                            btn_p01 = gr.Button("+0.1")
                            btn_p05 = gr.Button("+0.5")
                        with gr.Row():
                            btn_auto = gr.Button("Auto-find best", variant="primary")
                            btn_align_check = gr.Button("Check top/mid/bottom")
                            btn_reset = gr.Button("Use Bruker log value")
                        with gr.Row():
                            base_center_out = gr.Number(0, label="Base rotation center", interactive=False)
                            effective_center_out = gr.Number(
                                0, label="Effective center (base + shift)", interactive=False
                            )
                        gr.Markdown(
                            f"<span title='{INFO['auto_tune']}'>Auto-find best</span> tries many "
                            f"shifts and keeps the sharpest. "
                            f"<span title='{INFO['multi_row']}'>Check top/mid/bottom</span> warns if "
                            f"rows disagree."
                        )

                    with gr.Tab("Rings"):
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            preset = _dropdown(
                                choices=_preset_choices(),
                                value="default",
                                label="Preset recipe",
                                info=INFO["preset"],
                            )
                            btn_preset = gr.Button("Apply preset")
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            recipe_name = gr.Textbox(
                                label="Save current as",
                                placeholder="my_sample",
                                info="Writes a YAML file under config/presets/",
                            )
                            btn_save = gr.Button("Save")
                        ring_enable = gr.Checkbox(value=defaults.ring_enable, label="Clean rings")
                        ring_method = _dropdown(
                            choices=ring_choices,
                            value=ring_label(defaults.ring_method),
                            label="Ring cleaner",
                            info=INFO["ring_method"],
                        )
                        snr = gr.Slider(
                            1.0, 10.0, value=defaults.snr, step=0.1,
                            label="Ring strength gate", info=INFO["snr"],
                        )
                        la_size = gr.Slider(
                            3, 151, value=defaults.la_size, step=2,
                            label="Large-ring width (pixels)", info=INFO["la_size"],
                        )
                        sm_size = gr.Slider(
                            3, 101, value=defaults.sm_size, step=2,
                            label="Fine-ring smoothing (pixels)", info=INFO["sm_size"],
                        )
                        drop_ratio = gr.Slider(
                            0.0, 0.5, value=defaults.drop_ratio, step=0.01,
                            label="Aggressive cleanup amount", info=INFO["drop_ratio"],
                        )
                        dim = gr.Radio(
                            choices=[1, 2], value=defaults.dim,
                            label="Stripe axis", info=INFO["dim"],
                        )
                        btn_ring_cmp = gr.Button("Try all cleaners (picks a winner)")

                    with gr.Tab("Run"):
                        speed = gr.Radio(
                            choices=speed_choices,
                            value=speed_label(defaults.recon_type),
                            label="Speed / quality",
                            info=INFO["speed"],
                        )
                        filter_name = _dropdown(
                            choices=filter_choices,
                            value=filter_label(defaults.filter_name),
                            label="Image filter",
                            info=INFO["filter"],
                        )
                        apply_log = gr.Checkbox(
                            value=defaults.apply_log,
                            label="Use CT intensity conversion",
                            info=INFO["apply_log"],
                        )
                        output_dir = gr.Textbox(
                            value="",
                            label="Output folder override (optional)",
                            info="Leave blank to use <scan>/algotom/…",
                        )
                        btn_preview = gr.Button("Preview this slice", variant="primary")
                        btn_full = gr.Button("Reconstruct full volume", variant="stop")
                        gr.Markdown(
                            "Preview is for tuning. Full volume writes a **new** folder under "
                            "`<scan>/algotom/recon_…` and never overwrites an old run."
                        )

            # -------- viewer --------
            with gr.Column(scale=4, min_width=520, elem_classes=["ct-panel", "ct-viewer-wrap"]):
                view_choice = gr.Radio(
                    choices=["Align check", "Before cleanup", "After cleanup", "What changed"],
                    value="Align check",
                    label="Show in viewer",
                    info=INFO["viewer"],
                )
                main_img = gr.Image(
                    label="Viewer",
                    type="numpy",
                    height=620,
                )
                compare_gallery = gr.Gallery(
                    label="Ring-cleaner comparison (when you run Try all cleaners)",
                    columns=5,
                    height=160,
                    visible=True,
                )
                gr.Markdown("**History** — click a thumbnail to reload those settings")
                history_gallery = gr.Gallery(
                    label="Saved looks",
                    columns=6,
                    height=180,
                    elem_classes=["ct-history"],
                )

        status = gr.Textbox(
            label="Log",
            lines=8,
            value=startup_banner(),
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
            speed,
            filter_name,
            apply_log,
            pixel_shift,
            preview_row,
            output_dir,
        ]

        # sync slider <-> number (clamped)
        def _sync_from_slider(v):
            return _clamp_shift(v)

        def _sync_from_num(v):
            return _clamp_shift(v)

        pixel_shift.release(_sync_from_slider, inputs=pixel_shift, outputs=pixel_shift_num)
        pixel_shift_num.change(_sync_from_num, inputs=pixel_shift_num, outputs=pixel_shift)

        view_choice.change(
            on_view_choice,
            inputs=[view_choice, align_img_state, before_state, after_state, diff_state],
            outputs=[main_img],
        )

        btn_load.click(
            on_load_and_cache,
            inputs=[scan_dir, preview_row, view_choice],
            outputs=[
                scan_info,
                preview_row,
                pixel_shift,
                align_cache,
                align_img_state,
                before_state,
                after_state,
                diff_state,
                base_center_out,
                effective_center_out,
                main_img,
                history_gallery,
                history_entries,
                status,
            ],
        ).then(_sync_from_slider, inputs=pixel_shift, outputs=pixel_shift_num)

        btn_quick.click(
            on_quick_align,
            inputs=[align_cache, pixel_shift, apply_log_align, view_choice, before_state, after_state, diff_state],
            outputs=[
                align_img_state,
                base_center_out,
                effective_center_out,
                pixel_shift_num,
                main_img,
                history_gallery,
                history_entries,
                status,
            ],
        ).then(lambda v: _clamp_shift(v), inputs=pixel_shift_num, outputs=pixel_shift)

        for btn, delta in ((btn_m05, -0.5), (btn_m01, -0.1), (btn_p01, 0.1), (btn_p05, 0.5)):
            btn.click(
                lambda c, s, a, vc, b, af, d, dd=delta: on_nudge(c, s, a, dd, vc, b, af, d),
                inputs=[
                    align_cache, pixel_shift, apply_log_align, view_choice,
                    before_state, after_state, diff_state,
                ],
                outputs=[
                    pixel_shift, pixel_shift_num, align_img_state,
                    base_center_out, effective_center_out, main_img,
                    history_gallery, history_entries, status,
                ],
            )

        btn_auto.click(
            on_auto_tune,
            inputs=[
                align_cache,
                apply_log_align,
                view_choice,
                before_state,
                after_state,
                diff_state,
                snr,
                la_size,
                sm_size,
                drop_ratio,
                dim,
            ],
            outputs=[
                pixel_shift,
                pixel_shift_num,
                align_img_state,
                base_center_out,
                effective_center_out,
                main_img,
                after_state,
                compare_gallery,
                history_gallery,
                history_entries,
                ring_enable,
                ring_method,
                status,
            ],
        ).then(lambda: "After cleanup", outputs=[view_choice])

        btn_align_check.click(
            on_align_check,
            inputs=[scan_dir, pixel_shift, apply_log_align, view_choice, before_state, after_state, diff_state],
            outputs=[
                align_img_state, pixel_shift, pixel_shift_num, main_img,
                history_gallery, history_entries, status,
            ],
        )

        def _reset_shift(cache, view_choice, before, after, diff):
            val = _clamp_shift(cache.log_postalignment) if cache is not None else 0.0
            img, base, eff, shift, view, gallery, entries, st = on_quick_align(
                cache, val, True, view_choice, before, after, diff
            )
            return val, val, img, base, eff, view, gallery, entries, st

        btn_reset.click(
            _reset_shift,
            inputs=[align_cache, view_choice, before_state, after_state, diff_state],
            outputs=[
                pixel_shift, pixel_shift_num, align_img_state,
                base_center_out, effective_center_out, main_img,
                history_gallery, history_entries, status,
            ],
        )

        btn_preset.click(on_apply_preset, inputs=[preset], outputs=[*controls, status])
        btn_save.click(on_save_recipe, inputs=[recipe_name, *controls], outputs=[status, preset])

        btn_preview.click(
            on_full_preview,
            inputs=[scan_dir, before_cache, before_key, view_choice, *controls],
            outputs=[
                before_state, after_state, diff_state,
                before_cache, before_key, main_img,
                history_gallery, history_entries, status,
            ],
        ).then(
            lambda: "After cleanup",
            outputs=[view_choice],
        )

        btn_ring_cmp.click(
            on_ring_compare,
            inputs=[
                scan_dir, view_choice, align_img_state, before_state, after_state, diff_state, *controls
            ],
            outputs=[
                compare_gallery, ring_enable, ring_method, main_img,
                history_gallery, history_entries, status,
            ],
        )

        btn_full.click(on_full, inputs=[scan_dir, *controls], outputs=[status])

        history_gallery.select(
            on_history_select,
            inputs=[history_entries],
            outputs=[main_img, *controls, status],
        )

    return demo


def _pids_listening_on_port(port: int) -> list[int]:
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
    print(f"Log file: {log_path()}")
    demo = build_app()
    theme = build_theme()
    launch_kwargs: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "inbrowser": True,
        "show_error": True,
    }
    sig = inspect.signature(demo.launch)
    # Gradio 6 wants theme/css on launch()
    if "theme" in sig.parameters and theme is not None:
        launch_kwargs["theme"] = theme
    if "css" in sig.parameters:
        launch_kwargs["css"] = GUI_CSS
    if "allowed_paths" in sig.parameters:
        launch_kwargs["allowed_paths"] = _gradio_allowed_paths()
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
