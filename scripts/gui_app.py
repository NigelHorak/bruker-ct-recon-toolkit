"""
Gradio GUI — NRecon-style: generate options in a range, user picks the best.
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
from history_store import list_history_entries  # noqa: E402
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
    candidates_to_gallery,
    sweep_alignment,
    sweep_beam_hardening,
    sweep_ring_recipes,
)

PRESET_DIR = ROOT / "config" / "presets"
DEFAULT_CFG = ROOT / "config" / "default.yaml"


def _tip(text: str) -> str:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<span class="ct-i" title="{safe}">i</span>'


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


def _gallery_and_entries(scan_dir: str):
    entries: List[Dict[str, Any]] = []
    try:
        if scan_dir:
            entries = list_history_entries(Path(scan_dir))
    except Exception as exc:
        log_exception("history", exc)
    return [(e["image"], e["caption"]) for e in entries], entries


def _cand_list(cands: List[Candidate]) -> List[Dict[str, Any]]:
    return [{"label": c.label, "image": c.image, "payload": c.payload} for c in cands]


def on_load(scan_dir: str, preview_row: float):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        return "", 0, 0.0, None, None, None, 0.0, 0.0, [], [], "Enter a scan folder path."

    path = Path(scan_dir)
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
        try:
            from PIL import Image

            arr = np.asarray(Image.open(path).convert("RGB"))
            return str(path), 0, 0.0, None, arr, arr, 0.0, 0.0, [], [], f"Loaded {path.name}"
        except Exception as exc:
            return f"ERROR: {exc}", 0, 0.0, None, None, None, 0.0, 0.0, [], [], log_exception("Load", exc)

    progress = ProgressLog(f"LOAD {scan_dir}")
    try:
        text, height, width, mid, post = probe_scan_info(scan_dir)
        row_in = int(preview_row or 0)
        row = mid if row_in < 0 else min(max(0, row_in), height - 1)
        cache = prepare_align_cache(Path(scan_dir), preview_row=row, progress=progress)
        shift = _clamp_shift(cache.log_postalignment or post or 0.0)
        img, msg, base, eff = quick_align_preview(cache, shift, apply_log=True, save_history=True)
        hist, entries = _gallery_and_entries(scan_dir)
        status = (
            f"Loaded. Detector {height}x{width}. Slice {row} (middle={mid}).\n"
            f"Started at Bruker log alignment {shift:+.3f}.\n{msg}\n{progress.text()}"
        )
        return text, row, shift, cache, img, img, base, eff, hist, entries, status
    except Exception as exc:
        return f"ERROR: {exc}", 0, 0.0, None, None, None, 0.0, 0.0, [], [], log_exception("Load", exc)


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


def on_align_sweep(cache, shift_from, shift_to, shift_step, apply_log):
    if cache is None:
        return [], [], None, "Load a scan first."
    progress = ProgressLog("Alignment sweep")
    try:
        cands = sweep_alignment(
            cache, float(shift_from), float(shift_to), float(shift_step), bool(apply_log), progress
        )
        return (
            candidates_to_gallery(cands),
            _cand_list(cands),
            cands[0].image if cands else None,
            f"{len(cands)} alignment options. Click one, then Use this alignment.\n{progress.text()}",
        )
    except Exception as exc:
        return [], [], None, log_exception("Alignment sweep", exc)


def on_pick_candidate(evt, candidates):
    try:
        if not candidates:
            return None, {}, "Generate options first."
        idx = int(getattr(evt, "index", 0) or 0)
        idx = max(0, min(idx, len(candidates) - 1))
        item = candidates[idx]
        return item["image"], item["payload"], f"Selected: {item['label']}"
    except Exception as exc:
        return None, {}, log_exception("Select", exc)


def on_use_alignment(payload, cache):
    if not payload or "pixel_shift" not in payload:
        return 0.0, 0.0, 0.0, None, "Select an alignment option first."
    shift = _clamp_shift(payload["pixel_shift"])
    if cache is None:
        return shift, 0.0, shift, None, f"Using shift {shift:+.3f}."
    img, msg, base, eff = quick_align_preview(cache, shift, apply_log=True, save_history=True)
    return shift, base, eff, img, f"Locked alignment {shift:+.3f}.\n{msg}"


def on_nudge(cache, pixel_shift, delta, apply_log):
    if cache is None:
        return _clamp_shift(pixel_shift), None, 0.0, 0.0, "Load a scan first."
    shift = _clamp_shift(float(pixel_shift or 0.0) + float(delta))
    log_line(f"NUDGE {delta:+.1f} -> {shift:+.3f}")
    try:
        img, msg, base, eff = quick_align_preview(cache, shift, apply_log=bool(apply_log), save_history=True)
        return shift, img, base, eff, msg
    except Exception as exc:
        return shift, None, 0.0, 0.0, log_exception("Nudge", exc)


def on_ring_sweep(cache, pixel_shift, snr, la_size, sm_size, drop_ratio, dim, apply_log):
    if cache is None:
        return [], [], None, "Load a scan first."
    progress = ProgressLog("Ring options")
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
        return (
            candidates_to_gallery(cands),
            _cand_list(cands),
            cands[0].image if cands else None,
            f"{len(cands)} ring options. Click one, then Use this ring setting.\n{progress.text()}",
        )
    except Exception as exc:
        return [], [], None, log_exception("Ring sweep", exc)


def on_use_ring(payload):
    if not payload or "ring_method" not in payload:
        return ring_label("remove_all_stripe"), 3.0, 51, 21, 0.1, 1, "Select a ring option first."
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


def on_bh_sweep(base_img, bh_from, bh_to, bh_step):
    if base_img is None:
        return [], [], None, "Load or preview a slice first."
    progress = ProgressLog("BH options")
    try:
        cands = sweep_beam_hardening(base_img, float(bh_from), float(bh_to), float(bh_step), progress)
        return (
            candidates_to_gallery(cands),
            _cand_list(cands),
            cands[0].image if cands else None,
            f"{len(cands)} BH options. Click one, then Use this BH.\n{progress.text()}",
        )
    except Exception as exc:
        return [], [], None, log_exception("BH sweep", exc)


def on_use_bh(payload, base_img):
    if not payload or "bh_strength" not in payload:
        return 0.0, base_img, "Select a BH option first."
    s = float(payload["bh_strength"])
    img = apply_simple_beam_hardening(base_img, s) if base_img is not None else None
    return s, img, f"Locked BH strength {s:.2f}."


def on_preview(scan_dir, before_cache, before_key, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    progress = ProgressLog("PREVIEW")
    if not scan_dir:
        return None, None, before_cache, before_key or "", [], [], "Enter a scan folder."
    try:
        settings = _settings_from_ui(*ctrl)
        result = run_preview(
            Path(scan_dir),
            settings,
            progress=progress,
            cached_before=before_cache,
            cached_before_key=before_key or "",
        )
        hist, entries = _gallery_and_entries(scan_dir)
        return (
            result.display_corr,
            result.display_corr,
            result.img_raw,
            result.before_key,
            hist,
            entries,
            f"{result.message}\n{progress.text()}",
        )
    except Exception as exc:
        return None, None, before_cache, before_key or "", [], [], log_exception("Preview", exc)


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


def on_history_select(evt, entries):
    try:
        if not entries:
            return None, *ui_tuple(load_settings(DEFAULT_CFG)), "No history."
        idx = int(getattr(evt, "index", 0) or 0)
        item = entries[max(0, min(idx, len(entries) - 1))]
        cfg = item.get("settings") or {}
        s = Settings.from_config_dict(cfg) if cfg else load_settings(DEFAULT_CFG)
        return item.get("image"), *ui_tuple(s), f"Restored:\n{item.get('caption', '')}"
    except Exception as exc:
        return None, *ui_tuple(load_settings(DEFAULT_CFG)), log_exception("History", exc)


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

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            f"""
            <div class="ct-header">
              <div>
                <div class="ct-brand">Bruker CT <span>Algotom</span></div>
                <div class="ct-sub">Generate a range → pick the best → lock it in {_tip('Same idea as NRecon: you choose, the tool does not guess forever.')}</div>
              </div>
            </div>
            """
        )

        align_cache = gr.State(None)
        before_cache = gr.State(None)
        before_key = gr.State("")
        active_cands = gr.State([])
        pending = gr.State({})
        history_entries = gr.State([])
        work_img = gr.State(None)

        pixel_shift = gr.Number(value=0.0, visible=False)
        bh_strength = gr.Number(value=0.0, visible=False)
        apply_log = gr.Checkbox(value=True, visible=False)

        with gr.Row(elem_classes=["ct-panel"]):
            scan_dir = gr.Textbox(label="Scan folder", placeholder=r"D:\Results\...\MyScan", scale=4, elem_classes=["ct-mono"])
            preview_row = gr.Number(value=0, label="Slice #", precision=0, scale=1)
            btn_mid = gr.Button("Middle (fastest)", scale=1)
            btn_load = gr.Button("Load scan", variant="primary", scale=1)

        scan_info = gr.Textbox(label="Scan info", interactive=False, lines=2, elem_classes=["ct-mono"])

        with gr.Row():
            with gr.Column(scale=2, min_width=340, elem_classes=["ct-panel"]):
                with gr.Tabs():
                    with gr.Tab("Align"):
                        gr.HTML(
                            f"<div class='ct-help'>From / To / Step → Generate → click best → Use this alignment. "
                            f"{_tip(INFO['shift'])}</div>"
                        )
                        with gr.Row():
                            shift_from = gr.Number(value=-5.0, label="From", precision=2)
                            shift_to = gr.Number(value=15.0, label="To", precision=2)
                            shift_step = gr.Number(value=5.0, label="Step", precision=2)
                        btn_align_sweep = gr.Button("Generate alignment options", variant="primary")
                        with gr.Row(elem_classes=["ct-nudge"]):
                            btn_m1 = gr.Button("-1")
                            btn_m05 = gr.Button("-0.5")
                            btn_p05 = gr.Button("+0.5")
                            btn_p1 = gr.Button("+1")
                        btn_use_align = gr.Button("Use this alignment", variant="secondary")
                        with gr.Row():
                            base_out = gr.Number(0, label="Base center", interactive=False)
                            eff_out = gr.Number(0, label="Effective center", interactive=False)

                    with gr.Tab("Rings"):
                        gr.HTML(
                            f"<div class='ct-help'>Generate recipes → click best → Use this ring setting. "
                            f"{_tip(INFO['ring_method'])}</div>"
                        )
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            preset = _dd(choices=_preset_choices(), value="default", label="Preset")
                            btn_preset = gr.Button("Apply")
                        ring_method = _dd(
                            choices=ring_choices,
                            value=ring_label(defaults.ring_method),
                            label="Current ring recipe",
                        )
                        with gr.Accordion("Fine knobs (optional)", open=False):
                            snr = gr.Slider(1.0, 10.0, value=defaults.snr, step=0.1, label="Strength gate")
                            la_size = gr.Slider(3, 151, value=defaults.la_size, step=2, label="Large-ring width")
                            sm_size = gr.Slider(3, 101, value=defaults.sm_size, step=2, label="Fine smoothing")
                            drop_ratio = gr.Slider(0.0, 0.5, value=defaults.drop_ratio, step=0.01, label="Cleanup amount")
                            dim = gr.Radio(choices=[1, 2], value=defaults.dim, label="Stripe axis")
                        btn_ring_sweep = gr.Button("Generate ring options", variant="primary")
                        btn_use_ring = gr.Button("Use this ring setting", variant="secondary")
                        with gr.Row(elem_classes=["ct-preset-row"]):
                            recipe_name = gr.Textbox(label="Save recipe as", placeholder="my_sample")
                            btn_save = gr.Button("Save")

                    with gr.Tab("Beam hardening"):
                        gr.HTML("<div class='ct-help'>Simple post curve. Generate → pick → Use this BH.</div>")
                        with gr.Row():
                            bh_from = gr.Number(value=0.0, label="From", precision=2)
                            bh_to = gr.Number(value=2.0, label="To", precision=2)
                            bh_step = gr.Number(value=0.5, label="Step", precision=2)
                        btn_bh_sweep = gr.Button("Generate BH options", variant="primary")
                        btn_use_bh = gr.Button("Use this BH", variant="secondary")

                    with gr.Tab("Run"):
                        speed = gr.Radio(choices=speed_choices, value=speed_label(defaults.recon_type), label="Speed")
                        output_dir = gr.Textbox(value="", label="Output override (optional)")
                        btn_preview = gr.Button("Preview this slice", variant="primary")
                        btn_full = gr.Button("Reconstruct full volume", variant="stop")

            with gr.Column(scale=4, min_width=520, elem_classes=["ct-panel", "ct-viewer-wrap"]):
                main_img = gr.Image(label="Viewer", type="numpy", height=560)
                options_gallery = gr.Gallery(label="Options — click one, then Use…", columns=5, height=200)
                history_gallery = gr.Gallery(label="History — click to restore", columns=6, height=160)

        status = gr.Textbox(label="Log", lines=7, value=startup_banner(), elem_classes=["ct-mono", "ct-panel"])

        recon_controls = [
            ring_method, snr, la_size, sm_size, drop_ratio, dim,
            speed, apply_log, pixel_shift, preview_row, output_dir,
        ]
        all_controls = [*recon_controls, bh_strength]

        btn_mid.click(on_middle_slice, inputs=[scan_dir, align_cache], outputs=[preview_row, status])
        btn_load.click(
            on_load,
            inputs=[scan_dir, preview_row],
            outputs=[
                scan_info, preview_row, pixel_shift, align_cache, main_img, work_img,
                base_out, eff_out, history_gallery, history_entries, status,
            ],
        )

        btn_align_sweep.click(
            on_align_sweep,
            inputs=[align_cache, shift_from, shift_to, shift_step, apply_log],
            outputs=[options_gallery, active_cands, main_img, status],
        )
        btn_ring_sweep.click(
            on_ring_sweep,
            inputs=[align_cache, pixel_shift, snr, la_size, sm_size, drop_ratio, dim, apply_log],
            outputs=[options_gallery, active_cands, main_img, status],
        )
        btn_bh_sweep.click(
            on_bh_sweep,
            inputs=[work_img, bh_from, bh_to, bh_step],
            outputs=[options_gallery, active_cands, main_img, status],
        )

        options_gallery.select(on_pick_candidate, inputs=[active_cands], outputs=[main_img, pending, status])

        btn_use_align.click(
            on_use_alignment,
            inputs=[pending, align_cache],
            outputs=[pixel_shift, base_out, eff_out, main_img, status],
        ).then(lambda i: i, inputs=[main_img], outputs=[work_img])

        btn_m1.click(
            lambda c, s, a: on_nudge(c, s, -1.0, a),
            inputs=[align_cache, pixel_shift, apply_log],
            outputs=[pixel_shift, main_img, base_out, eff_out, status],
        ).then(lambda i: i, inputs=[main_img], outputs=[work_img])
        btn_m05.click(
            lambda c, s, a: on_nudge(c, s, -0.5, a),
            inputs=[align_cache, pixel_shift, apply_log],
            outputs=[pixel_shift, main_img, base_out, eff_out, status],
        ).then(lambda i: i, inputs=[main_img], outputs=[work_img])
        btn_p05.click(
            lambda c, s, a: on_nudge(c, s, 0.5, a),
            inputs=[align_cache, pixel_shift, apply_log],
            outputs=[pixel_shift, main_img, base_out, eff_out, status],
        ).then(lambda i: i, inputs=[main_img], outputs=[work_img])
        btn_p1.click(
            lambda c, s, a: on_nudge(c, s, 1.0, a),
            inputs=[align_cache, pixel_shift, apply_log],
            outputs=[pixel_shift, main_img, base_out, eff_out, status],
        ).then(lambda i: i, inputs=[main_img], outputs=[work_img])

        btn_use_ring.click(
            on_use_ring,
            inputs=[pending],
            outputs=[ring_method, snr, la_size, sm_size, drop_ratio, dim, status],
        )
        btn_use_bh.click(on_use_bh, inputs=[pending, work_img], outputs=[bh_strength, main_img, status])

        btn_preset.click(on_apply_preset, inputs=[preset], outputs=[*all_controls, status])
        btn_save.click(on_save_recipe, inputs=[recipe_name, *all_controls], outputs=[status, preset])

        btn_preview.click(
            on_preview,
            inputs=[scan_dir, before_cache, before_key, *recon_controls],
            outputs=[
                main_img,
                work_img,
                before_cache,
                before_key,
                history_gallery,
                history_entries,
                status,
            ],
        )
        btn_full.click(on_full, inputs=[scan_dir, *recon_controls], outputs=[status])

        history_gallery.select(
            on_history_select,
            inputs=[history_entries],
            outputs=[main_img, *all_controls, status],
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
        if local.endswith(needle) or local.endswith(f"]{needle}"):
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                pass
    return sorted(pids)


def _reclaim_port(port: int) -> None:
    import subprocess
    import time

    pids = [p for p in _pids_listening_on_port(port) if p != os.getpid()]
    if not pids:
        return
    print(f"Port {port} busy — closing {pids}")
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True, check=False)
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
        "inbrowser": False,
        "show_error": True,
    }
    sig = inspect.signature(demo.launch)
    if "theme" in sig.parameters and theme is not None:
        launch_kwargs["theme"] = theme
    if "css" in sig.parameters:
        launch_kwargs["css"] = GUI_CSS
    if "allowed_paths" in sig.parameters:
        launch_kwargs["allowed_paths"] = _gradio_allowed_paths()
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
