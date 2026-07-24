"""
Gradio GUI for Bruker CT Algotom toolkit.
Partner workflow: double-click Start Toolkit.bat — no CLI needed.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recon_core import (  # noqa: E402
    FILTER_NAMES,
    RECON_METHODS,
    RECON_TYPES,
    RING_METHODS,
    Settings,
    load_settings,
    probe_scan_info,
    run_full,
    run_preview,
    save_yaml,
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
    preview_row: int,
    output_dir: str,
) -> Settings:
    center_mode_l = center_mode.lower()
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
        preview_row=int(preview_row) if preview_row >= 0 else None,
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
        -1 if s.preview_row is None else int(s.preview_row),
        s.output_dir or "",
    )


def on_load_folder(scan_dir: str):
    scan_dir = (scan_dir or "").strip().strip('"')
    if not scan_dir:
        return "Enter a scan folder path.", 0, 0, -1
    try:
        text, height, width, mid = probe_scan_info(scan_dir)
        return text, height, width, mid
    except Exception as exc:
        return f"ERROR: {exc}", 0, 0, -1


def on_apply_preset(name: str):
    s = _load_preset(name)
    return (*ui_settings_tuple(s), f"Loaded preset: {name}")


def _ui_args_to_settings(args) -> Settings:
    return _settings_from_ui(*args)


def on_preview(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    center_value = ctrl[14] if len(ctrl) > 14 else 0.0
    try:
        settings = _ui_args_to_settings(ctrl)
        result = run_preview(Path(scan_dir), settings, progress=progress)
        status = (
            f"{result.message}\n"
            f"Center used: {result.center:.3f} | row={result.row} | "
            f"type={settings.recon_type} | {result.n_projections} projections\n"
            + "\n".join(logs[-8:])
        )
        return result.display_raw, result.display_corr, status, float(result.center)
    except Exception as exc:
        return None, None, f"PREVIEW FAILED: {exc}\n" + "\n".join(logs[-12:]), center_value


def on_full(scan_dir: str, *ctrl):
    scan_dir = (scan_dir or "").strip().strip('"')
    logs: List[str] = []

    def progress(msg: str) -> None:
        logs.append(msg)

    try:
        settings = _ui_args_to_settings(ctrl)
        result = run_full(Path(scan_dir), settings, progress=progress)
        return f"{result.message}\n" + "\n".join(logs[-20:])
    except Exception as exc:
        return f"FULL RECON FAILED: {exc}\n" + "\n".join(logs[-20:])


def on_save_recipe(recipe_name: str, *ctrl):
    name = (recipe_name or "").strip()
    if not name:
        name = f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    settings = _ui_args_to_settings(ctrl)
    out = PRESET_DIR / f"{safe}.yaml"
    save_yaml(out, settings.to_config_dict())
    return f"Saved recipe: {out}", gr_preset_update()


def gr_preset_update():
    import gradio as gr

    return gr.update(choices=_preset_choices())


def build_app():
    import gradio as gr

    defaults = load_settings(DEFAULT_CFG)

    with gr.Blocks(title="Bruker CT Algotom Toolkit") as demo:
        gr.Markdown(
            "# Bruker CT Algotom Toolkit\n"
            "1) Paste the scan folder path  2) Adjust sliders  3) **Preview**  "
            "4) When happy, **Run full reconstruction**"
        )

        with gr.Row():
            scan_dir = gr.Textbox(
                label="Scan folder (contains TIFF projections + .log)",
                placeholder=r"D:\Data\MySample_scan",
                scale=4,
            )
            btn_load = gr.Button("Load folder info", scale=1)

        scan_info = gr.Textbox(label="Scan info", interactive=False)
        height_state = gr.Number(value=0, visible=False)
        width_state = gr.Number(value=0, visible=False)

        with gr.Row():
            preset = gr.Dropdown(choices=_preset_choices(), value="default", label="Preset / recipe")
            btn_preset = gr.Button("Apply preset")
            recipe_name = gr.Textbox(label="Save recipe as", placeholder="my_sample_rings")
            btn_save = gr.Button("Save recipe")

        with gr.Accordion("Ring removal", open=True):
            ring_enable = gr.Checkbox(value=defaults.ring_enable, label="Enable ring removal")
            ring_method = gr.Dropdown(choices=list(RING_METHODS), value=defaults.ring_method, label="Ring method")
            snr = gr.Slider(1.0, 10.0, value=defaults.snr, step=0.1, label="snr (higher = less sensitive)")
            la_size = gr.Slider(3, 151, value=defaults.la_size, step=2, label="la_size (large stripes, odd)")
            sm_size = gr.Slider(3, 101, value=defaults.sm_size, step=2, label="sm_size (small stripes, odd)")
            drop_ratio = gr.Slider(0.0, 0.5, value=defaults.drop_ratio, step=0.01, label="drop_ratio")
            dim = gr.Radio(choices=[1, 2], value=defaults.dim, label="dim")

        with gr.Accordion("Reconstruction", open=True):
            recon_type = gr.Radio(
                choices=list(RECON_TYPES),
                value=getattr(defaults, "recon_type", "FBP"),
                label="Algorithm family — FBP (fast per-slice) or FDK (true cone-beam, uses .log SOD/SDD)",
            )
            method = gr.Dropdown(
                choices=list(RECON_METHODS),
                value=defaults.method,
                label="FBP family method (ignored when FDK is selected)",
            )
            filter_name = gr.Dropdown(choices=list(FILTER_NAMES), value=defaults.filter_name, label="Filter")
            apply_log = gr.Checkbox(value=defaults.apply_log, label="Apply log (transmission → absorption)")
            num_iter = gr.Slider(1, 500, value=defaults.num_iter, step=1, label="Iterations (SIRT/SART/CGLS; FBP only)")
            chunk_size = gr.Slider(1, 128, value=defaults.chunk_size, step=1, label="Chunk size (FBP full recon)")
            center_mode = gr.Radio(choices=["auto", "manual"], value=defaults.center_mode, label="Center of rotation")
            center_value = gr.Number(value=0.0, label="Manual center (pixels)", precision=3)
            preview_row = gr.Number(value=-1, label="Preview row (-1 = middle)", precision=0)
            output_dir = gr.Textbox(value="", label="Output folder override (optional)")

        controls = [
            ring_enable, ring_method, snr, la_size, sm_size, drop_ratio, dim,
            recon_type, method, filter_name, apply_log, num_iter, chunk_size,
            center_mode, center_value, preview_row, output_dir,
        ]

        with gr.Row():
            btn_preview = gr.Button("Preview mid-slice", variant="primary")
            btn_full = gr.Button("Run full reconstruction", variant="stop")

        with gr.Row():
            img_before = gr.Image(label="BEFORE ring removal", type="numpy")
            img_after = gr.Image(label="AFTER ring removal", type="numpy")

        status = gr.Textbox(label="Status / log", lines=12)

        btn_load.click(on_load_folder, inputs=[scan_dir], outputs=[scan_info, height_state, width_state, preview_row])
        btn_preset.click(on_apply_preset, inputs=[preset], outputs=[*controls, status])
        btn_preview.click(on_preview, inputs=[scan_dir, *controls], outputs=[img_before, img_after, status, center_value])
        btn_full.click(on_full, inputs=[scan_dir, *controls], outputs=[status])
        btn_save.click(on_save_recipe, inputs=[recipe_name, *controls], outputs=[status, preset])

    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
