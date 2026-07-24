"""
Reconstruction viewer HTML + head assets (zoom/pan + scale bar + hover tips).
Pattern adapted from tooltip-interaction demos; scale bar uses real pixel size.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Optional

import numpy as np

VIEWER_HEAD = """
<style>
.ct-tip {
  position: relative;
  display: inline-flex;
  width: 1.15rem;
  height: 1.15rem;
  border-radius: 999px;
  background: #0f766e;
  color: #fff;
  font-size: 0.72rem;
  font-weight: 700;
  align-items: center;
  justify-content: center;
  cursor: help;
  margin-left: 0.35rem;
  vertical-align: middle;
  line-height: 1;
  border: none;
  flex-shrink: 0;
}
.ct-tip:focus { outline: 2px solid #0f766e; outline-offset: 2px; }
.ct-tip-bubble {
  display: none;
  position: absolute;
  left: 50%;
  bottom: calc(100% + 8px);
  transform: translateX(-50%);
  min-width: 14rem;
  max-width: 20rem;
  padding: 0.55rem 0.7rem;
  border-radius: 8px;
  background: #142033;
  color: #f4f7fb;
  font-size: 0.78rem;
  font-weight: 400;
  line-height: 1.35;
  text-align: left;
  z-index: 10050;
  box-shadow: 0 10px 28px rgba(20,32,51,0.35);
  pointer-events: none;
  white-space: normal;
}
.ct-tip-bubble::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #142033;
}
.ct-tip:hover .ct-tip-bubble,
.ct-tip:focus .ct-tip-bubble,
.ct-tip:focus-within .ct-tip-bubble { display: block; }

.ct-readout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin: 0.55rem 0 0.65rem;
}
.ct-readout-card {
  background: #eef3f8;
  border: 1px solid #cfd8e3;
  border-radius: 10px;
  padding: 0.55rem 0.65rem;
}
.ct-readout-card .k {
  display: block;
  font-size: 0.72rem;
  color: #5a6a7c;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.2rem;
}
.ct-readout-card .v {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 1.05rem;
  font-weight: 600;
  color: #142033;
}

.ct-viewer-root {
  width: 100%;
  user-select: none;
}
.ct-viewer-stage {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  max-height: min(72vh, 820px);
  overflow: hidden;
  border-radius: 14px;
  border: 1px solid #cfd8e3;
  background: #1a2332;
  touch-action: none;
}
.ct-viewer-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
  transform-origin: center center;
  will-change: transform;
}
.ct-viewer-empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9aa8b8;
  font-size: 0.95rem;
  text-align: center;
  padding: 1.5rem;
  background: #1a2332;
}
.ct-scalebar {
  position: absolute;
  left: 12px;
  bottom: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  pointer-events: none;
  z-index: 2;
}
.ct-scalebar-label {
  background: rgba(20,32,51,0.78);
  color: #f4f7fb;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
  font-variant-numeric: tabular-nums;
}
.ct-scalebar-line {
  height: 8px;
  border-left: 2px solid #f4f7fb;
  border-right: 2px solid #f4f7fb;
  border-bottom: 2px solid #f4f7fb;
  box-shadow: 0 1px 2px rgba(0,0,0,0.55);
}
.ct-viewer-hint {
  margin: 0.45rem 0 0;
  color: #5a6a7c;
  font-size: 0.8rem;
  text-align: center;
}
.ct-scan-row {
  display: flex !important;
  align-items: flex-end !important;
  gap: 0.55rem !important;
}
.ct-scan-row > * { margin-top: 0 !important; margin-bottom: 0 !important; }
.ct-scan-row button { height: 42px !important; }
</style>
<script>
(function () {
  const MIN = 1, MAX = 8, STEP = 0.35;

  function niceNumber(value) {
    if (!(value > 0) || !isFinite(value)) return 0;
    const exponent = Math.floor(Math.log10(value));
    const base = Math.pow(10, exponent);
    const fraction = value / base;
    let nice = 1;
    if (fraction >= 5) nice = 5;
    else if (fraction >= 2) nice = 2;
    return nice * base;
  }

  function formatUm(um) {
    if (um >= 1000) {
      const mm = um / 1000;
      return (Number.isInteger(mm) ? mm : mm.toFixed(1)) + " mm";
    }
    if (um >= 1) return (Number.isInteger(um) ? um : um.toFixed(1)) + " µm";
    return Math.round(um * 1000) + " nm";
  }

  function bindViewer(root) {
    if (!root || root.dataset.bound === "1") return;
    root.dataset.bound = "1";
    const stage = root.querySelector(".ct-viewer-stage");
    const img = root.querySelector(".ct-viewer-img");
    const bar = root.querySelector(".ct-scalebar");
    const barLabel = root.querySelector(".ct-scalebar-label");
    const barLine = root.querySelector(".ct-scalebar-line");
    if (!stage || !img) return;

    const nw = Math.max(1, parseFloat(root.dataset.nw || "1"));
    const nh = Math.max(1, parseFloat(root.dataset.nh || "1"));
    const pixUm = Math.max(0, parseFloat(root.dataset.pixUm || "0"));

    let scale = 1, x = 0, y = 0, panning = false;
    let last = null;

    function clamp() {
      const r = stage.getBoundingClientRect();
      const maxX = ((scale - 1) * r.width) / 2;
      const maxY = ((scale - 1) * r.height) / 2;
      x = Math.min(maxX, Math.max(-maxX, x));
      y = Math.min(maxY, Math.max(-maxY, y));
      if (scale <= MIN) { x = 0; y = 0; }
    }

    function apply() {
      clamp();
      img.style.transform = "translate(" + x + "px," + y + "px) scale(" + scale + ")";
      img.style.transition = panning ? "none" : "transform 0.15s ease-out";
      stage.style.cursor = scale > MIN ? (panning ? "grabbing" : "grab") : "default";
      updateScaleBar();
    }

    function updateScaleBar() {
      if (!bar || !barLabel || !barLine || !(pixUm > 0)) {
        if (bar) bar.style.display = "none";
        return;
      }
      const r = stage.getBoundingClientRect();
      const fit = Math.min(r.width / nw, r.height / nh); // CSS px per image pixel at zoom 1
      if (!(fit > 0)) { bar.style.display = "none"; return; }
      // Physical µm per CSS pixel at current zoom
      const umPerCss = pixUm / (fit * scale);
      const targetUm = umPerCss * (r.width * 0.25);
      const niceUm = niceNumber(targetUm);
      const barPx = niceUm / umPerCss;
      if (!(barPx > 0) || !(niceUm > 0)) { bar.style.display = "none"; return; }
      bar.style.display = "flex";
      barLabel.textContent = formatUm(niceUm);
      barLine.style.width = barPx + "px";
    }

    stage.addEventListener("wheel", function (e) {
      e.preventDefault();
      const dir = e.deltaY > 0 ? -1 : 1;
      scale = Math.min(MAX, Math.max(MIN, scale + dir * STEP));
      apply();
    }, { passive: false });

    stage.addEventListener("pointerdown", function (e) {
      if (scale <= MIN) return;
      stage.setPointerCapture(e.pointerId);
      panning = true;
      last = { x: e.clientX, y: e.clientY };
      apply();
    });
    stage.addEventListener("pointermove", function (e) {
      if (!panning || !last) return;
      x += e.clientX - last.x;
      y += e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      apply();
    });
    function endPan(e) {
      if (stage.hasPointerCapture && stage.hasPointerCapture(e.pointerId)) {
        stage.releasePointerCapture(e.pointerId);
      }
      panning = false;
      last = null;
      apply();
    }
    stage.addEventListener("pointerup", endPan);
    stage.addEventListener("pointerleave", endPan);
    stage.addEventListener("dblclick", function () {
      scale = scale >= MAX ? MIN : Math.min(MAX, scale + 1);
      apply();
    });

    if (window.ResizeObserver) {
      new ResizeObserver(function () { apply(); }).observe(stage);
    }
    if (img.complete) apply();
    else img.addEventListener("load", apply);
  }

  function scan() {
    document.querySelectorAll(".ct-viewer-root").forEach(bindViewer);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def tip_html(text: str) -> str:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<button type="button" class="ct-tip" aria-label="More information">'
        f"i<span class='ct-tip-bubble'>{safe}</span></button>"
    )


def centers_html(base: float, effective: float, shift: float) -> str:
    return (
        "<div class='ct-readout'>"
        f"<div class='ct-readout-card'><span class='k'>Base center</span>"
        f"<span class='v'>{float(base):.3f}</span></div>"
        f"<div class='ct-readout-card'><span class='k'>Effective center</span>"
        f"<span class='v'>{float(effective):.3f}</span></div>"
        f"<div class='ct-readout-card' style='grid-column:1/-1'><span class='k'>Center shift (what you are viewing)</span>"
        f"<span class='v'>{float(shift):+.3f} px</span></div>"
        "</div>"
    )


def empty_viewer_html() -> str:
    return (
        "<div class='ct-viewer-root'>"
        "<div class='ct-viewer-stage'>"
        "<div class='ct-viewer-empty'>Reconstruction will appear here after you load a scan "
        "or generate options.</div>"
        "</div>"
        "<p class='ct-viewer-hint'>Scroll to zoom · drag to pan · double-click to zoom in</p>"
        "</div>"
    )


def _to_png_b64(image: Any) -> tuple[str, int, int]:
    from PIL import Image

    x = np.asarray(image)
    if x.dtype != np.uint8:
        lo, hi = np.percentile(x.astype(np.float64), (1, 99))
        if hi <= lo:
            hi = lo + 1e-6
        x = (np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if x.ndim == 2:
        im = Image.fromarray(x)
    else:
        im = Image.fromarray(x[..., :3] if x.shape[-1] >= 3 else x)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, int(im.size[0]), int(im.size[1])


def viewer_html(image: Optional[Any], pixel_size_um: float = 0.0) -> str:
    if image is None:
        return empty_viewer_html()
    try:
        b64, nw, nh = _to_png_b64(image)
    except Exception:
        return empty_viewer_html()
    pix = float(pixel_size_um or 0.0)
    return (
        f"<div class='ct-viewer-root' data-nw='{nw}' data-nh='{nh}' data-pix-um='{pix}'>"
        f"<div class='ct-viewer-stage'>"
        f"<img class='ct-viewer-img' alt='Reconstruction' draggable='false' "
        f"src='data:image/png;base64,{b64}' />"
        f"<div class='ct-scalebar' style='display:none'>"
        f"<span class='ct-scalebar-label'>—</span>"
        f"<div class='ct-scalebar-line'></div>"
        f"</div>"
        f"</div>"
        f"<p class='ct-viewer-hint'>Scroll to zoom · drag to pan · double-click to zoom in"
        f"{' · scale bar from Bruker pixel size' if pix > 0 else ''}</p>"
        f"</div>"
    )
