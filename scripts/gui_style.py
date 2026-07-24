"""
Visual theme for the Bruker CT toolkit GUI.
Instrument-style light UI: cool slate + teal accent (not purple / not dark-default).
"""
from __future__ import annotations

GUI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ct-ink: #1a2332;
  --ct-muted: #5c6b7a;
  --ct-line: #d5dde6;
  --ct-panel: rgba(255, 255, 255, 0.78);
  --ct-teal: #0f766e;
  --ct-teal-deep: #115e59;
  --ct-amber: #b45309;
  --ct-wash: #e8eef4;
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif !important;
  color: var(--ct-ink) !important;
}

/* Soft instrument backdrop — cool wash, not flat white / not purple glow */
.gradio-container,
.main, .app {
  background:
    radial-gradient(1200px 500px at 8% -10%, rgba(15, 118, 110, 0.10), transparent 55%),
    radial-gradient(900px 420px at 100% 0%, rgba(180, 83, 9, 0.06), transparent 50%),
    linear-gradient(180deg, #f3f6f9 0%, var(--ct-wash) 100%) !important;
}

.ct-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.35rem 0.15rem 0.85rem;
  border-bottom: 1px solid var(--ct-line);
  margin-bottom: 0.85rem;
}
.ct-brand {
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--ct-ink);
}
.ct-brand span {
  color: var(--ct-teal);
}
.ct-tagline {
  font-size: 0.92rem;
  color: var(--ct-muted);
  max-width: 34rem;
  line-height: 1.35;
}
.ct-steps {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.78rem;
  color: var(--ct-teal-deep);
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.ct-panel {
  background: var(--ct-panel) !important;
  border: 1px solid var(--ct-line) !important;
  border-radius: 14px !important;
  padding: 0.65rem 0.75rem 0.85rem !important;
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 28px rgba(26, 35, 50, 0.05) !important;
}

.ct-viewer {
  min-height: 520px;
}

.ct-rail .tabs,
.ct-viewer .tabs {
  background: transparent !important;
}

button.primary, .primary {
  background: linear-gradient(180deg, #0f766e, #0d9488) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}
button.stop, .stop {
  background: linear-gradient(180deg, #c2410c, #b45309) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}

.ct-compact label, .ct-compact .label-wrap {
  font-size: 0.82rem !important;
}
.ct-mono textarea, .ct-mono input {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 0.8rem !important;
}

footer { display: none !important; }
"""


def build_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        text_size=gr.themes.sizes.text_md,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_text_color="#1a2332",
        block_title_text_weight="600",
        block_label_text_size="*text_sm",
        button_large_padding="8px 14px",
        button_small_padding="6px 10px",
    )
