"""
Visual theme — full-width instrument UI for Gradio 6+.
"""
from __future__ import annotations

GUI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ct-ink: #142033;
  --ct-muted: #5a6a7c;
  --ct-line: #cfd8e3;
  --ct-panel: rgba(255,255,255,0.88);
  --ct-teal: #0f766e;
  --ct-teal-deep: #0b5f59;
  --ct-amber: #b45309;
}

html, body {
  margin: 0 !important;
  padding: 0 !important;
  background: #e7eef5 !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0.75rem 1.1rem 1.25rem !important;
  font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif !important;
  color: var(--ct-ink) !important;
}

.gradio-container, .main, .app {
  background:
    radial-gradient(1100px 420px at 0% -5%, rgba(15,118,110,0.12), transparent 55%),
    radial-gradient(900px 380px at 100% 0%, rgba(180,83,9,0.07), transparent 50%),
    linear-gradient(180deg, #f4f7fb 0%, #e7eef5 100%) !important;
}

.ct-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.4rem 0.2rem 0.85rem;
  border-bottom: 1px solid var(--ct-line);
  margin-bottom: 0.85rem;
}
.ct-brand {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.05;
}
.ct-brand span { color: var(--ct-teal); }
.ct-sub {
  color: var(--ct-muted);
  font-size: 0.92rem;
  margin-top: 0.15rem;
}

.ct-panel {
  background: var(--ct-panel) !important;
  border: 1px solid var(--ct-line) !important;
  border-radius: 16px !important;
  padding: 0.7rem 0.85rem 0.9rem !important;
  box-shadow: 0 12px 30px rgba(20,32,51,0.06) !important;
  backdrop-filter: blur(8px);
}

.ct-nudge {
  display: grid !important;
  grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
  gap: 0.45rem !important;
}
.ct-nudge > * { width: 100% !important; min-width: 0 !important; }

.ct-preset-row {
  display: grid !important;
  grid-template-columns: 1fr auto !important;
  gap: 0.45rem !important;
  align-items: end !important;
}

.ct-viewer-wrap .image-container,
.ct-viewer-wrap img {
  max-height: min(72vh, 820px) !important;
  object-fit: contain !important;
}

.ct-history .gallery,
.ct-history {
  min-height: 140px;
}

button.primary {
  background: linear-gradient(180deg, #0f766e, #0d9488) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
}
button.stop {
  background: linear-gradient(180deg, #c2410c, #b45309) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
}

.ct-mono textarea, .ct-mono input {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 0.8rem !important;
}

footer { display: none !important; }

@media (max-width: 1100px) {
  .ct-nudge { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
"""


def build_theme():
    import gradio as gr

    try:
        return gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.stone,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        )
    except Exception:
        return None
