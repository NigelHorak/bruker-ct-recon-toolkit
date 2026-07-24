# Bruker CT Algotom Toolkit — Project Handoff

**Repo:** https://github.com/NigelHorak/bruker-ct-recon-toolkit (public)  
**Owner:** Nigel Horak (`NigelHorak`, nhhorak@gmail.com)  
**Local (Nigel):** `C:\Users\NIGEL\Downloads\CT\bruker-ct-recon-toolkit`  
**Typical lab clone:** `C:\Users\010\Downloads\bruker-ct-recon-toolkit`  
**Instrument context:** Bruker X4 / Poseidon micro-CT; partner dislikes NRecon ring artifacts  
**Handoff written:** 2026-07-23 (evening, US Mountain / UTC−6)  
**Conversation arc:** ~2026-07-23 afternoon → late evening (same calendar day)

---

## 1. Goal (what we set out to build)

1. A **public GitHub** home for lab recon docs/code  
2. An open-source reconstruction path with **strong ring post-processing** (Algotom + Astra)  
3. **One-shot Windows setup** for a lab PC (~128 GB RAM, ~20 GB NVIDIA / RTX 4000)  
4. A **GUI** (not CLI day-to-day) with sliders  
5. Fast **pixel / postalignment** tuning (~30 s target)  
6. Efficient **before/after** previews and **saved history** for comparison  

Stack chosen: **Algotom** (esp. `remove_all_stripe`) + **Astra** (`FBP_CUDA` / `FDK_CUDA`) + **Gradio** via `Start Toolkit.bat`, conda env `algotom-gpu` (Python 3.11).

Validated early against Nigel’s Poseidon log `BCO2-TP-01.log` (721 projs, 2400×2800, 0.5°/360°, SOD/SDD present, Postalignment=0.50, flat field OFF).

---

## 2. Timeline of work (same day, chronological)

| When (local Jul 23, 2026) | What happened |
|---|---|
| Earlier session(s) | Toolkit scaffold: `setup.ps1`, parse log, `recon_core`, Gradio GUI, presets, partner docs, GitHub public repo |
| Evening | BEFORE-cache reuse when only rings change; history gallery under scan |
| ~21:30 | Port 7860 busy → reclaim port / kill previous toolkit |
| ~21:39 | Feature pack: ring QC score, difference image, preflight, multi-row align, ring bake-off, timestamped full recons |
| ~21:50–22:00 | First UI “workstation” overhaul (teal theme, tabs) — still too jargony / cramped |
| ~22:08–22:20 | Logging overhaul (`toolkit_gui.log`, Activity log); Gradio blocked `D:\Results\…` history paths → in-memory gallery + `allowed_paths` |
| ~22:32–23:00 | Partner/Nigel feedback: nudge layout, ±5 crash, jargon, full-width, single viewer, plain names |
| ~23:00 | Full UI rebuild: Align / Rings / Run, Fast vs Careful 3D, `scan/algotom/` outputs, history restore |
| ~23:06+ | Handoff doc; Auto-find then short ring bake-off |

Exact commit SHAs live in `git log` on `master` (e.g. history cache, port reclaim, QC features, Gradio path fix, UI rebuild).

---

## 3. What exists today (capabilities)

### Install / launch
- `setup.ps1` — Miniconda, conda ToS, algotom/astra/gradio, smoke test  
- `Start Toolkit.bat` — launches Gradio on `127.0.0.1:7860`, reclaims port if busy  
- Partner docs: `PARTNER_SETUP.md`, `README.md`  

### Reconstruction (`scripts/recon_core.py`)
- Parse Bruker/SkyScan `.log` (`parse_bruker_log.py`)  
- **Fast path:** FBP_CUDA per detector row (preview / full)  
- **Careful 3D:** FDK_CUDA from SOD/SDD + pixel size  
- Ring methods via Algotom: `remove_all_stripe`, sorting, large, dead, or off  
- Pixel shift = NRecon-style postalignment added to auto COR  
- Align cache: load one sinogram row once, then fast FBP nudges  
- Preview reuses BEFORE recon when only ring knobs change  
- Full recon writes a **new** timestamped folder (no overwrite)  

### Outputs (current layout)
Inside the **scan folder**:
```
<scan>/algotom/history/          # PNG + settings.yaml per action
<scan>/algotom/preview/          # mid-slice QC
<scan>/algotom/recon_YYYYMMDD_HHMMSS/   # full volume slices + qc
```

### GUI (`scripts/gui_app.py` + `gui_style.py` + `gui_labels.py`)
- Full-width layout; controls left, large viewer right  
- Tabs: **Align** | **Rings** | **Run**  
- Viewer modes: Align check / Before / After / What changed  
- History filmstrip: click restores settings from that run  
- Plain-language labels + Gradio `info=` tooltips  
- Log box at bottom; also `toolkit_gui.log`  
- Optional email/ntfy alerts via `config/notify.yaml` (from `notify.example.yaml`)  

### QC / lab helpers
- `qc_metrics.py` — ring score + sharpness + difference image  
- `lab_tools.py` — multi-row align check, ring bake-off, preflight (still in code; not a main button)  

---

## 4. How “Auto-find best” decides “best” (important)

**Not built into Algotom.** Algotom provides reconstruction + ring filters.  
**We** score each candidate slice with a **Laplacian variance** sharpness metric (`_sharpness_score` in `recon_core.py` / `sharpness_score` in `qc_metrics.py`):

1. Walk pixel shifts around the current/log value (default search ±5 px, step 0.25)  
2. Reconstruct mid-row FBP (no rings) for each shift  
3. Pick the shift with **highest sharpness**  

Limitations: mid-row only; sharpness ≠ “ anatomically correct”; can prefer noisy edges. Multi-row check exists separately for agreement.

### Ring “winner” (Try all / post–Auto-find bake-off)
Also **our** metric, not Algotom’s:

1. Reconstruct with each ring method  
2. Compute `ring_score` (polar angular variance of high-pass residual)  
3. Prefer **lowest ring score**, break ties with **higher sharpness**  

As of this handoff, **Auto-find best** then automatically runs a **short bake-off** of 4 approaches (`none`, `remove_all_stripe`, sorting, large) and applies the winner to the Rings controls.

---

## 5. Problems we hit (and fixes)

| Problem | Fix |
|---|---|
| PowerShell Unicode / em-dashes in `setup.ps1` | ASCII-safe scripts |
| Conda ToS / PATH with spaces | Accept ToS; call `conda.exe` directly |
| Smoke test CPU-only exit codes | Allow finish on AMD; GPU still needed for CUDA recon |
| Partner Git PATH | Document full `git.exe` path / reboot |
| Port 7860 in use | Kill listener on 7860 before launch |
| Full Preview doing two expensive recons | Cache BEFORE; skip when key matches |
| Gradio cannot serve `D:\Results\…_algotom_history` | Return numpy images; `allowed_paths` for drives |
| Nudge past ±5 crashed Gradio slider | Range ±20 + clamp |
| Gradio 6: theme/css on Blocks | Pass theme/css to `launch()` |
| Jargon UI / wrapped nudge buttons | Rebuild + CSS grid + plain labels |
| Errors invisible in Activity log | Unified log + traceback + file |
| Accordion-hidden controls sometimes flaky | Keep Run/Algorithm controls always present |

Open / known gaps:
- Flat field OFF on real scan — no flats/darks pipeline yet  
- Postalignment sign may need empiric check vs NRecon  
- FDK preview loads full stack (RAM/GPU heavy)  
- True drag-pan/zoom viewer still limited (browser zoom / Gradio image)  
- Sharpness/ring scores are heuristics — human eye still wins  

---

## 6. Key files

| Path | Role |
|---|---|
| `setup.ps1` | One-shot Windows install |
| `Start Toolkit.bat` | Daily launcher |
| `scripts/gui_app.py` | Gradio UI + wiring |
| `scripts/gui_labels.py` | Plain names + help strings |
| `scripts/gui_style.py` | Theme CSS |
| `scripts/gui_log.py` | Console + `toolkit_gui.log` |
| `scripts/notify.py` | Optional email / ntfy |
| `scripts/recon_core.py` | Load / ring / FBP / FDK / preview / full / align |
| `scripts/lab_tools.py` | Preflight, multi-row, ring compare |
| `scripts/qc_metrics.py` | Ring/sharpness/diff QC |
| `scripts/history_store.py` | History under `algotom/history` |
| `scripts/parse_bruker_log.py` | `.log` parser |
| `config/default.yaml` + `config/presets/*.yaml` | Recipes |
| `config/notify.example.yaml` | Copy → `notify.yaml` for alerts |
| `PARTNER_SETUP.md` | Lab user steps |

---

## 7. Partner day-to-day

```powershell
cd $env:USERPROFILE\Downloads\bruker-ct-recon-toolkit
git pull
```
Double-click `Start Toolkit.bat`.

Workflow: Load scan → Align (or Auto-find best) → tweak Rings → Preview → Reconstruct full volume.  
Click history thumbnails to restore a prior look.

Optional alerts (Nigel’s machine / lab):
```powershell
copy config\notify.example.yaml config\notify.yaml
notepad config\notify.yaml
```
(Gmail app password; do not commit `notify.yaml`.)

---

## 8. Future plans (suggested)

**Near-term**
1. Real pan/zoom viewer (or Plotly/OpenSeadragon-style)  
2. Flat/dark correction when log says Flat Field OFF  
3. Batch queue: folder list → same recipe → report  
4. Empiric Postalignment sign check vs NRecon on one gold scan  
5. Memory-friendlier FDK (chunked / not full-stack in RAM)  

**Medium-term**
6. Export bundle (TIFF + geometry + settings + QC JSON/ZIP)  
7. Linked before/after swipe + shared window/level  
8. Confidence UI when multi-row align disagrees  

**Process**
9. Keep `HANDOFF.md` updated when behavior changes  
10. Prefer small PRs if the repo grows collaborators  

---

## 9. Design decisions to preserve

- Partner must not need CLI for daily work  
- Never silently overwrite a previous full recon  
- Prefer plain language in the UI; keep Algotom names in YAML/logs  
- Lab PC is NVIDIA — ship CUDA path as default; don’t clutter with CPU/AMD algo menus  
- History + settings restore beats “remember the slider values by hand”  

---

## 10. Contacts / accounts

- GitHub: NigelHorak / bruker-ct-recon-toolkit  
- Email for alerts example: nhhorak@gmail.com  
- Lab Windows user often `010`  

---

## 11. UX redesign (2026-07-23 late evening) — NRecon-style pickers

Partner feedback drove a second UX pass:

- **Removed** reconstruction image-filter menu (use post tools instead; recon uses fixed `hann`)
- **Removed** duplicate “exact value” + “fine alignment” pair; shift is one hidden working value
- **Removed** “Clean rings” checkbox (recipe list includes Off)
- **Slice #** shows a real index; **Middle (fastest)** sets mid-slice
- **Alignment / Rings / BH**: user sets **From / To / Step** (or fixed recipes for rings), **Generate options**, click the best thumbnail, **Use this…** to lock
- Coarse nudges only: **±1 / ±0.5** (wired with explicit lambdas so they actually fire)
- Hover **(i)** tips instead of long on-screen Gradio `info=` text
- **Start Toolkit.bat** launches Python **hidden**, opens the browser, then **exits** (no sticky black console). Logs still in `toolkit_gui.log`.

### How to use (intended)
1. Load scan → set slice (or Middle)
2. Align: e.g. From −5, To 15, Step 5 → Generate → click best → Use this alignment
3. Rings: Generate ring options → click best → Use this ring setting
4. Optional BH strength sweep on the current view
5. Run → Preview → Full volume

---

*End of handoff. For runtime truth, prefer this file + `git log` + `toolkit_gui.log` on a failing machine.*
