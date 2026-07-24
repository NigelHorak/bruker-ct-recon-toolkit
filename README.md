# Bruker CT Algotom Toolkit

GUI-first reconstruction for Bruker X4 / SkyScan-style micro-CT, focused on **ring artifact removal**.

Stack: [Algotom](https://github.com/algotom/algotom) + [Astra Toolbox](https://www.astra-toolbox.com/) + Gradio UI.

Ring docs: [Algotom](https://algotom.readthedocs.io/en/latest/toc/section4/section4_4.html) · [Sarepy](https://sarepy.readthedocs.io/)

## Partner workflow (recommended)

```powershell
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1
```

Then **double-click `Start Toolkit.bat`** and use the browser GUI (sliders, preview, full recon).

Full checklist: [PARTNER_SETUP.md](PARTNER_SETUP.md)

## What the GUI controls
- **FBP vs FDK** algorithm family (FDK uses cone-beam geometry from the `.log`)
- Scan folder + preview row
- Center of rotation (auto / manual) + **pixel shift** (NRecon postalignment)
- **Multi-row align check** (confidence when mid-row alone is unsure)
- Ring method + `snr`, `la_size`, `sm_size`, `drop_ratio`, `dim`
- **Compare ring methods** bake-off with ring QC scores
- Full Preview: matched BEFORE/AFTER + **difference image** + ring-reduction %
- **Preflight** (log, projection count, RAM estimate, GPU, FDK geometry)
- FBP method / filter / log / iterations / chunk size
- Presets (mild / strong / off / fdk_conebeam) and save-your-own recipes
- History gallery under `<scan>_algotom_history/`

## Outputs
- Preview → `<scan>_algotom_preview/qc/` (`before.png`, `after.png`, `diff.png`, TIFFs, `run_config.yaml`)
- Full → **timestamped** `<scan>_algotom_recon_YYYYMMDD_HHMMSS/` (never overwrites a prior full run)
- History → `<scan>_algotom_history/`

## Power-user CLI (optional)
`run_recon.ps1` still works if you want scripts; lab partners should use the GUI.

## Invite a collaborator
GitHub → **Settings** → **Collaborators** → add their account.

## License / credit
Glue code for lab use. Cite Algotom / Astra / Vo et al. ring-removal paper when publishing.
