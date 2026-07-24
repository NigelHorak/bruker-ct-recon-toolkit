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
- Center of rotation (auto / manual)
- Ring method + `snr`, `la_size`, `sm_size`, `drop_ratio`, `dim`
- FBP method / filter / log / iterations / chunk size
- Presets (mild / strong / off / fdk_conebeam) and save-your-own recipes
- Side-by-side BEFORE / AFTER mid-slice preview

## Scan folder layout
See [examples/example_scan_layout.txt](examples/example_scan_layout.txt).

## Outputs
- Preview → `<scan>_algotom_preview/qc/` (`before.png`, `after.png`, TIFFs, `run_config.yaml`)
- Full → `<scan>_algotom_recon/slices/` + QC

## Power-user CLI (optional)
`run_recon.ps1` still works if you want scripts; lab partners should use the GUI.

## Invite a collaborator
GitHub → **Settings** → **Collaborators** → add their account.

## License / credit
Glue code for lab use. Cite Algotom / Astra / Vo et al. ring-removal paper when publishing.
