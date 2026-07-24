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
- **Align** — fine alignment (Bruker postalignment), auto-find, top/mid/bottom check
- **Rings** — plain-language ring cleaner + presets (YAML in `config/presets`)
- **Run** — Fast vs Careful 3D, preview, full volume
- One large viewer + history filmstrip (click a thumb to restore settings)

## Outputs (inside the scan folder)
- `<scan>/algotom/history/`
- `<scan>/algotom/preview/`
- `<scan>/algotom/recon_YYYYMMDD_HHMMSS/`


## Power-user CLI (optional)
`run_recon.ps1` still works if you want scripts; lab partners should use the GUI.

## Invite a collaborator
GitHub → **Settings** → **Collaborators** → add their account.

## License / credit
Glue code for lab use. Cite Algotom / Astra / Vo et al. ring-removal paper when publishing.
