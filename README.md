# Bruker CT Algotom Toolkit

Open-source reconstruction workflow for Bruker X4 / SkyScan-style micro-CT data, focused on **ring artifact removal** that NRecon often leaves behind.

Stack:
- [Algotom](https://github.com/algotom/algotom) — preprocessing, center of rotation, stripe/ring removal
- [Astra Toolbox](https://www.astra-toolbox.com/) — GPU `FBP_CUDA` reconstruction

Docs for ring methods: [Algotom ring removal](https://algotom.readthedocs.io/en/latest/toc/section4/section4_4.html) · [Sarepy background](https://sarepy.readthedocs.io/)

## Partner: install once, preview first, then full recon

```powershell
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1

# Fast: one mid-slice for ring tuning (do this first)
.\run_recon.ps1 -ScanDir "D:\path\to\bruker_scan" -Preview

# After rings look good: full volume
.\run_recon.ps1 -ScanDir "D:\path\to\bruker_scan"
```

Exact checklist: [PARTNER_SETUP.md](PARTNER_SETUP.md)

## What you need on disk

A Bruker scan folder with:
- Projection TIFF stack (`prefix00000000.tif`, `prefix00000001.tif`, …)
- Matching `prefix.log` (acquisition geometry)

See [examples/example_scan_layout.txt](examples/example_scan_layout.txt).

## Outputs

**Preview** (`-Preview`) → next to the scan:

```
<scan>_algotom_preview/
  qc/mid_slice.png
  qc/preview_slice.tif
  run_config.yaml
```

**Full recon** →:

```
<scan>_algotom_recon/
  slices/          # reconstructed TIFFs
  qc/mid_slice.png
  run_config.yaml
```

## Tuning rings

1. Run with `-Preview`
2. Edit [config/default.yaml](config/default.yaml)
3. Re-run `-Preview` until happy
4. Run full recon (no `-Preview`)

| Knob | Meaning |
|------|---------|
| `ring.snr` | Stripe detection sensitivity (try 2–4) |
| `ring.la_size` | Large-stripe filter size (odd, e.g. 51) |
| `ring.sm_size` | Small-stripe filter size (odd, e.g. 21) |

Heavier rings → lower `snr` and/or larger filter sizes. Too aggressive → soft detail loss.

Optional row pick: `.\run_recon.ps1 -ScanDir "..." -Preview -PreviewRow 800`

## Hardware notes

Designed for a Windows lab PC with NVIDIA GPU (~20 GB VRAM) and lots of RAM. Sinograms are processed in chunks (`recon.chunk_size` in config) so large detectors stay manageable.

## Invite a collaborator

GitHub repo → **Settings** → **Collaborators** → **Add people** → their GitHub username/email.

## License / credit

This toolkit is glue code for lab use. Cite Algotom / Astra / the Vo et al. ring-removal paper when publishing results that depend on them.
