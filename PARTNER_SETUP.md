# Partner setup checklist

For the lab PC (Windows, NVIDIA GPU, lots of RAM).

---

## Exact steps (copy/paste)

### 0) One-time: get access

1. Tell Nigel your **GitHub username**.
2. Accept the email invite to `NigelHorak/bruker-ct-recon-toolkit`.
3. Install [Git for Windows](https://git-scm.com/download/win) if you do not have it.
4. Confirm the GPU works:

```powershell
nvidia-smi
```

If that fails, update the NVIDIA driver and reboot before continuing.

### 1) One-time: clone + install (15–40 min)

Open **PowerShell**:

```powershell
cd $env:USERPROFILE\Downloads
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1
```

Wait until you see:

```text
SMOKE TEST PASSED
SETUP COMPLETE
```

If clone asks for login: sign in with the GitHub account that was invited (browser or Personal Access Token).

### 2) Point at a Bruker scan folder

The folder must contain **raw projections + `.log`**, not an NRecon reconstruction folder. Example:

```text
D:\Data\MySample_scan\
  MySample_scan.log
  MySample_scan00000000.tif
  MySample_scan00000001.tif
  ...
```

### 3) Fast preview first (tune rings)

```powershell
cd $env:USERPROFILE\Downloads\bruker-ct-recon-toolkit
.\run_recon.ps1 -ScanDir "D:\Data\MySample_scan" -Preview
```

Open the PNG it prints, usually:

```text
D:\Data\MySample_scan_algotom_preview\qc\mid_slice.png
```

Also saved: `qc\preview_slice.tif`

Optional — pick a different detector row:

```powershell
.\run_recon.ps1 -ScanDir "D:\Data\MySample_scan" -Preview -PreviewRow 800
```

### 4) If rings are still bad, edit knobs and re-preview

Edit `config\default.yaml`:

| Knob | Try |
|------|-----|
| `ring.snr` | lower (e.g. `2.0`) for stronger detection |
| `ring.la_size` | larger odd number (e.g. `71`) for big rings |
| `ring.sm_size` | larger odd number (e.g. `31`) for fine rings |

Then run step 3 again. Repeat until the mid slice looks good.

### 5) Full volume recon (slow)

```powershell
.\run_recon.ps1 -ScanDir "D:\Data\MySample_scan"
```

Output:

```text
D:\Data\MySample_scan_algotom_recon\
  slices\recon_00000.tif ...
  qc\mid_slice.png
  run_config.yaml
```

---

## If something breaks

| Symptom | Fix |
|---------|-----|
| Cannot clone private repo | Accept invite; `gh auth login` or use a GitHub PAT |
| `nvidia-smi` missing | Install/update NVIDIA driver; reboot |
| Setup fails mid-install | Re-run `.\setup.ps1` (safe to repeat) |
| Smoke test CUDA fail | Free the GPU; reboot; `.\env\activate.ps1` then `python scripts\smoke_test.py` |
| No `.log` / no projections | Wrong folder — need acquisition TIFF stack + log |
| Already have the repo | `cd ...\bruker-ct-recon-toolkit` then `git pull` |

## What not to do

- Do not point `-ScanDir` at NRecon **reconstructed** slices.
- Do not commit huge TIFF scans into GitHub.
