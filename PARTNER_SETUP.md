# Partner setup checklist

For the lab PC (Windows, NVIDIA GPU, 128 GB RAM class machine).

## Before you start

1. Ask Nigel to invite your GitHub account to the private repo `NigelHorak/bruker-ct-recon-toolkit`.
2. Install [Git for Windows](https://git-scm.com/download/win) if needed.
3. Confirm the GPU driver works:

```powershell
nvidia-smi
```

You should see the GPU name and driver version. If this fails, update the NVIDIA driver first.

4. Have a Bruker scan folder ready: projection TIFFs + `.log` (see `examples/example_scan_layout.txt`).

## One-time install

Open **PowerShell** (normal user is fine; Admin only if Miniconda install fails):

```powershell
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1
```

First run can take 15–40 minutes (downloads Miniconda + Algotom + Astra).

When it finishes you should see `SMOKE TEST PASSED`.

## Reconstruct a scan

```powershell
cd path\to\bruker-ct-recon-toolkit
.\run_recon.ps1 -ScanDir "D:\Data\MySample_scan"
```

Optional:

```powershell
.\run_recon.ps1 -ScanDir "D:\Data\MySample_scan" -OutDir "D:\Data\MySample_algotom" -Config ".\config\default.yaml"
```

## If something breaks

| Symptom | What to try |
|---------|-------------|
| `gh` / clone auth failed | Accept the GitHub invite; run `gh auth login` or use HTTPS with a PAT |
| `nvidia-smi` missing | Install/update NVIDIA driver; reboot |
| Conda env create failed | Re-run `.\setup.ps1`; check internet / proxy |
| Astra CUDA error | Confirm GPU is free; reboot; re-run smoke test: `.\env\activate.ps1` then `python scripts\smoke_test.py` |
| Wrong center / soft rings | Edit `config\default.yaml` ring knobs; re-run |

## What not to do

- Do not point `-ScanDir` at an NRecon *reconstructed* folder — this pipeline wants **raw projections + log**.
- Do not commit huge TIFF stacks into GitHub.
