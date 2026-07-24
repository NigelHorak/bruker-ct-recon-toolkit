# Partner setup — GUI only after install

For the lab PC (Windows + NVIDIA GPU).

## Exact steps

### 1) One-time access
1. Tell Nigel your **GitHub username**.
2. Accept the invite to `NigelHorak/bruker-ct-recon-toolkit`.
3. Install [Git for Windows](https://git-scm.com/download/win) if needed.
4. Check the GPU:

```powershell
nvidia-smi
```

### 2) One-time install (only commands you need)

Open **PowerShell**:

```powershell
cd $env:USERPROFILE\Downloads
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1
```

Wait for `SMOKE TEST PASSED` and `SETUP COMPLETE` (15–40 minutes the first time).

### 3) Every day — double-click only

In File Explorer open:

`Downloads\bruker-ct-recon-toolkit\`

Double-click **`Start Toolkit.bat`**

A browser opens at `http://127.0.0.1:7860`. Keep the black window open while you work.

### 4) In the GUI
1. Paste the Bruker scan folder path (TIFF projections + `.log`).
2. Click **Load folder info**.
3. Pick a preset if you want (mild / strong / rings off), or move the sliders.
4. Click **Preview mid-slice**.
   - Use **FBP** for fast ring tuning.
   - Use **FDK** when you want true cone-beam geometry (needs SOD/SDD in the `.log`; slower preview).
5. Compare **BEFORE** vs **AFTER** images.
6. Tweak sliders and Preview again until rings look good.
7. Optional: type a name → **Save recipe**.
8. Click **Run full reconstruction**.

Outputs appear next to the scan as `*_algotom_preview` or `*_algotom_recon`.

### Already installed earlier?
```powershell
cd $env:USERPROFILE\Downloads\bruker-ct-recon-toolkit
git pull
.\setup.ps1
```
Then use **Start Toolkit.bat** again.

## Do not
- Point the GUI at an NRecon *reconstructed* folder — use raw projections + log.
- Close the black console window while the GUI is open.
