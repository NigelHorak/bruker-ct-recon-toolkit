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
1. Paste the scan folder → **Load scan**.
2. **Align** tab: nudge / Auto-tune / Multi-row check (image stays on the right).
3. **Rings** tab: preset or Compare all methods.
4. **Algorithm** tab: FBP vs FDK (Advanced for COR / output).
5. **Run** tab: Full Preview → check **Rings QC** viewer → Preflight → full recon.
6. Use viewer tabs: Align view | Rings QC | Method bake-off | History.

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
