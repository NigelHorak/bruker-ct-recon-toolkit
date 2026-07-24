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

### 2) One-time install

```powershell
cd $env:USERPROFILE\Downloads
git clone https://github.com/NigelHorak/bruker-ct-recon-toolkit.git
cd bruker-ct-recon-toolkit
.\setup.ps1
```

### 3) Every day

Double-click **`Start Toolkit.bat`** — browser opens at `http://127.0.0.1:7860`.

### 4) In the GUI
1. Paste scan folder → **Load scan**
2. **Align** — nudge / Auto-find best until the viewer looks sharp
3. **Rings** — preset or Try all cleaners
4. **Run** — Preview this slice → when happy, Reconstruct full volume
5. Click **History** thumbnails to reload an earlier look

Outputs are saved under the scan folder:

`...\MyScan\algotom\history\`  
`...\MyScan\algotom\preview\`  
`...\MyScan\algotom\recon_YYYYMMDD_HHMMSS\`

### Already installed?
```powershell
cd $env:USERPROFILE\Downloads\bruker-ct-recon-toolkit
git pull
```
Then restart **Start Toolkit.bat**.

## Do not
- Point at an NRecon *reconstructed* folder — use raw projections + log
- Close the black console while the GUI is open
