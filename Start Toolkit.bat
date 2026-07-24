@echo off
REM Launch toolkit hidden, open browser, then close this window.
setlocal
cd /d "%~dp0"

set "CONDA_EXE="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%ProgramData%\miniconda3\Scripts\conda.exe"

if "%CONDA_EXE%"=="" (
  echo ERROR: conda not found. Run setup.ps1 first.
  pause
  exit /b 1
)

REM Hidden console; logs still go to toolkit_gui.log from Python.
powershell -NoProfile -WindowStyle Hidden -Command ^
  "Start-Process -WindowStyle Hidden -FilePath '%CONDA_EXE%' -ArgumentList @('run','-n','algotom-gpu','--no-capture-output','python','%~dp0scripts\gui_app.py') -WorkingDirectory '%~dp0'"

timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:7860
endlocal
exit
