@echo off
REM Launch the Bruker CT Algotom GUI (Gradio) in the algotom-gpu conda env.
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

echo Starting Bruker CT Algotom Toolkit GUI...
echo A browser window should open at http://127.0.0.1:7860
echo (Any previous toolkit on that port is closed automatically.)
echo Keep this black window open while you use the toolkit.
echo Errors also appear here and in toolkit_gui.log next to this folder.
echo.

"%CONDA_EXE%" run -n algotom-gpu --no-capture-output python "%~dp0scripts\gui_app.py"
if errorlevel 1 (
  echo.
  echo GUI exited with an error. If Gradio is missing, re-run setup.ps1
  pause
)
endlocal
