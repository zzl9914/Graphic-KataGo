@echo off
REM =====================================================================
REM  base/distill_cnn1d - KataGo distillation -> 1DCNN base model (基础培养)
REM
REM  Same KataGo supervised pretrain, for the 1DCNN ablation net (CPU, NumPy).
REM  Output: base/cnn1d/new.npz.
REM
REM  The 1DCNN convolves vertices in index order (no graph edges): it catches
REM  same-row locality on a grid but misses vertical/diagonal edges. Control
REM  line between the MLP and the GNN.
REM
REM  Usage:
REM    base\distill_cnn1d.bat                    (distill_data/m2_19x19.jsonl)
REM    base\distill_cnn1d.bat <path-to.jsonl>    (explicit data, root-relative)
REM
REM  Mid-run: --resume from the highest round*.npz. After new.npz, epoch
REM  snapshots are deleted; re-run keeps new.npz. No new and no rounds = epoch 1.
REM  --aug-from-epoch 4: epochs 1-3 identity numbering, then S_n.
REM =====================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY="
for /f "delims=" %%I in ('py -3.14 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%I"
if not defined PY (
  python -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,14) else 1)" 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  if exist "D:\Python\pythoncore-3.14-64\python.exe" set "PY=D:\Python\pythoncore-3.14-64\python.exe"
)
if not defined PY (
  echo ERROR: CPython 3.14 ^(non-free-threading^) required for gkt_native.cp314-*.pyd.
  pause
  exit /b 1
)

set "DATA=%~1"
if "%DATA%"=="" set "DATA=distill_data/m2_19x19.jsonl"

if not exist "%~dp0..\%DATA%" (
  echo [ERROR] distill data not found: %~dp0..\%DATA%
  echo.
  echo  Generate the JSONL:  distill_data\gen_m2_19x19.bat
  echo  then re-run: base\distill_cnn1d.bat ^<data.jsonl^>
  echo.
  pause
  exit /b 1
)

echo Data: %~dp0..\%DATA%
echo Interpreter: %PY%

if exist "%~dp0cnn1d\new.npz" (
  echo Product already at base\cnn1d\new.npz
) else if exist "%~dp0cnn1d\round1.npz" (
  echo Resume: latest round*.npz under base\cnn1d
) else (
  echo No checkpoint under base\cnn1d - starting a new distillation.
)

cd /d "%~dp0..\scr"
"%PY%" distill.py --net 1dcnn --data "../%DATA%"

echo.
echo Distillation finished. The base 1DCNN is at base\cnn1d\new.npz.
pause
