@echo off
REM =====================================================================
REM  base/distill_cnn2d - KataGo distillation -> 2DCNN base model (基础培养)
REM
REM  Same KataGo supervised pretrain, for the 2DCNN net (GPU, torch). Output:
REM  base/cnn2d/new.pt. The 2DCNN is the "traditional" grid baseline with
REM  toroidal circular padding — the classical-convolution control line to the
REM  GNN's message-passing route (needs a rectangular .grid graph).
REM
REM  Usage:
REM    base\distill_cnn2d.bat                    (distill_data/m2_19x19.jsonl)
REM    base\distill_cnn2d.bat <path-to.jsonl>    (explicit data, root-relative)
REM
REM  Re-run the same bat to resume: --resume loads the highest round*.pt
REM  in base/cnn2d/ and continues at epoch N+1. No checkpoint = epoch 1.
REM  --aug-from-epoch 4: epochs 1-3 identity board, then D4/Klein/torus.
REM
REM  batch 16 is a hard ceiling on 19x19 (batch 32 OOMs on 6 GB VRAM).
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
  echo  Generate the JSONL via scr\gen_katago_data.py ^(drives katago\katago.exe^),
  echo  then re-run: base\distill_cnn2d.bat ^<data.jsonl^>
  echo.
  pause
  exit /b 1
)

echo Data: %~dp0..\%DATA%
echo Interpreter: %PY%

if exist "%~dp0cnn2d\round*.pt" (
  echo Resume: latest round*.pt under base\cnn2d
) else (
  echo No round*.pt under base\cnn2d - starting a new distillation.
)

cd /d "%~dp0..\scr"
"%PY%" distill.py --net 2dcnn --data "../%DATA%" --graph-key 0 --outdir ../base/cnn2d --hidden 512 --n-blocks 20 --epochs 10 --batch-size 16 --lr 1e-4 --value-weight 30 --own-weight 5 --aug-from-epoch 4 --device cuda --resume

echo.
echo Distillation finished. The base 2DCNN is at base\cnn2d\new.pt.
pause
