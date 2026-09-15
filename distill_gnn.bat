@echo off
REM =====================================================================
REM  base/distill_gnn - KataGo distillation -> GNN base model (基础培养)
REM  *** PIPELINE STAGE 1 (distill). Next: cultivate2, then starter/train_*.bat ***
REM
REM  Produces the FIRST strong graph-agnostic GNN by supervised-pretraining
REM  against KataGo teacher labels (policy / score lead / ownership).
REM  Output: base/gnn/new.pt. Then run base\cultivate2_gnn.bat.
REM
REM  Why distillation (not from-zero): pure self-play stalls because the
REM  value head collapses to a constant (weak signal; 72h+ with no progress
REM  on a 6 GB GPU). KataGo labels give a real, discriminative signal from
REM  step 1. From-zero self-play is kept only as M0 (sanity gate).
REM
REM  Usage:
REM    base\distill_gnn.bat                    (distill_data/m2_19x19.jsonl)
REM    base\distill_gnn.bat <path-to.jsonl>    (explicit data, root-relative)
REM
REM  Re-run the same bat to resume: distill.py --resume loads the highest
REM  round*.pt in base/gnn/ and continues at epoch N+1. No checkpoint = epoch 1.
REM  3 stages x 10 epochs (round1-30): joint P+30V+5O @1e-4 (no freeze),
REM  then own @25x / value @100x with trunk frozen.
REM  --aug-from-epoch 4: epochs 1-3 keep original vertex numbering, then S_n.
REM
REM  batch 16 is a hard ceiling on 19x19 (n=361 adjacency): batch 32 OOMs.
REM  H=512 / 20 blocks matches the self-play trainer so the base net loads
REM  back with --resume unchanged.
REM =====================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Resolve CPython 3.14 (gkt_native.cp314-*.pyd); no absolute paths.
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
  echo  1. KataGo binary + b18c384nbt weights are under katago/ ^(already set up^).
  echo  2. Generate the JSONL:  python scr\gen_katago_data.py ^<see its --help^>
  echo     This drives katago\katago.exe analysis and writes
  echo     distill_data\m2_19x19.jsonl via scr\distill_katago.py.
  echo  3. Re-run: base\distill_gnn.bat ^<data.jsonl^>
  echo.
  pause
  exit /b 1
)

echo Data: %~dp0..\%DATA%
echo Interpreter: %PY%

if exist "%~dp0gnn\round*.pt" (
  echo Resume: latest round*.pt under base\gnn
) else (
  echo No round*.pt under base\gnn - starting a new distillation.
)

cd /d "%~dp0..\scr"
"%PY%" distill.py --net gnn --data "../%DATA%" --graph-key 0 --outdir ../base/gnn --hidden 512 --n-blocks 20 --epochs 10 --batch-size 16 --lr 1e-4 --value-weight 30 --own-weight 5 --aug-from-epoch 4 --device cuda --resume

echo.
echo Distillation finished. The base GNN is at base\gnn\new.pt.
echo Self-play finetune resumes from it (--resume base\gnn\new.pt).
pause
