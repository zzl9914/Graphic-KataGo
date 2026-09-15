@echo off
REM =====================================================================
REM  base/cultivate2_cnn2d - 基础培养2 (2DCNN)
REM  *** NOT official cross-graph training (that is starter/train_*.bat) ***
REM
REM  After distillation (基础培养 / base/cnn2d/new.pt), before
REM  starter/train_*.bat (30/5, mix, all graphs, --infinite, Arena).
REM
REM  20 rounds on graph 0 (19x19) only, value/own 30/5, mix, --no-arena, then STOP.
REM  Rounds 1-10: freeze the distilled policy readout; train trunk + value/own/aux.
REM  Policy CE still backprops into the trunk; readout weights do not move.
REM  Rounds 11-20: unfreeze policy; still graph 0 only.
REM
REM  Output: base/cultivate2/cnn2d/new.pt
REM  Re-run the same bat to resume from that file. If it does not exist yet,
REM  --resume the distill checkpoint. No distill product = error (no random init).
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

echo Interpreter: %PY%
echo Stage: cultivate2 / 2DCNN - graph 0, 20 rounds, 30/5, freeze policy 1-10

if exist "%~dp0cultivate2\cnn2d\new.pt" (
  echo Resume: base\cultivate2\cnn2d\new.pt
  set "RESUME=../base/cultivate2/cnn2d/new.pt"
) else if exist "%~dp0cnn2d\new.pt" (
  echo Resume: base\cnn2d\new.pt - distill / 基础培养
  set "RESUME=../base/cnn2d/new.pt"
) else (
  echo [ERROR] missing distill checkpoint: %~dp0cnn2d\new.pt
  echo.
  echo  Run base\distill_cnn2d.bat first, then re-run this bat.
  echo.
  pause
  exit /b 1
)

cd /d "%~dp0..\scr"
"%PY%" gkt_train_gpu.py --net 2dcnn --hidden 512 --n-blocks 20 --device cuda --selfplay-device cuda --graphs 0 --rounds 20 --freeze-policy-until-round 10 --value-weight 30 --own-weight 5 --q-lambda 0.5 --lr 1e-4 --temperature 0.1 --sim 256 --workers 1 --gpw 32 --steps 16 --no-arena --buffer-drop-from-round 21 --model-snapshot-rounds 25 --outdir ../base/cultivate2/cnn2d --resume !RESUME!

echo.
echo cultivate2 done. Model: base\cultivate2\cnn2d\new.pt
echo Official training remains starter\train_*.bat - not this bat.
pause
