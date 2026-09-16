@echo off
REM Generate distill JSONL: 300 games, KataGo maxVisits=350 (sim).
REM Writes distill_data\m2_19x19.jsonl (overwrites). Then: base\distill_*.bat
REM
REM Usage:
REM   distill_data\gen_m2_19x19.bat
REM   distill_data\gen_m2_19x19.bat [extra gen_katago_data.py args...]

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

if not exist "%~dp0..\katago\katago.exe" (
  echo ERROR: missing %~dp0..\katago\katago.exe
  pause
  exit /b 1
)
if not exist "%~dp0..\katago\b18c384nbt.bin.gz" (
  echo ERROR: missing %~dp0..\katago\b18c384nbt.bin.gz
  pause
  exit /b 1
)
if not exist "%~dp0..\katago\analysis_distill.cfg" (
  echo ERROR: missing %~dp0..\katago\analysis_distill.cfg
  pause
  exit /b 1
)

echo Interpreter: %PY%
echo games=300  visits=350  max-moves=350
echo out: %~dp0m2_19x19.jsonl

cd /d "%~dp0..\scr"
"%PY%" gen_katago_data.py ^
  --katago ../katago/katago.exe ^
  --model ../katago/b18c384nbt.bin.gz ^
  --config ../katago/analysis_distill.cfg ^
  --games 300 ^
  --visits 350 ^
  --max-moves 350 ^
  --out ../distill_data/m2_19x19.jsonl ^
  %*

echo.
echo Done. Distill: base\distill_gnn.bat
pause
