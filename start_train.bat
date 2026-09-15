@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Portable CPython 3.14 (gkt_native.cp314-*.pyd). No absolute paths:
REM prefer the 'py' launcher (py -3.14), fall back to 'python' on PATH.
REM Do not use "py -3" (may pick 3.14t free-threading) or 3.13 (no .pyd).
set "PY="
for /f "delims=" %%I in ('py -3.14 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%I"
if not defined PY (
  python -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,14) else 1)" 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo ERROR: CPython 3.14 ^(non-free-threading^) required for gkt_native.cp314-*.pyd.
  echo        Install Python 3.14 or ensure 'py -3.14' / 'python' resolves to it.
  pause
  exit /b 1
)

set "KIND=%~1"
if /i "%KIND%"=="gnn" goto :gnn
if /i "%KIND%"=="mlp" goto :mlp
if /i "%KIND%"=="cnn1d" goto :cnn1d
if /i "%KIND%"=="cnn2d" goto :cnn2d
echo Usage: start_train.bat gnn^|mlp^|cnn1d^|cnn2d [trainer args...]
echo Graph-Go:  train_gnn.bat / train_mlp.bat / train_cnn1d.bat / train_cnn2d.bat
echo Gomoku:    train_gomoku_gnn.bat / train_gomoku_mlp.bat / train_gomoku_cnn1d.bat / train_gomoku_cnn2d.bat
echo AntiGomoku: train_antigomoku_gnn.bat / train_antigomoku_mlp.bat / train_antigomoku_cnn1d.bat / train_antigomoku_cnn2d.bat
pause
exit /b 1

:gnn
set "KINDDIR=gnn"
set "CKPT=new.pt"
set "TRAINER=gkt_train_gpu.py"
set "NET=gnn"
set "EXTRA=--device cuda --selfplay-device cuda"
goto :dispatch

:mlp
set "KINDDIR=mlp"
set "CKPT=new.npz"
set "TRAINER=gkt_train_cpu.py"
set "NET=mlp"
set "EXTRA="
goto :dispatch

:cnn1d
set "KINDDIR=cnn1d"
set "CKPT=new.npz"
set "TRAINER=gkt_train_cpu.py"
set "NET=1dcnn"
set "EXTRA="
goto :dispatch

:cnn2d
set "KINDDIR=cnn2d"
set "CKPT=new.pt"
set "TRAINER=gkt_train_gpu.py"
set "NET=2dcnn"
set "EXTRA=--device cuda --selfplay-device cuda"
goto :dispatch

:dispatch
shift
set "REST="
:collect
if "%~1"=="" goto :gotrest
set "REST=!REST! %1"
shift
goto :collect
:gotrest

set "RULES=go"
echo !REST! | findstr /I /C:"--rules antigomoku" /C:"--rules=antigomoku" /C:"--rules anti-gomoku" /C:"--rules=anti-gomoku" /C:"--rules anti_gomoku" /C:"--rules=anti_gomoku" >nul
if not errorlevel 1 set "RULES=antigomoku"
if /i not "%RULES%"=="antigomoku" (
  echo !REST! | findstr /I /C:"--rules gomoku" /C:"--rules=gomoku" >nul
  if not errorlevel 1 set "RULES=gomoku"
)

if /i "%RULES%"=="antigomoku" (
  set "MODDIR=cur_mod_antigomoku_!KINDDIR!"
) else if /i "%RULES%"=="gomoku" (
  set "MODDIR=cur_mod_gomoku_!KINDDIR!"
) else (
  set "MODDIR=cur_mod_!KINDDIR!"
)

REM Optional override for experiments (M0/M1/M2): GKT_OUTDIR is relative
REM to the project root (this bat folder). It overrides MODDIR and
REM forwards an explicit --outdir (relative to scr/) to the trainer.
if defined GKT_OUTDIR (
  set "MODDIR=!GKT_OUTDIR!"
  set "OUTDIR_ARG=--outdir ../!GKT_OUTDIR!"
) else (
  set "OUTDIR_ARG="
)

if not exist "%~dp0scr\%TRAINER%" (
  echo Cannot find scr\%TRAINER%
  pause
  exit /b 1
)

set "RESUME="
if exist "%~dp0!MODDIR!\%CKPT%" (
  echo Resume: !MODDIR!\%CKPT%
  set "RESUME=--resume ../!MODDIR!/%CKPT%"
) else (
  echo No !MODDIR!\%CKPT% - starting a new net.
)

echo Interpreter: %PY%
echo Trainer: %TRAINER%  net=%NET%  rules=%RULES%  outdir=!MODDIR!
echo Requires gkt_native: python cpp/build.py
echo Ctrl+C stops training. Next launch resumes if %CKPT% exists.
echo.

cd /d "%~dp0scr"
"%PY%" %TRAINER% --net %NET% %EXTRA% !OUTDIR_ARG! !RESUME! !REST!
echo.
echo Training exited.
pause
