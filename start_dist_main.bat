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

if not exist "%~dp0scr\gkt_dist.py" (
  echo Cannot find scr\gkt_dist.py
  pause
  exit /b 1
)

set "KIND=%~1"
if "!KIND!"=="" set "KIND=gnn"
if /i "!KIND:~0,2!"=="--" (
  set "KIND=gnn"
) else if not "%~1"=="" (
  shift
)

if /i "!KIND!"=="gnn" goto :gnn
if /i "!KIND!"=="mlp" goto :mlp
if /i "!KIND!"=="cnn1d" goto :cnn1d
if /i "!KIND!"=="cnn2d" goto :cnn2d
echo Usage: start_dist_main.bat [gnn^|mlp^|cnn1d^|cnn2d] [init args...]
echo Coach: init dist_run if needed, then open shuffle / train / serve windows.
echo First run needs a seed in cur_mod_* and GKT_DIST_TOKEN or a prompt.
echo Clients: start_dist_cont.bat http://THIS_PC_LAN_IP:8877
echo Do not run start_train.bat on the same GPU as train.
pause
exit /b 1

:gnn
set "KINDDIR=gnn"
set "CKPT=new.pt"
set "NET=gnn"
set "TRAINDEV=cuda"
goto :dispatch

:mlp
set "KINDDIR=mlp"
set "CKPT=new.npz"
set "NET=mlp"
set "TRAINDEV=cpu"
goto :dispatch

:cnn1d
set "KINDDIR=cnn1d"
set "CKPT=new.npz"
set "NET=1dcnn"
set "TRAINDEV=cpu"
goto :dispatch

:cnn2d
set "KINDDIR=cnn2d"
set "CKPT=new.pt"
set "NET=2dcnn"
set "TRAINDEV=cuda"
goto :dispatch

:dispatch
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
if /i not "!RULES!"=="antigomoku" (
  echo !REST! | findstr /I /C:"--rules gomoku" /C:"--rules=gomoku" >nul
  if not errorlevel 1 set "RULES=gomoku"
)

if /i "!RULES!"=="antigomoku" (
  set "MODDIR=cur_mod_antigomoku_!KINDDIR!"
) else if /i "!RULES!"=="gomoku" (
  set "MODDIR=cur_mod_gomoku_!KINDDIR!"
) else (
  set "MODDIR=cur_mod_!KINDDIR!"
)

set "BASEDIR=dist_run"
if defined GKT_DIST_BASEDIR set "BASEDIR=!GKT_DIST_BASEDIR!"
set "PORT=8877"
if defined GKT_DIST_PORT set "PORT=!GKT_DIST_PORT!"
set "HOST=0.0.0.0"
if defined GKT_DIST_HOST set "HOST=!GKT_DIST_HOST!"

echo Interpreter: %PY%
echo Dist main: net=%NET% rules=!RULES! basedir=!BASEDIR! train-device=!TRAINDEV!
echo Ctrl+C in each child window stops that process.
echo Do not run start_train.bat on this GPU while train is running.
echo.

if exist "%~dp0!BASEDIR!\run.json" (
  echo Already initialized: !BASEDIR!\run.json
  goto :launch
)

set "SEED="
if exist "%~dp0!MODDIR!\!CKPT!" (
  set "SEED=../!MODDIR!/!CKPT!"
  echo Seed: !MODDIR!\!CKPT!
  goto :haveseed
)
if /i not "!RULES!"=="go" goto :noseed
if exist "%~dp0base\cultivate2\!KINDDIR!\!CKPT!" (
  set "SEED=../base/cultivate2/!KINDDIR!/!CKPT!"
  echo Seed: base\cultivate2\!KINDDIR!\!CKPT!
  goto :haveseed
)
if exist "%~dp0base\!KINDDIR!\!CKPT!" (
  set "SEED=../base/!KINDDIR!/!CKPT!"
  echo Seed: base\!KINDDIR!\!CKPT!
  goto :haveseed
)
:noseed
echo ERROR: no seed checkpoint. Train locally first, or copy a net into !MODDIR!\!CKPT!
pause
exit /b 1

:haveseed
if not defined GKT_DIST_TOKEN (
  echo First init: HTTP token, at least 16 characters. Contribute clients need the same string.
  set /p GKT_DIST_TOKEN=Token: 
)
if not defined GKT_DIST_TOKEN (
  echo ERROR: token required to init HTTP serve.
  pause
  exit /b 1
)
if "!GKT_DIST_TOKEN:~15,1!"=="" (
  echo ERROR: token must be at least 16 characters.
  pause
  exit /b 1
)

cd /d "%~dp0scr"
echo Init !BASEDIR! ...
"%PY%" gkt_dist.py init --basedir ../!BASEDIR! --from !SEED! --net !NET! --token "!GKT_DIST_TOKEN!" !REST!
if errorlevel 1 (
  echo init failed.
  pause
  exit /b 1
)

:launch
cd /d "%~dp0scr"
start "GKT-dist-shuffle" cmd /k call "%PY%" gkt_dist.py shuffle --basedir ../!BASEDIR!
start "GKT-dist-train" cmd /k call "%PY%" gkt_dist.py train --basedir ../!BASEDIR! --device !TRAINDEV!
start "GKT-dist-serve" cmd /k call "%PY%" gkt_dist.py serve --basedir ../!BASEDIR! --host !HOST! --port !PORT!
if defined GKT_DIST_GATE (
  start "GKT-dist-gate" cmd /k call "%PY%" gkt_dist.py gate --basedir ../!BASEDIR! --device !TRAINDEV!
)

echo.
echo Opened shuffle / train / serve.
echo Clients: start_dist_cont.bat http://THIS_PC_LAN_IP:!PORT!
echo Windows firewall: allow inbound TCP !PORT! on the LAN.
echo.
pause
