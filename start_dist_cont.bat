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

set "URL=%~1"
set "TOKEN=%~2"
set "DEV=%~3"
if "!URL!"=="" if defined GKT_DIST_URL set "URL=%GKT_DIST_URL%"
if "!TOKEN!"=="" if defined GKT_DIST_TOKEN set "TOKEN=%GKT_DIST_TOKEN%"
if "!DEV!"=="" if defined GKT_DIST_DEVICE set "DEV=%GKT_DIST_DEVICE%"
if "!DEV!"=="" set "DEV=cuda"

if /i "!URL!"=="-h" goto :usage
if /i "!URL!"=="/?" goto :usage
if /i "!URL!"=="--help" goto :usage

if "!URL!"=="" (
  set /p URL=Coach URL, e.g. http://192.168.1.10:8877 : 
)
if "!TOKEN!"=="" (
  set /p TOKEN=HTTP token: 
)
if "!URL!"=="" goto :usage
if "!TOKEN!"=="" (
  echo ERROR: token required.
  pause
  exit /b 1
)
if "!TOKEN:~15,1!"=="" (
  echo ERROR: token must be at least 16 characters.
  pause
  exit /b 1
)

set "INSECURE="
if defined GKT_DIST_INSECURE set "INSECURE=--insecure"

echo Interpreter: %PY%
echo Dist contribute: url=!URL! device=!DEV!
echo Ctrl+C stops this client. Token is not printed.
echo.

cd /d "%~dp0scr"
"%PY%" gkt_dist.py contribute --url "!URL!" --token "!TOKEN!" --device !DEV! --cache ../_dist_cache !INSECURE!
echo.
echo Contribute exited.
pause
exit /b 0

:usage
echo Usage: start_dist_cont.bat [url] [token] [cuda^|cpu]
echo Example: start_dist_cont.bat http://192.168.1.10:8877
echo Token: arg 2, or GKT_DIST_TOKEN, or prompt. Device default cuda.
echo Coach: start_dist_main.bat
pause
exit /b 1
