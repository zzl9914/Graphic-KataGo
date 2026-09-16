@echo off
setlocal
cd /d "%~dp0scr"
if not exist "web_ui\server.py" (
  echo Cannot find scr\web_ui\server.py
  pause
  exit /b 1
)

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

REM Tsinghua PyPI mirror
set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_HOST=pypi.tuna.tsinghua.edu.cn"

echo GKT web UI  http://127.0.0.1:8765/
echo Interpreter: %PY%
echo Ctrl+C stops the server.
echo.

"%PY%" -c "import numpy, gkt_cpp; gkt_cpp.require_native()" 2>nul
if errorlevel 1 (
  echo Missing numpy or gkt_native. Installing deps / build native ...
  "%PY%" -m pip install -i %PIP_INDEX% --trusted-host %PIP_HOST% numpy pybind11
  "%PY%" "%~dp0cpp\build.py"
  if errorlevel 1 (
    echo Native build failed. Install Visual Studio C++ and retry: python cpp/build.py
    pause
    exit /b 1
  )
)

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765/"
"%PY%" web_ui\server.py
echo.
echo Server exited.
pause
