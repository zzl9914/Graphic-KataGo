"""Build gkt_native and leave the extension in cpp/.

Two Windows issues this script papers over:

1. VS 18's ScopeCppSDK layout is not the classic ``Windows Kits\\10`` tree
   setuptools looks for, so cl.exe can fail on ``io.h``. We prepend those
   include/lib dirs.
2. setuptools does not yet probe VS 2026, so it reports "MSVC 14.0 required"
   unless vcvars already put cl.exe on PATH. We call vcvars64.bat first and
   set DISTUTILS_USE_SDK.

Copy into scr/ with ``python cpp/_deploy_pyd.py`` after a trainer releases
the old pyd.
"""
from __future__ import annotations
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VS = r"D:\Program Files\Microsoft Visual Studio\18\Community"
_VCVARS = os.path.join(_VS, r"VC\Auxiliary\Build\vcvars64.bat")
_SDK = os.path.join(_VS, r"SDK\ScopeCppSDK\vc15\SDK")

_INC = os.pathsep.join([
    os.path.join(_SDK, "include", "ucrt"),
    os.path.join(_SDK, "include", "um"),
    os.path.join(_SDK, "include", "shared"),
])
_LIB = os.path.join(_SDK, "lib")


def _env_after_vcvars() -> dict:
    env = os.environ.copy()
    if os.path.isfile(_VCVARS):
        dumped = subprocess.check_output(
            ["cmd", "/c", f'call "{_VCVARS}" >nul && set'],
            encoding="mbcs",
            errors="replace",
        )
        for line in dumped.splitlines():
            if not line or line.startswith("=") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key] = val
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    env["INCLUDE"] = _INC + os.pathsep + env.get("INCLUDE", "")
    env["LIB"] = _LIB + os.pathsep + env.get("LIB", "")
    env["PATH"] = os.path.join(_SDK, "bin") + os.pathsep + env.get("PATH", "")
    return env


def main() -> int:
    env = _env_after_vcvars()
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    print("INCLUDE ->", _INC)
    print("LIB ->", _LIB)
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=_HERE, env=env)


if __name__ == "__main__":
    sys.exit(main())
