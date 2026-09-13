"""Build gkt_native with the ScopeCppSDK (UCRT/um) injected into INCLUDE/LIB.

The stock setuptools MSVC compiler cannot locate this VS 18 "ScopeCppSDK"
layout (it looks for the classic `Windows Kits\\10` tree), so cl.exe fails on
`io.h`. We hand it the SDK include/lib dirs via the environment that cl.exe and
link.exe already consult, then run the normal setuptools build.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDK = (r"D:\Program Files\Microsoft Visual Studio\18\Community"
        r"\SDK\ScopeCppSDK\vc15\SDK")

_INC = os.pathsep.join([
    os.path.join(_SDK, "include", "ucrt"),
    os.path.join(_SDK, "include", "um"),
    os.path.join(_SDK, "include", "shared"),
])
_LIB = os.path.join(_SDK, "lib")

env = os.environ.copy()
env["INCLUDE"] = _INC + os.pathsep + env.get("INCLUDE", "")
env["LIB"] = _LIB + os.pathsep + env.get("LIB", "")
# link.exe locates rc.exe (resource compiler) via PATH.
env["PATH"] = os.path.join(_SDK, "bin") + os.pathsep + env.get("PATH", "")

cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
print("INCLUDE ->", _INC)
print("LIB ->", _LIB)
print(" ".join(cmd))
r = subprocess.call(cmd, cwd=_HERE, env=env)
sys.exit(r)
