"""Build gkt_native and copy the extension into scr/.

Prefers setuptools + pybind11 (uses MSVC on Windows). CMake is optional.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
SCR = os.path.join(REPO, "scr")


def _copy_ext() -> int:
    found = []
    for dirpath, _, files in os.walk(ROOT):
        if "build" in dirpath.split(os.sep) or dirpath == ROOT or dirpath == SCR:
            for f in files:
                if f.startswith("gkt_native") and (f.endswith(".pyd") or f.endswith(".so")):
                    found.append(os.path.join(dirpath, f))
    if os.path.isdir(SCR):
        for f in os.listdir(SCR):
            if f.startswith("gkt_native") and (f.endswith(".pyd") or f.endswith(".so")):
                found.append(os.path.join(SCR, f))
    if not found:
        print("build produced no gkt_native extension")
        return 1
    src = max(found, key=os.path.getmtime)
    dst = os.path.join(SCR, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    print(f"native module: {dst}")
    return 0


def main() -> int:
    try:
        import pybind11  # noqa: F401
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pybind11",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--trusted-host", "pypi.tuna.tsinghua.edu.cn",
        ])
    env = os.environ.copy()
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    print(" ".join(cmd))
    r = subprocess.call(cmd, cwd=ROOT, env=env)
    if r != 0:
        return r
    return _copy_ext()


if __name__ == "__main__":
    sys.exit(main())
