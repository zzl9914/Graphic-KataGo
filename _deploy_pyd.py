"""Deploy the freshly built gkt_native.pyd from cpp/ into scr/.

Run this AFTER the training process releases the old ``scr/*.pyd`` (a running
trainer holds it open, so the copy fails with WinError 32 while it is alive).
The pyd is built first via ``_build_with_sdk.py`` (which injects the VS 18
ScopeCppSDK include/lib that stock setuptools cannot locate).
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCR = os.path.join(os.path.dirname(_HERE), "scr")

src = os.path.join(_HERE, "gkt_native.cp314-win_amd64.pyd")
dst = os.path.join(_SCR, "gkt_native.cp314-win_amd64.pyd")

if not os.path.isfile(src):
    print("no built pyd at", src, "-> run _build_with_sdk.py first")
    sys.exit(1)

try:
    shutil.copy2(src, dst)
    print("deployed ->", dst)
except PermissionError as e:
    print("BLOCKED (old pyd still in use by a running process):", e)
    print("Stop the trainer / UI / any python holding gkt_native, then re-run.")
    sys.exit(2)
