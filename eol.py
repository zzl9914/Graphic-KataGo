"""Line endings for this repo.

Write everything as LF (editor / tool default). Windows ``cmd.exe`` needs
CRLF on ``*.bat``. After writing bats, run this; ``--to lf`` is the reverse.
Do not hand-convert files, and do not change encoding (UTF-8 vs ANSI).

  python eol.py                  # all *.bat under the repo → CRLF
  python eol.py path [path...]   # files or dirs (dirs: *.bat inside)
  python eol.py --to lf          # reverse: those files → LF
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "katago",
    "distill_data",
    "dist_run",
    "_dist_cache",
    "models",
    "node_modules",
}

TO_CRLF = "crlf"
TO_LF = "lf"


def _skip_dir(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES or path.name.startswith("cur_mod_")


def _iter_bats(base: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not _skip_dir(Path(d))]
        for name in filenames:
            if name.lower().endswith(".bat"):
                out.append(Path(dirpath) / name)
    return sorted(out)


def _collect(paths: list[Path]) -> list[Path]:
    if not paths:
        return _iter_bats(ROOT)
    files: list[Path] = []
    for raw in paths:
        p = raw if raw.is_absolute() else (Path.cwd() / raw)
        p = p.resolve()
        if p.is_dir():
            files.extend(_iter_bats(p))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(str(raw))
    # stable unique
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _normalize_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def convert(data: bytes, to: str) -> bytes:
    lf = _normalize_lf(data)
    if to == TO_CRLF:
        return lf.replace(b"\n", b"\r\n")
    return lf


def apply(path: Path, to: str) -> str:
    before = path.read_bytes()
    after = convert(before, to)
    if after == before:
        return "ok"
    path.write_bytes(after)
    return "wrote"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert line endings. Default: LF → CRLF on *.bat.")
    ap.add_argument(
        "--to", choices=(TO_CRLF, TO_LF), default=TO_CRLF,
        help="crlf = LF→CRLF (default, for cmd.exe bats); lf = reverse")
    ap.add_argument(
        "paths", nargs="*", type=Path,
        help="files or directories; default is all *.bat under the repo")
    args = ap.parse_args(argv)
    try:
        files = _collect(args.paths)
    except FileNotFoundError as e:
        print(f"not found: {e}", file=sys.stderr)
        return 1
    if not files:
        print("no files")
        return 0
    n_wrote = 0
    for f in files:
        status = apply(f, args.to)
        rel = f
        try:
            rel = f.relative_to(ROOT)
        except ValueError:
            pass
        print(f"{status:5} {rel.as_posix()}")
        if status == "wrote":
            n_wrote += 1
    print(f"--to {args.to}: {n_wrote} written, {len(files) - n_wrote} unchanged, {len(files)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
