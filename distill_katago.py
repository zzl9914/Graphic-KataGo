"""KataGo -> GKT distillation JSONL conversion.

This module turns KataGo's raw analysis-engine output into the JSONL records
that ``distill.py`` consumes. It is the ONLY place that knows about KataGo's
coordinate system; everything downstream is in gkt vertex order.

The actual data-generation driver is ``gen_katago_data.py``, which launches
KataGo's analysis engine (``katago/katago.exe analysis``), plays self-play
games to produce diverse positions, queries policy / scoreLead / ownership at
each one, and calls ``parse_analysis`` here to write the JSONL. This module
just supplies the coordinate/format conversion primitives.

Coordinate mapping (verified empirically against KataGo v1.18.0 output):

  KataGo's ``ownership`` is a flat list of length ``boardYSize*boardXSize`` in
  ROW-MAJOR order starting at the TOP-LEFT (A19) and ending at the bottom-right
  (T1). That is exactly index ``y * boardXSize + x`` with y = row (0 = TOP),
  x = column (0 = left). Positive ownership = black, negative = white.

  gkt grids are also row-major ``A_{i*n+j}`` with i = row (0 = TOP), j = column
  (0 = left). Because ``boardXSize == cols`` and ``n == rows*cols``, the two
  indexings are IDENTICAL: ``gkt[i*cols + j] == katago[i*boardXSize + j]``.
  No flip is needed (the old code's vertical flip was wrong).

  ``scoreLead`` (with ``reportAnalysisWinratesAs = SIDETOMOVE`` in the config)
  is the mover's lead in points. Distill maps it to abs (stones) and
  rto (``/n``) separately.

GTP note: KataGo GTP column letters skip 'I' (A..H, J..T); row "1" is the
bottom row. So a GTP move string maps to a gkt index as
``(rows - row_number) * cols + col``.
"""
from __future__ import annotations

import json

# KataGo GTP column letters (skips 'I') — 19 columns for board size 19.
_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def gtp_to_gkt_index(move: str, rows: int, cols: int) -> int:
    """GTP move string -> gkt flat index (row-major, i=0=TOP, j=0=left).

    ``move`` is like "Q16" (column letter + row number, row 1 = bottom).
    Returns the gkt index ``(rows - row_number) * cols + col``. Pass / resign /
    empty return -1.
    """
    move = (move or "").strip().lower()
    if move in ("pass", "resign", ""):
        return -1
    col = _GTP_COLS.index(move[0].upper())
    row_number = int(move[1:])          # 1 = bottom row
    i = rows - row_number               # row index from top (0 = top)
    return i * cols + col


def gkt_index_to_gtp(gidx: int, rows: int, cols: int) -> str:
    """gkt flat index -> GTP move string (inverse of gtp_to_gkt_index)."""
    if gidx < 0 or gidx >= rows * cols:
        return "pass"
    i, j = divmod(gidx, cols)           # i = row from top, j = column
    row_number = rows - i               # 1 = bottom row
    return _GTP_COLS[j] + str(row_number)


def katago_flat_to_gkt(flat, rows: int, cols: int):
    """KataGo flat array -> gkt row-major list.

    KataGo ownership is already ``y * boardXSize + x`` with y=0 = top, which is
    identical to gkt's ``i * cols + j`` (i=0 = top). This is therefore just a
    length-preserving copy (with zero-padding if the input is short; extra
    entries are dropped)."""
    n = rows * cols
    out = [0.0] * n
    k = min(len(flat), n)
    out[:k] = list(flat)[:k]
    return out


def parse_analysis(resp: dict, board: list, to_move: int, rows: int, cols: int):
    """One KataGo analysis response -> one distill record (gkt vertex order).

    Args:
        resp:     the response object for ONE analyzed turn. It carries
                  ``moveInfos`` and ``ownership`` at the TOP level and
                  ``scoreLead`` / ``currentPlayer`` inside ``rootInfo``.
        board:    current board colors in *gkt* order (row-major, 0/1/2).
        to_move:  1=black, 2=white.
        rows/cols: board height / width (19 for a standard board).

    Returns the JSONL dict that ``distill.py`` expects:
        {"to_move", "board", "policy" (n+1), "scoreLead", "ownership" (n)}
        ``gen_katago_data.py`` also writes ``game_id`` (0-based game index).
    """
    n = rows * cols
    ri = resp.get("rootInfo", resp)

    # policy: rebuild the n+1 vector from moveInfos, weighted by MCTS VISITS
    # (the strong search distribution; falls back to `prior` if visits absent).
    policy = [0.0] * (n + 1)
    mis = resp.get("moveInfos")
    if mis:
        for mi in mis:
            m = mi.get("move")
            w = mi.get("visits")
            if w is None:
                w = mi.get("prior", 0.0)
            w = float(w)
            if m and str(m).lower() == "pass":
                policy[n] = w
            else:
                gidx = gtp_to_gkt_index(m, rows, cols)
                if 0 <= gidx < n:
                    policy[gidx] = w
    else:
        # no moveInfos: uniform over empties + pass
        for i, c in enumerate(board):
            if c == 0:
                policy[i] = 1.0
        policy[n] = 1.0
    s = sum(policy)
    if s > 0:
        policy = [p / s for p in policy]

    score_lead = float(ri.get("scoreLead", 0.0))

    own_kat = resp.get("ownership")
    ownership = katago_flat_to_gkt(own_kat, rows, cols) if own_kat is not None \
        else [0.0] * n

    return {
        "to_move": int(to_move),
        "board": [int(c) for c in board],
        "policy": policy,
        "scoreLead": score_lead,
        "ownership": ownership,
    }


def apply_gkt_move(board, gidx, color):
    """Place a stone on the gkt-ordered board (no capture logic; the driver
    relies on KataGo for legality via the ``moves`` history). Returns a copy."""
    b = list(board)
    if 0 <= gidx < len(b):
        b[gidx] = color
    return b


if __name__ == "__main__":
    # Library only; the driver is gen_katago_data.py. Running directly does a
    # tiny self-check of the coordinate primitives.
    assert _GTP_COLS.index("A") == 0 and _GTP_COLS.index("T") == 18
    assert gtp_to_gkt_index("A1", 19, 19) == 18 * 19 + 0   # bottom-left -> top row
    assert gtp_to_gkt_index("T19", 19, 19) == 0 * 19 + 18  # top-right -> top row
    assert gtp_to_gkt_index("Q16", 19, 19) == 3 * 19 + 15
    assert gtp_to_gkt_index("pass", 19, 19) == -1
    print("distill_katago coordinate primitives OK")
