"""Generate KataGo distillation JSONL (pipeline stage 1 data).

Launches ``katago/katago.exe analysis`` and plays self-play games with
temperature-sampled moves, querying policy / scoreLead / ownership at each
position. Every position is written as a distill JSONL record (gkt vertex
order) that ``distill.py`` consumes directly.

The board is reconstructed by replaying the same moves through the gkt native
engine (``gkt_cpp``), so captures / superko match gkt's own rules exactly and
the resulting ``board`` field is a correct gkt-order occupancy array.

Usage:
  python gen_katago_data.py \
      --katago ../katago/katago.exe \
      --model ../katago/b18c384nbt.bin.gz \
      --config ../katago/analysis_distill.cfg \
      --games 200 --max-moves 350 --visits 200 \
      --out ../distill_data/m2_19x19.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gkt_cpp
from graphs import get_builtin
from distill_katago import parse_analysis, gkt_index_to_gtp, gtp_to_gkt_index

PASS_INDEX = None  # set per board size (== n)


def _sample_move(move_infos, temperature, rng, rows, cols):
    """Sample a gkt move index from KataGo moveInfos (visits-weighted, tempered).

    Returns (gidx, gtp) where gidx is the gkt index (or n for pass) and gtp is
    the corresponding GTP move string ("pass" for the pass slot).
    """
    moves = []   # (gidx_or_pass, gtp, weight)
    for mi in move_infos:
        m = mi.get("move")
        w = float(mi.get("visits") or mi.get("prior") or 0.0)
        if w <= 0:
            continue
        if str(m).lower() == "pass":
            moves.append((PASS_INDEX, "pass", w))
        else:
            gidx = gtp_to_gkt_index(m, rows, cols)
            if 0 <= gidx < rows * cols:
                moves.append((gidx, m, w))
    if not moves:
        return PASS_INDEX, "pass"

    if temperature <= 0:
        gidx, gtp, _ = max(moves, key=lambda t: t[2])
    else:
        probs = np.array([w for _, _, w in moves], dtype=np.float64)
        probs = probs ** (1.0 / max(temperature, 1e-3))
        probs /= probs.sum()
        idx = rng.choices(range(len(moves)), weights=probs, k=1)[0]
        gidx, gtp, _ = moves[idx]
    return gidx, gtp


def _temperature_for(ply, max_moves):
    """Opening diversity (tau ~1) decaying to near-argmax (tau ~0.1)."""
    if ply < 10:
        return 1.0
    return 0.1


def main():
    ap = argparse.ArgumentParser(description="Generate KataGo distill JSONL")
    ap.add_argument("--katago", default="../katago/katago.exe")
    ap.add_argument("--model", default="../katago/b18c384nbt.bin.gz")
    ap.add_argument("--config", default="../katago/analysis_distill.cfg")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-moves", type=int, default=350)
    ap.add_argument("--visits", type=int, default=200)
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--out", default="../distill_data/m2_19x19.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global PASS_INDEX
    n = args.board_size * args.board_size
    PASS_INDEX = n
    rows = cols = args.board_size
    rng = random.Random(args.seed)

    # gkt native engine for board reconstruction (capture-correct).
    native = gkt_cpp.require_native()
    graph = get_builtin("0") if args.board_size == 19 else get_builtin("0.5")
    if len(graph.vertices) != n:
        raise SystemExit(f"builtin graph has {len(graph.vertices)} vertices != n={n}")
    ng = gkt_cpp.py_graph_to_native(graph)

    # Launch KataGo analysis engine.
    proc = subprocess.Popen(
        [args.katago, "analysis", "-config", args.config, "-model", args.model],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    outf = open(args.out, "w", encoding="utf-8")

    total_records = 0
    try:
        for g in range(args.games):
            game = native.Game(ng, 2, rules="go")
            gtp_moves = []          # KataGo query history: [["B","Q16"], ...]
            ply = 0
            passes = 0
            while ply < args.max_moves:
                board = list(game.position.occupancy)  # gkt order 0/1/2
                to_move = int(game.position.to_move)

                # --- query KataGo for this position ---
                q = {"id": f"g{g}p{ply}",
                     "moves": gtp_moves,
                     "rules": "tromp-taylor", "komi": 0,
                     "boardXSize": cols, "boardYSize": rows,
                     "analyzeTurns": [ply],
                     "maxVisits": args.visits,
                     "includeOwnership": True}
                proc.stdin.write(json.dumps(q) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("KataGo analysis process exited early")
                resp = json.loads(line)

                # --- write the distill record ---
                rec = parse_analysis(resp, board, to_move, rows, cols)
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1

                # --- sample the next move (tempered visits) ---
                tau = _temperature_for(ply, args.max_moves)
                gidx, gtp = _sample_move(resp.get("moveInfos", []), tau, rng,
                                         rows, cols)

                # apply move to local engine + KataGo history
                if gtp == "pass":
                    gtp_moves.append([("B" if to_move == 1 else "W"), "pass"])
                    game.play(n)
                    passes += 1
                    if passes >= 2:
                        break
                else:
                    gtp_moves.append([("B" if to_move == 1 else "W"), gtp])
                    game.play(gidx)
                    passes = 0
                ply += 1

            if g % 10 == 0 or g == args.games - 1:
                print(f"[gen] game {g + 1}/{args.games}, records={total_records}",
                      flush=True)
    finally:
        outf.close()
        try:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print(f"[gen] done: {total_records} records -> {args.out}")


if __name__ == "__main__":
    main()
