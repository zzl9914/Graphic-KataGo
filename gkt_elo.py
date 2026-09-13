"""Shared Elo ratings across MLP / 1DCNN / 2DCNN / GNN / uniform-random.

Display only: does not accept or reject checkpoints. Training uses the Arena
gate in ``gkt_train_*.py`` / ``gkt_dist.py`` (on by default in the GPU
trainer, ``--no-arena`` to disable). Games use the same empty-board, fixed-sim,
argmax, no-Dirichlet protocol as Arena. Instead of a 0/1 win/loss, each game
feeds Elo a *continuous normalized score-lead* (``score_lead``, the value
head's target) mapped to [0, 1], so the rating difference reflects an average
per-point strength gap and is komi-independent.

  python gkt_elo.py --models-dir ../models --sim 100 --games 2 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphs import builtin_graphs, get_builtin, GOMOKU_TRAIN_KEYS, RULES_CHOICES, is_k_in_row_rules  # noqa: E402
from gkt import play_eval_match, score_lead  # noqa: E402

RANDOM_ID = "__random__"
SCALE = 400.0
K_FACTOR = 24.0
BASE_ELO = 1500.0
EXCLUDE_GRAPHS = {"2", "6"}
ELO_JSON = "elo.json"


def expected_score(ra: float, rb: float, scale: float = SCALE) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


def apply_elo(ra: float, rb: float, sa: float, k: float = K_FACTOR,
              scale: float = SCALE) -> Tuple[float, float]:
    ea = expected_score(ra, rb, scale)
    return ra + k * (sa - ea), rb + k * ((1.0 - sa) - (1.0 - ea))


def agent_ok_on_graph(kind: Optional[str], graph) -> bool:
    """2DCNN needs ``graph.grid``; others (and random) play any builtin."""
    lab = (kind or "").strip().upper()
    if lab == "2DCNN":
        return getattr(graph, "grid", None) is not None
    return True


def _peek(path: str) -> Tuple[str, int]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        from gkt_cpu import peek_cpu_tags
        return peek_cpu_tags(path)
    from gkt_gpu import peek_gpu_tags
    return peek_gpu_tags(path)


def _load_net(path: str, device: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        from gkt_cpu import load_cpu_net
        return load_cpu_net(path)
    from gkt_gpu import load_net
    net = load_net(path, device=device, graph=None)
    if hasattr(net, "eval"):
        net.eval()
    return net


def _bind_graph(net, graph) -> bool:
    if net is None:
        return True
    fn = getattr(net, "set_graph", None)
    if fn is None:
        return True
    try:
        fn(graph)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def list_checkpoints(models_dir: str) -> List[Dict]:
    out = []
    root = os.path.abspath(models_dir)
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".pt", ".pth", ".npz"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        try:
            kind, npl = _peek(path)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}")
            continue
        out.append({"id": name, "path": path, "kind": kind, "players": int(npl)})
    return out


def _max_moves(graph, rules: str = "go", factor: float = 2.0) -> int:
    n = int(len(graph.vertices))
    if is_k_in_row_rules(rules):
        return n
    return int(n * factor) + 40


def _net_of(agent: Dict, cache: Dict, graph, device: str):
    """Return the net, or None for random. Raises ValueError if this graph is unusable."""
    if agent["id"] == RANDOM_ID:
        return None
    key = agent["path"]
    if key not in cache:
        cache[key] = _load_net(key, device)
    net = cache[key]
    if not _bind_graph(net, graph):
        raise ValueError("set_graph failed")
    return net


def rate_pool(agents: List[Dict], graph_keys: List[str], games: int, sim: int,
              device: str, k: float, scale: float, seed: int,
              rules: str = "go", win_length: int = 5,
              log=print) -> Dict:
    rng = random.Random(seed)
    ratings = {a["id"]: BASE_ELO for a in agents}
    stats = {a["id"]: {"kind": a.get("kind"), "players": a.get("players", 2),
                       "n": 0, "score": 0.0} for a in agents}
    cache: Dict = {}
    pairs = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            pairs.append((agents[i], agents[j]))
    rng.shuffle(pairs)
    history = []
    t0 = time.time()
    for a, b in pairs:
        for key in graph_keys:
            g = get_builtin(key)
            if not agent_ok_on_graph(a.get("kind"), g):
                continue
            if not agent_ok_on_graph(b.get("kind"), g):
                continue
            amax = _max_moves(g, rules=rules)
            for gi in range(games):
                a_black = (gi % 2 == 0)
                try:
                    na = _net_of(a, cache, g, device)
                    nb = _net_of(b, cache, g, device)
                except (ValueError, RuntimeError, TypeError) as e:
                    log(f"skip {a['id']} vs {b['id']} on {key}: {e}")
                    break
                net_b = na if a_black else nb
                net_w = nb if a_black else na
                scores = play_eval_match({1: net_b, 2: net_w}, g, sim, amax,
                                         dirichlet_frac=0.0, batch_size=64,
                                         rules=rules, win_length=win_length)
                total_score = sum(float(scores[s]) for s in (1, 2))
                lead_black = score_lead(float(scores[1]), total_score,
                                        max(1, len(g.vertices)), 2)
                # A's normalized lead over B (sign flips when A is white).
                lead_a = lead_black if a_black else -lead_black
                # Map the continuous lead in [-1, 1] to a win-probability-like
                # score in [0, 1]: lead=-1 -> 0, lead=0 -> 0.5, lead=+1 -> 1.
                sa = (lead_a + 1.0) / 2.0
                ra, rb = apply_elo(ratings[a["id"]], ratings[b["id"]], sa,
                                   k=k, scale=scale)
                ratings[a["id"]], ratings[b["id"]] = ra, rb
                stats[a["id"]]["n"] += 1
                stats[b["id"]]["n"] += 1
                stats[a["id"]]["score"] += sa
                stats[b["id"]]["score"] += 1.0 - sa
                history.append({
                    "a": a["id"], "b": b["id"], "graph": key,
                    "sa": sa, "a_black": a_black,
                })
                log(f"{a['id']} vs {b['id']} graph={key} "
                    f"{'A' if a_black else 'B'} black  S_A={sa:.2f}  "
                    f"Elo {ra:.0f} / {rb:.0f}")
    players = {}
    for a in agents:
        i = a["id"]
        players[i] = {
            "elo": round(ratings[i], 1),
            "kind": a.get("kind"),
            "players": a.get("players", 2),
            "games": stats[i]["n"],
            "score": round(stats[i]["score"], 2),
        }
    return {
        "scale": scale,
        "k": k,
        "base": BASE_ELO,
        "sim": sim,
        "games_per_pair_graph": games,
        "rules": rules,
        "win_length": win_length,
        "graphs": graph_keys,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(time.time() - t0, 1),
        "players": players,
        "history": history,
    }


def default_graph_keys(rules: str = "go") -> List[str]:
    if is_k_in_row_rules(rules):
        return list(GOMOKU_TRAIN_KEYS)
    keys = [k for k in builtin_graphs() if k not in EXCLUDE_GRAPHS]
    prefer = ["0.5", "3", "5", "7", "R1"]
    ordered = [k for k in prefer if k in keys]
    ordered.extend(k for k in keys if k not in ordered)
    return ordered[:5]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Shared Elo board (display). Arena still gates training.")
    ap.add_argument("--models-dir", default=os.path.join(here, "..", "models"))
    ap.add_argument("--out", default="",
                    help="JSON path (default: <models-dir>/elo.json)")
    ap.add_argument("--sim", type=int, default=100,
                    help="MCTS simulations per move (Arena-style, fixed)")
    ap.add_argument("--games", type=int, default=2,
                    help="games per pair per graph (colours alternate)")
    ap.add_argument("--graphs", default="",
                    help="comma-separated builtin keys (default: a mixed 5)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=float, default=K_FACTOR)
    ap.add_argument("--no-random", action="store_true",
                    help="omit the uniform-legal random anchor")
    ap.add_argument("--include-multi", action="store_true",
                    help="include checkpoints tagged 3P/4P (still 2-player games)")
    ap.add_argument("--rules", default=None, choices=list(RULES_CHOICES),
                    help="default: antigomoku/gomoku if --models-dir path contains that name")
    ap.add_argument("--win-length", type=int, default=5,
                    help="Gomoku k-in-a-row (ignored for Graph-Go)")
    args = ap.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    rules = args.rules
    if rules is None:
        low = models_dir.replace("\\", "/").lower()
        if "antigomoku" in low:
            rules = "antigomoku"
        elif "gomoku" in low:
            rules = "gomoku"
        else:
            rules = "go"
    ckpts = list_checkpoints(models_dir)
    if not args.include_multi:
        skipped = [c for c in ckpts if c["players"] != 2]
        ckpts = [c for c in ckpts if c["players"] == 2]
        for c in skipped:
            print(f"skip {c['id']}: {c['players']}P (Elo board is 2-player; "
                  f"pass --include-multi to force)")
    agents = list(ckpts)
    if not args.no_random:
        agents.append({"id": RANDOM_ID, "path": None, "kind": "random",
                       "players": 2})
    if len(agents) < 2:
        print("need at least two agents (checkpoints and/or random)")
        sys.exit(1)
    if args.graphs.strip():
        graph_keys = [k.strip() for k in args.graphs.split(",") if k.strip()]
    else:
        graph_keys = default_graph_keys(rules)
    for k in graph_keys:
        if k not in builtin_graphs():
            print(f"unknown graph {k}")
            sys.exit(1)

    print(f"Elo pool: {[a['id'] for a in agents]}")
    print(f"rules={rules} graphs={graph_keys} sim={args.sim} "
          f"games/pair/graph={args.games}")
    table = rate_pool(agents, graph_keys, args.games, args.sim, args.device,
                      args.k, SCALE, args.seed, rules=rules,
                      win_length=args.win_length)
    out = os.path.abspath(args.out) if args.out else os.path.join(
        models_dir, ELO_JSON)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    ranked = sorted(table["players"].items(),
                    key=lambda kv: -kv[1]["elo"])
    print("---- Elo (display) ----")
    for name, p in ranked:
        print(f"  {p['elo']:7.1f}  {name}  ({p.get('kind')} · "
              f"{p.get('players')}P, {p['games']} games)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
