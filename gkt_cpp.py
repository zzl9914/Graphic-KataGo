"""C++ engine / MCTS facade. Native module is required (no silent Python fallback)."""
from __future__ import annotations

import os
import sys

_SCR = os.path.dirname(os.path.abspath(__file__))
_CPP = os.path.abspath(os.path.join(_SCR, "..", "cpp"))
_BUILD = os.path.join(_CPP, "build")
for p in (_BUILD, os.path.join(_BUILD, "Debug"), os.path.join(_BUILD, "Release"),
          _CPP, _SCR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def require_native():
    try:
        import gkt_native as native
    except ImportError as e:
        raise ImportError(
            "gkt_native is required. From the repo root run: "
            "python cpp/_build_with_sdk.py && python cpp/_deploy_pyd.py"
        ) from e
    return native


def py_graph_to_native(graph):
    native = require_native()
    cached = getattr(graph, "_gkt_native", None)
    if cached is not None:
        return cached
    edges = []
    idx = graph._idx
    for u in graph.vertices:
        ui = idx[u]
        for v in graph.out_adj[u]:
            edges.append((ui, idx[v]))
    grid = tuple(graph.grid) if getattr(graph, "grid", None) else None
    ng = native.Graph(len(graph.vertices), edges, grid)
    graph._gkt_native = ng
    return ng


def play_one_game(graph, net, n_simulations=800, temperature=1.0, max_moves=None,
                  batch_size=32, q_lambda=0.5,
                  randomize_sim=True, seed=0, heartbeat=None,
                  rules="go", win_length=5):
    native = require_native()
    ng = py_graph_to_native(graph)
    k = int(getattr(net, "num_players", 2))
    mm = 0 if max_moves is None else int(max_moves)
    return list(native.play_one_game(
        ng, net, k,
        n_simulations=int(n_simulations),
        temperature=float(temperature),
        max_moves=mm,
        batch_size=int(batch_size),
        q_lambda=float(q_lambda),
        randomize_sim=bool(randomize_sim),
        seed=int(seed),
        heartbeat=heartbeat,
        rules=str(rules),
        win_length=int(win_length)))
