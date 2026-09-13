"""HTTP backend for index.html: graph data and the play engine.

Usage (repo root or ``scr/``; this script adds ``scr/`` to ``sys.path``):
    python web_ui/server.py [--port 8765] [--device cuda]

A game is 2/3/4 players (Gomoku / Anti-Gomoku: two). Each seat is human or a model.
Unmounted model seats play uniformly at random. Humans click the board;
models use MCTS. Graph-Go subtracts per-player komi from territory;
k-in-a-row scores win/draw/loss with no pass.
"""
from __future__ import annotations

import os
import sys
import json
import math
import random
import argparse
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np

# Sibling modules (`graphs`, `gkt_cpp`) live in parent ``scr/``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs import builtin_graphs, get_builtin, is_k_in_row_rules, normalize_rules      # noqa: E402
from gkt_cpp import require_native, py_graph_to_native  # noqa: E402
from gkt import DIRICHLET_FRAC  # noqa: E402
from grid_sym import SearchAugNet  # noqa: E402

# Layout hints for the frontend. type:
#   grid       — rectangular grid (no wrap)
#   torus      — rectangular wrap lattice, pannable (graph 3)
#   triangular — triangular diamond (graph 5)
#   tri_torus  — triangular wrap lattice, pannable (graph 5.5)
#   snake      — path laid out as an M (graph 7)
#   robinson   — 3D shell projected to a sphere, then Robinson (graphs 4 / 4.5)
#   cubic      — 19^3 lattice, rotatable three-plane slice (graph 6)
# Missing key (random) → force layout in the frontend.
LAYOUT = {
    "0": {"type": "grid", "m": 19, "n": 19},
    "0.5": {"type": "grid", "m": 7, "n": 7, "scale": 2.0},
    "1": {"type": "grid", "m": 27, "n": 13, "rotate": True, "scale": 1.25},
    "2": {"type": "grid", "m": 61, "n": 61},
    "7": {"type": "snake", "m": 1, "n": 19, "rotate": True, "scale": 2.0},
    "3": {"type": "torus", "m": 19, "n": 19},
    "4": {"type": "robinson", "scale": 1.25},
    "5": {"type": "triangular", "m": 19, "n": 19, "rotate": True, "scale": 1.25},
    "5.5": {"type": "tri_torus", "m": 19, "n": 19, "rotate": True, "scale": 1.25},
    "6": {"type": "cubic", "n": 19},
    "4.5": {"type": "robinson", "scale": 1.25},
    "G9": {"type": "grid", "m": 9, "n": 9},
    "G15": {"type": "grid", "m": 15, "n": 15},
    "G7d": {"type": "grid", "m": 7, "n": 7},
    "G9d": {"type": "grid", "m": 9, "n": 9},
}

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "_models")
# Trained checkpoints live in repo models/; the UI loads by filename so the
# browser does not POST a whole .pt.
REPO_MODELS = os.path.abspath(os.path.join(HERE, "..", "..", "models"))
SELFPLAY_BATCH = 64  # MCTS leaf batch; attention n×n caps VRAM around this

# Session state (single-thread HTTPServer; no lock)
STATE = {
    "key": None,
    "graph": None,
    "game": None,
    "device": "cuda",
    # players: [{komi, is_human, sim, net, model_name}] in move order (index = side-1)
    "players": [],
    "last_move": None,
    "last_captured": [],
    "history": [],            # [{move_no, player, vertex, captured}]
    "rules": "go",
    "win_length": 5,
}

# Mounted nets, 0-based player index; bound onto model seats at setup
MODELS = {}


# ---------------------------------------------------------------------------
# Graph payloads
# ---------------------------------------------------------------------------

def _rect_grid(g):
    grid = getattr(g, "grid", None)
    if not grid or len(grid) < 2:
        return None
    rows, cols = int(grid[0]), int(grid[1])
    if rows * cols != g.n:
        return None
    return [rows, cols, bool(grid[2]) if len(grid) > 2 else False]


def _graph_list():
    out = []
    for key, info in builtin_graphs().items():
        g = info["graph"]
        grid = _rect_grid(g)
        out.append({"key": key, "label": info["label"], "n": g.n,
                    "spec": info["spec"],
                    "grid": grid,
                    "gomoku": grid is not None})
    return out


def _graph_data(key):
    g = get_builtin(key)
    info = builtin_graphs()[key]
    edges = [[g.index_of(u), g.index_of(v)]
             for u in g.vertices for v in g.out_adj[u]]
    lo = LAYOUT.get(key)
    return {
        "key": key,
        "label": info.get("label", key),
        "spec": info.get("spec", ""),
        "n": g.n,
        "layout": lo if lo else None,
        "vertices": list(g.vertices),
        "edges": edges,
        "coords3d": _coords3d(key, g),
    }


def _coords3d(key, g):
    """Unit sphere vectors [x,y,z] per vertex for graphs 4 / 4.5 (Robinson).

    Graph 4.5 (9^3 cubic shell): integer coords centered at (4,4,4), then
    radial normalize. Graph 4 (19x19 lat/lon + poles): lat/lon to unit
    vectors; P0/P1 at the poles.
    """
    if key == "4.5":
        vecs = []
        for v in g.vertices:
            num = int(v[1:])           # strip leading 'A'
            i = num // 81
            j = (num % 81) // 9
            k = num % 9
            x, y, z = i - 4, j - 4, k - 4
            r = math.sqrt(x * x + y * y + z * z)
            vecs.append([x / r, y / r, z / r])
        return vecs
    if key == "4":
        m = n = 19
        max_lat = 78.0
        vecs = []
        for idx, _v in enumerate(g.vertices):
            if idx < m * n:
                i = idx // n
                j = idx % n
                lat = math.radians(max_lat - (i + 0.5) * (2 * max_lat) / m)
                lon = math.radians(j * 360.0 / n)
                vecs.append([math.cos(lat) * math.cos(lon),
                             math.cos(lat) * math.sin(lon),
                             math.sin(lat)])
            elif idx == m * n:
                vecs.append([0.0, 0.0, 1.0])    # P0 north pole
            else:
                vecs.append([0.0, 0.0, -1.0])   # P1 south pole
        return vecs
    return None


# ---------------------------------------------------------------------------
# Play / MCTS
# ---------------------------------------------------------------------------

def _model_roots():
    return [os.path.abspath(REPO_MODELS)]


def _peek_model_tags(path):
    """Architecture label and player-count tag (both required on the file)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        from gkt_cpu import peek_cpu_tags
        return peek_cpu_tags(path)
    if ext in (".pt", ".pth"):
        from gkt_gpu import peek_gpu_tags
        return peek_gpu_tags(path)
    raise ValueError(f"unsupported model file: {os.path.basename(path)}")


def _peek_model_kind(path):
    lab, k = _peek_model_tags(path)
    return f"{lab} · {k}P"


def _elo_by_name():
    """Optional display ratings from models/elo.json (gkt_elo.py)."""
    path = os.path.join(os.path.abspath(REPO_MODELS), "elo.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    players = data.get("players") or {}
    out = {}
    for name, info in players.items():
        if name == "__random__":
            continue
        if isinstance(info, dict) and "elo" in info:
            out[name] = float(info["elo"])
    return out


def _list_repo_models():
    """Checkpoints in repo models/ (not upload leftovers named player_N)."""
    out = []
    root = os.path.abspath(REPO_MODELS)
    if not os.path.isdir(root):
        return out
    elo = _elo_by_name()
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in sorted(names):
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".pt", ".pth", ".npz"):
            continue
        if name.lower().startswith("player_"):
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                kind, npl = _peek_model_tags(path)
            except Exception:  # noqa: BLE001
                continue
            out.append({"name": name, "bytes": os.path.getsize(path),
                        "kind": kind, "players": npl, "elo": elo.get(name)})
    return out


def _loaded_public():
    return {str(i): {"name": m["name"], "kind": m["kind"],
                     "players": int(m["players"])}
            for i, m in MODELS.items()}


def _bind_net(idx, net, name, kind, players=2):
    MODELS[idx] = {"net": net, "name": name, "kind": kind,
                   "players": int(players)}


def _preload_default(spec):
    """Load one warehouse checkpoint into every player slot (shared net)."""
    if not spec:
        listed = _list_repo_models()
        names = [m["name"] for m in listed]
        spec = next((n for n in ("new.pt", "new.npz", "best.pt", "best.npz")
                     if n in names),
                    names[0] if names else None)
    if not spec:
        return None
    if os.path.isfile(spec):
        path = os.path.abspath(spec)
        name = os.path.basename(path)
    else:
        path = _resolve_model_name(os.path.basename(spec))
        name = os.path.basename(path)
    net, kind, npl = _load_net(path)
    for i in range(4):
        _bind_net(i, net, name, kind, npl)
    return name, kind


def _resolve_model_name(name):
    """Basename only, and the path must stay under models/ or _models/."""
    base = os.path.basename(name or "")
    if not base or base in (".", ".."):
        raise ValueError("illegal model filename")
    if os.sep in (name or "") or "/" in (name or "") or "\\" in (name or ""):
        raise ValueError("illegal model filename")
    for root in _model_roots():
        path = os.path.abspath(os.path.join(root, base))
        if os.path.commonpath([path, root]) != root:
            continue
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"model not found: {base}")


def _load_torch_net(path):
    from gkt_gpu import load_net, gpu_net_label

    net = load_net(path, device=STATE["device"], graph=None)
    return net, gpu_net_label(net.net_type), int(net.num_players)


def _load_net(path):
    """Load from the file's own type tag. Returns (net, kind, num_players)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        from gkt_cpu import load_cpu_net, cpu_net_label
        net = load_cpu_net(path)
        return net, cpu_net_label(net), int(net.num_players)
    if ext in (".pt", ".pth"):
        return _load_torch_net(path)
    raise ValueError(f"unsupported model file: {os.path.basename(path)}")


def _new_game(key, players, rules="go", win_length=5):
    g = get_builtin(key)
    rules = normalize_rules(rules)
    if rules not in ("go", "gomoku", "antigomoku"):
        rules = "go"
    if is_k_in_row_rules(rules):
        label = "反五子棋" if rules == "antigomoku" else "五子棋"
        if getattr(g, "grid", None) is None:
            raise ValueError(f"{label}需要矩形网格（例如 0 / 0.5 / G9）")
        if len(players) != 2:
            raise ValueError(f"{label}只支持两人")
        for p in players:
            p["komi"] = 0.0
    STATE["key"] = key
    STATE["graph"] = g
    STATE["rules"] = rules
    STATE["win_length"] = int(win_length)
    STATE["players"] = []
    for i, p in enumerate(players):
        is_human = bool(p.get("is_human", False))
        entry = {
            "komi": float(p.get("komi", 0.0)),
            "is_human": is_human,
            "sim": int(p.get("sim", 200)) if not is_human else 0,
            "net": None,
            "model_name": None,
            "model_kind": None,
        }
        if not is_human:
            m = MODELS.get(i)
            if m is not None:
                net = m.get("net")
                npl = int(net.num_players) if net is not None else int(m["players"])
                k = len(players)
                if net is not None and npl != k:
                    raise ValueError(
                        f"checkpoint is {int(npl)}-player; this game has {k} players.")
                entry["net"] = net
                entry["model_name"] = m["name"]
                entry["model_kind"] = m["kind"]
        STATE["players"].append(entry)
    native = require_native()
    ng = py_graph_to_native(g)
    STATE["native"] = native
    STATE["ngraph"] = ng
    STATE["game"] = native.Game(
        ng, len(STATE["players"]),
        komi_schedule=[-p["komi"] for p in STATE["players"]],
        rules=STATE["rules"], win_length=STATE["win_length"])
    STATE["last_move"] = None
    STATE["last_captured"] = []
    STATE["history"] = []
    for p in STATE["players"]:
        if p["net"] is None:
            continue
        if hasattr(p["net"], "set_graph"):
            try:
                p["net"].set_graph(g)
            except ValueError as e:
                raise ValueError(
                    f"model {p.get('model_kind') or ''} ({p.get('model_name')}) "
                    f"cannot be used on graph {key}: {e}"
                ) from e
        nt = getattr(p["net"], "net_type", None)
        if nt in ("gnn", "2dcnn"):
            from gkt_gpu import maybe_script_infer
            p["net"] = maybe_script_infer(p["net"], g, STATE["device"])


def _f2(xs):
    return [round(float(x), 2) for x in xs]


def _f6(xs):
    return [round(float(x), 6) for x in xs]


def _net_ownership(net, X):
    """Per-vertex P(own) in [0, 1] from tanh occupancy (None if unavailable)."""
    src = getattr(net, "_src", net)
    X = np.asarray(X, dtype=np.float32)
    nt = getattr(src, "net_type", None)
    if nt in ("gnn", "2dcnn"):
        import torch
        x = torch.from_numpy(X).float().unsqueeze(0).to(src.device)
        src.eval()
        with torch.no_grad():
            own = src.forward(x)["own"][0].detach().float().cpu().numpy()
        return _f2((np.asarray(own, dtype=np.float32) + 1.0) * 0.5)
    if hasattr(src, "own_map"):
        own = src.own_map(X)
        return _f2((np.asarray(own, dtype=np.float32) + 1.0) * 0.5)
    return None


def _model_analysis(legal, counts):
    """Visit policy + net ownership for the position *before* the AI move."""
    n = len(STATE["graph"].vertices)
    acts = np.asarray(legal, dtype=np.int64).reshape(-1)
    vis = np.asarray(counts, dtype=np.float64).reshape(-1)
    k = int(min(acts.size, vis.size))
    tot = float(np.sum(vis[:k])) if k else 0.0
    if tot <= 0.0:
        tot = 1.0
    policy = np.zeros(n, dtype=np.float64)
    pass_p = 0.0
    for i in range(k):
        frac = float(vis[i]) / tot
        a = int(acts[i])
        if a == n:
            pass_p = frac
        elif 0 <= a < n:
            policy[a] = frac
    out = {"policy": _f6(policy), "pass_p": round(float(pass_p), 6)}
    game = STATE["game"]
    player = game.position.to_move
    net = STATE["players"][player - 1].get("net")
    if net is None:
        return out
    try:
        X = STATE["native"].extract_features(game.position, int(player))
        own = _net_ownership(net, X)
        if own is not None and len(own) == n:
            out["own"] = own
    except Exception:  # noqa: BLE001
        pass
    return out


def _apply_move(action, player, analysis=None):
    """Play one action (vertex index or None=pass) for `player`. dict or None."""
    game = STATE["game"]
    n = len(STATE["graph"].vertices)
    if action is None:
        action = n
    r = game.play(int(action))
    if not r.legal:
        return None
    vertex = None if int(action) == n else int(action)
    captured = list(r.captured)
    STATE["last_move"] = vertex
    STATE["last_captured"] = captured
    rec = {"move_no": len(STATE["history"]) + 1,
           "player": player, "vertex": vertex, "captured": captured}
    if analysis:
        rec.update(analysis)
    STATE["history"].append(rec)
    return {"player": player, "vertex": vertex, "captured": captured,
            "game_over": game.game_over()}


def _model_move():
    """One MCTS (or random) move for the side to move. dict or None."""
    game = STATE["game"]
    if game.game_over():
        return None
    player = game.position.to_move
    p = STATE["players"][player - 1]
    n = len(STATE["graph"].vertices)
    if p["net"] is not None:
        n_sim = max(1, int(p["sim"] or 0))
        native = STATE["native"]
        wrapped = SearchAugNet(p["net"], STATE["graph"])
        wrapped.begin_search()
        legal, counts, _ = native.search(
            game.position, wrapped, n_sim,
            min(SELFPLAY_BATCH, n_sim), DIRICHLET_FRAC, 1.5,
            random.getrandbits(32), False)
        if not legal:
            return None
        action = legal[int(np.argmax(counts))]
        analysis = _model_analysis(legal, counts)
    else:
        # uniform random over legal stones + pass (index n)
        legal = list(game.legal_moves())
        if not legal:
            return None
        action = random.choice(legal)
        counts = np.ones(len(legal), dtype=np.float64)
        analysis = _model_analysis(legal, counts)
    return _apply_move(action, player, analysis)


def _auto_advance():
    """Play model seats until a human to-move or game over. List of moves."""
    results = []
    game = STATE["game"]
    n = len(STATE["graph"].vertices)
    cap = n * 4 + 80
    while not game.game_over() and len(results) < cap:
        player = game.position.to_move
        if STATE["players"][player - 1]["is_human"]:
            break
        r = _model_move()
        if r is None:
            break
        results.append(r)
    return results


def _human_move(vertex):
    """Human stone (vertex=None is pass), then auto-play model seats.
    Returns (human_result, auto_results) or None if illegal / not a human turn."""
    game = STATE["game"]
    if game.game_over():
        return None
    player = game.position.to_move
    if not STATE["players"][player - 1]["is_human"]:
        return None
    r = _apply_move(vertex, player)
    if r is None:
        return None
    auto = _auto_advance()
    return r, auto


def _state():
    game = STATE["game"]
    if game is None:
        return {"ready": False}
    # Snapshot occupancy before focus-fill: finalize is scoring only. The
    # canvas must show the last real position, not the filled endgame board.
    occ = [int(x) for x in game.position.occupancy]
    scores = None
    if game.game_over():
        raw, _ = game.finalize()
        scores = {
            int(side): float(raw[side] + game.komi_schedule[k])
            for k, side in enumerate(game.sides)
        }
    players_out = [{
        "komi": p["komi"], "is_human": p["is_human"], "sim": p["sim"],
        "model_name": p["model_name"], "has_model": p["net"] is not None,
        "model_kind": p.get("model_kind"),
    } for p in STATE["players"]]
    return {
        "ready": True,
        "key": STATE["key"],
        "n": STATE["graph"].n,
        "num_players": len(STATE["players"]),
        "players": players_out,
        "occupancy": occ,
        "to_move": game.position.to_move,
        "move_no": game.position.move_no,
        "game_over": game.game_over(),
        "scores": scores,
        "pass_streak": list(game.position.pass_streak),
        "eliminated": [s >= 2 for s in game.position.pass_streak],
        "last_move": STATE["last_move"],
        "last_captured": STATE["last_captured"],
        "history": STATE["history"],
        "rules": STATE.get("rules", "go"),
        "win_length": int(STATE.get("win_length") or 5),
        "winner": int(getattr(game.position, "winner", 0) or 0),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        def _json_default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(type(o).__name__)
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _serve_html(self):
        p = os.path.join(HERE, "index.html")
        with open(p, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_upload(self):
        q = parse_qs(urlparse(self.path).query)
        idx = int(q.get("player", ["0"])[0])
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        name = unquote(self.headers.get("X-Filename", f"player_{idx}.pt"))
        ext = os.path.splitext(name)[1].lower() or ".pt"
        os.makedirs(MODEL_DIR, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=ext, dir=MODEL_DIR)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            net, kind, npl = _load_net(path)
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"无法加载模型：{e}"})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        _bind_net(idx, net, os.path.basename(name), kind, npl)
        return self._send(200, {"ok": True, "name": os.path.basename(name),
                                "kind": kind, "players": npl})

    def _handle_load_named(self, body):
        idx = int(body.get("player", 0))
        name = body.get("name") or ""
        if name in ("__random__", "random"):
            _bind_net(idx, None, "random", "random", 0)
            return self._send(200, {"ok": True, "name": "random", "kind": "random",
                                    "players": 0})
        try:
            path = _resolve_model_name(name)
            for j, m in MODELS.items():
                if m["name"] == os.path.basename(path) and j != idx:
                    _bind_net(idx, m["net"], m["name"], m["kind"], m["players"])
                    return self._send(200, {"ok": True, "name": m["name"],
                                            "kind": m["kind"],
                                            "players": m["players"]})
            net, kind, npl = _load_net(path)
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"无法加载模型：{e}"})
        _bind_net(idx, net, os.path.basename(path), kind, npl)
        return self._send(200, {"ok": True, "name": os.path.basename(path),
                                "kind": kind, "players": npl})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/models":
            return self._send(200, _list_repo_models())
        if path == "/api/loaded":
            return self._send(200, _loaded_public())
        if path == "/api/graphs":
            return self._send(200, _graph_list())
        if path == "/api/graph":
            q = parse_qs(urlparse(self.path).query)
            key = q.get("key", ["0.5"])[0]
            return self._send(200, _graph_data(key))
        if path == "/api/state":
            return self._send(200, _state())
        if path in ("/", "/index.html"):
            return self._serve_html()
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/upload_model":
            return self._handle_upload()
        body = self._read_body()
        if path == "/api/load_model":
            return self._handle_load_named(body)
        if path == "/api/setup":
            players = body.get("players", [])
            if len(players) < 2:
                return self._send(400, {"error": "need at least 2 players"})
            try:
                _new_game(body.get("key", "0.5"), players,
                          rules=body.get("rules", "go"),
                          win_length=int(body.get("win_length") or 5))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            # If any human is seated, play model seats until a human to-move.
            if any(p["is_human"] for p in STATE["players"]):
                _auto_advance()
            return self._send(200, {"state": _state()})
        if path == "/api/human":
            r = _human_move(body.get("vertex"))
            if r is None:
                return self._send(400, {"error": "illegal move"})
            human_result, auto = r
            return self._send(200, {"human": human_result, "auto": auto,
                                    "state": _state()})
        if path == "/api/advance":
            # One step if the current side is a model; otherwise no-op.
            moved = None
            if STATE["game"] is not None and not STATE["game"].game_over():
                player = STATE["game"].position.to_move
                if not STATE["players"][player - 1]["is_human"]:
                    moved = _model_move()
            return self._send(200, {"moved": moved, "state": _state()})
        return self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass  # quiet access log


def main():
    ap = argparse.ArgumentParser(description="GKT web play server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="new.pt",
                    help="warehouse checkpoint to preload (basename under models/ "
                         "or a path). Empty string skips preload.")
    args = ap.parse_args()

    if args.device in ("cuda", "auto"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA 不可用，回退到 CPU。")
                args.device = "cpu"
        except Exception:
            args.device = "cpu"
    STATE["device"] = args.device
    os.makedirs(MODEL_DIR, exist_ok=True)

    pre = None
    if args.model:
        try:
            pre = _preload_default(args.model)
        except Exception as e:  # noqa: BLE001
            print(f"preload failed ({args.model}): {e}")
            print("Setup 里另选 models/ 下的文件；未挂载席位走随机。")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"GKT web UI: http://127.0.0.1:{args.port}/ (device={args.device})")
    if pre:
        print(f"已预载 {pre[0]}（{pre[1]}）；开局不必再选文件")
    else:
        print(f"模型目录: {REPO_MODELS}（未预载；在设置里选文件）")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
