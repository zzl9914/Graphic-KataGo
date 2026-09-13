"""KataGo-style asynchronous distributed training (optional; idle until started).

Mirrors official KataGo's closed loop (SelfplayTraining.md / `katago contribute`):

  selfplay  →  shuffle  →  train  →  (optional gatekeeper)  →  accepted models
                 ↑                                              ↓
           contribute clients  ←—— HTTP serve ——←  latest accepted net

Processes only talk through a basedir (and optionally HTTP). They can run on
one machine or many. Single-machine training is `gkt_train_*.py`.

Usage (from ``scr/``):
  python gkt_dist.py init --basedir ../dist_run --from ../models/new.pt
  python gkt_dist.py selfplay --basedir ../dist_run
  python gkt_dist.py shuffle --basedir ../dist_run
  python gkt_dist.py train --basedir ../dist_run --device cuda
  python gkt_dist.py gate --basedir ../dist_run
  python gkt_dist.py serve --basedir ../dist_run --port 8877
  python gkt_dist.py contribute --url http://127.0.0.1:8877
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gkt import GktSelfPlay, stack_aux, aux_from_sample, feature_dim, pack_samples, unpack_samples  # noqa: E402
from graphs import (  # noqa: E402
    builtin_graphs, get_builtin, GOMOKU_KEYS, GOMOKU_TRAIN_KEYS,
    RULES_CHOICES, is_k_in_row_rules, normalize_rules,
)

EXCLUDE = {"2", "6"}  # oversized; 2 is a post-train grid generalization board
GRID_KEYS = ("0", "0.5", "1", "2", "3", "G9", "G15", "G7d", "G9d")  # 2 kept for 2DCNN inference; EXCLUDE drops it from training
GPU_NETS = ("gnn", "2dcnn")
CPU_NETS = ("mlp", "1dcnn")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_gpu(net_type: str) -> bool:
    return net_type in GPU_NETS


def _dirs(basedir: str) -> Dict[str, str]:
    d = {
        "root": basedir,
        "models": os.path.join(basedir, "models"),
        "rejected": os.path.join(basedir, "rejectedmodels"),
        "pending": os.path.join(basedir, "modelstobetested"),
        "selfplay": os.path.join(basedir, "selfplay"),
        "shuffled": os.path.join(basedir, "shuffled"),
        "logs": os.path.join(basedir, "logs"),
        "cache": os.path.join(basedir, "client_cache"),
    }
    return d


def _ensure_dirs(basedir: str) -> Dict[str, str]:
    d = _dirs(basedir)
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def run_path(basedir: str) -> str:
    return os.path.join(basedir, "run.json")


def load_run(basedir: str) -> Dict:
    path = run_path(basedir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run(basedir: str, run: Dict) -> None:
    path = run_path(basedir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _graphs_for(run: Dict) -> List[str]:
    keys = list(run.get("graphs") or [])
    if not is_k_in_row_rules(run.get("rules", "go")):
        keys = [k for k in keys if k not in EXCLUDE and k not in GOMOKU_KEYS]
    if run.get("net_type") == "2dcnn":
        keys = [k for k in keys if k in GRID_KEYS]
    return keys


def _max_moves(run: Dict, n: int) -> int:
    if is_k_in_row_rules(run.get("rules", "go")):
        return max(1, int(n))
    cap = int(n * float(run.get("max_move_factor", 2.0))) + int(run.get("min_moves", 40))
    return max(int(run.get("min_moves", 40)), cap)


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _list_npz(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, n) for n in os.listdir(folder)
            if n.endswith(".npz") and not n.endswith(".tmp")]


_SHUFFLED_NAME = re.compile(r"^(.+)_\d{8}_\d{6}_\d+\.npz$")


def _graph_from_shuffled_path(path: str) -> Optional[str]:
    m = _SHUFFLED_NAME.match(os.path.basename(path))
    if m:
        return m.group(1)
    try:
        g, _ = unpack_samples(path)
        return g
    except Exception:  # noqa: BLE001
        return None


def _latest_shuffled_by_graph(folder: str) -> Dict[str, str]:
    """Newest shuffled pack per graph (by mtime)."""
    files = _list_npz(folder)
    files.sort(key=os.path.getmtime)
    latest: Dict[str, str] = {}
    for p in files:
        g = _graph_from_shuffled_path(p)
        if g is not None:
            latest[g] = p
    return latest


def _unlink_quiet(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _prune_older_than(paths_mtime_asc: List[str], keep: int) -> int:
    """Delete all but the last `keep` paths. `paths` must be oldest-first."""
    if keep < 0:
        keep = 0
    n = 0
    drop = paths_mtime_asc if keep == 0 else paths_mtime_asc[:-keep]
    for p in drop:
        if _unlink_quiet(p):
            n += 1
    return n


def _list_models(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in (".pt", ".pth", ".npz") and not name.endswith(".tmp"):
            out.append(os.path.join(folder, name))
    out.sort(key=lambda p: os.path.getmtime(p))
    return out


def latest_model(folder: str) -> Optional[str]:
    files = _list_models(folder)
    return files[-1] if files else None


def _atomic_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_selfplay_file(folder: str, samples: List[Tuple], graph_key: str,
                        model_name: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{graph_key}_{model_name}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:6]}.npz"
    path = os.path.join(folder, name)
    _atomic_bytes(path, pack_samples(samples, graph_key, extra={"model": model_name}))
    return path


def load_net_for_run(path: str, run: Dict, graph, device: str):
    nt = run["net_type"]
    if _is_gpu(nt):
        from gkt_gpu import load_net
        net = load_net(path, device=device, graph=graph)
        return net
    from gkt_cpu import load_cpu_net
    return load_cpu_net(path)


def save_net_for_run(net, path: str, run: Dict) -> None:
    if _is_gpu(run["net_type"]):
        from gkt_gpu import save_net
        save_net(net, path)
        return
    from gkt_cpu import save_cpu_net
    save_cpu_net(net, path)


def _make_fresh_net(run: Dict, graph, device: str, lr: float):
    nt = run["net_type"]
    if nt in GPU_NETS:
        from gkt_gpu import make_net
        return make_net(nt, n_features=int(run["n_features"]),
                        hidden_dim=int(run["hidden"]), n_blocks=int(run["n_blocks"]),
                        graph=graph, device=device, lr=lr,
                        attn_layer=int(run["attn_layer"]),
                        n_heads=int(run["n_heads"]),
                        num_players=int(run["num_players"]))
    from gkt_cpu import MlpPolicyValueNet, Cnn1dPolicyValueNet
    F, H = int(run["n_features"]), int(run["hidden"])
    k = int(run["num_players"])
    if nt == "1dcnn":
        return Cnn1dPolicyValueNet(F, H, kernel_size=int(run.get("kernel_size", 3)),
                                   n_layers=int(run.get("conv_layers", 20)), lr=lr,
                                   num_players=k)
    return MlpPolicyValueNet(F, H, lr=lr, num_players=k)


def _selfplay_job(job: Dict) -> Tuple[str, List[Tuple]]:
    """Spawn-friendly: play games on one graph with a checkpoint file."""
    run = job["run"]
    graph_key = job["graph_key"]
    g = get_builtin(graph_key)
    net = load_net_for_run(job["model_path"], run, g, job["device"])
    if hasattr(net, "eval"):
        net.eval()
    if hasattr(net, "set_graph"):
        net.set_graph(g)
    if run.get("net_type") in GPU_NETS:
        from gkt_gpu import maybe_script_infer
        net = maybe_script_infer(net, g, job["device"])
    driver = GktSelfPlay(
        g, net, n_simulations=int(run["sim"]),
        max_moves=int(job.get("max_moves") or _max_moves(run, len(g.vertices))),
        batch_size=int(run.get("selfplay_batch", 64)),
        value_target=run.get("value_target", "mc"),
        q_lambda=float(run.get("q_lambda", 0.5)),
        log_fn=lambda *_: None,
        rules=str(run.get("rules", "go")),
        win_length=int(run.get("win_length", 5)),
    )
    samples: List[Tuple] = []
    for _ in range(int(job["n_games"])):
        samples.extend(driver.play_one_game())
    return graph_key, samples


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    basedir = os.path.abspath(args.basedir)
    d = _ensure_dirs(basedir)
    krow = is_k_in_row_rules(args.rules)
    if krow:
        if args.graphs:
            graphs = [k.strip() for k in args.graphs.split(",") if k.strip()]
        else:
            graphs = list(GOMOKU_TRAIN_KEYS)
    else:
        if args.graphs:
            graphs = [k.strip() for k in args.graphs.split(",") if k.strip()]
            graphs = [k for k in graphs if k not in EXCLUDE and k not in GOMOKU_KEYS]
        else:
            graphs = [k for k in builtin_graphs()
                      if k not in EXCLUDE and k not in GOMOKU_KEYS]
    nt = args.net
    if nt == "2dcnn":
        graphs = [k for k in graphs if k in GRID_KEYS]
        if not graphs:
            raise SystemExit(
                "2DCNN needs a rectangular .grid key "
                "(0, 0.5, 1, 3; Gomoku also G9/G15/G7d/G9d)")
    seed = args.src
    if not seed or not os.path.isfile(seed):
        raise SystemExit("init needs --from <existing .pt/.npz checkpoint>")
    seed_ext = os.path.splitext(seed)[1].lower()
    if _is_gpu(nt) and seed_ext not in (".pt", ".pth"):
        raise SystemExit(f"--net {nt} needs a .pt/.pth seed, got {seed}")
    if not _is_gpu(nt) and seed_ext != ".npz":
        raise SystemExit(f"--net {nt} needs a .npz seed, got {seed}")
    dest_ext = ".pt" if _is_gpu(nt) else ".npz"
    dest = os.path.join(d["models"], f"s0_d0{dest_ext}")
    if seed_ext in (".pt", ".pth"):
        import torch
        from gkt_gpu import tagged_num_players, gpu_net_type
        ckpt = torch.load(seed, map_location="cpu")
        if not isinstance(ckpt, dict) or "n_features" not in ckpt or "net_type" not in ckpt:
            raise SystemExit("seed is not a GKT GPU checkpoint")
        n_features = int(ckpt["n_features"])
        npl = tagged_num_players(ckpt)
        seed_nt = gpu_net_type(ckpt["net_type"])
    elif seed_ext == ".npz":
        from gkt_cpu import peek_cpu_tags, cpu_net_type
        _, npl = peek_cpu_tags(seed)
        data = np.load(seed, allow_pickle=False)
        try:
            if "_net_type" not in data.files:
                raise SystemExit("seed npz missing _net_type")
            seed_nt = cpu_net_type(np.asarray(data["_net_type"]).item())
            if "W_enc" in data.files:
                n_features = int(data["W_enc"].shape[0]) - 1
            else:
                n_features = int(data["W1"].shape[0]) - 1
        finally:
            data.close()
    else:
        raise SystemExit(f"unsupported seed {seed}")
    if seed_nt != nt:
        raise SystemExit(f"seed net_type={seed_nt} != --net {nt}")
    if n_features != feature_dim(npl):
        raise SystemExit(
            f"seed F={n_features} != feature_dim(num_players)={feature_dim(npl)}")
    if krow and npl != 2:
        raise SystemExit(f"{normalize_rules(args.rules)} requires a 2-player seed, got {npl}P")
    with open(seed, "rb") as f:
        _atomic_bytes(dest, f.read())
    run = {
        "net_type": nt,
        "graphs": graphs,
        "sim": int(args.sim),
        "selfplay_batch": int(args.selfplay_batch),
        "value_target": args.value_target,
        "q_lambda": float(args.q_lambda),
        "n_features": n_features,
        "num_players": npl,
        "hidden": int(args.hidden),
        "n_blocks": int(args.n_blocks),
        "attn_layer": int(args.attn_layer),
        "n_heads": int(args.attn_heads),
        "kernel_size": int(args.kernel_size),
        "conv_layers": int(args.conv_layers),
        "min_moves": int(args.min_moves),
        "max_move_factor": float(args.max_move_factor),
        "use_gating": bool(args.gate),
        "arena_games": int(args.arena_games),
        "arena_sim": int(args.arena_sim),
        "arena_threshold": float(args.arena_threshold),
        "token": args.token or "",
        "window_files": int(args.window_files),
        "shuffled_keep": int(args.shuffled_keep),
        "train_batch": int(args.train_batch),
        "steps_per_save": int(args.steps_per_save),
        "lr": float(args.lr),
        "rules": normalize_rules(args.rules),
        "win_length": int(args.win_length),
    }
    save_run(basedir, run)
    _log(f"initialized {basedir}")
    _log(f"  net={nt} rules={run['rules']} graphs={','.join(graphs)}")
    _log(f"  seed model → {dest}")
    _log("leave the loops unused until you start selfplay/shuffle/train[/serve]")


def cmd_selfplay(args) -> None:
    basedir = os.path.abspath(args.basedir)
    d = _ensure_dirs(basedir)
    run = load_run(basedir)
    graphs = _graphs_for(run)
    if not graphs:
        raise SystemExit("run.json has no usable graphs")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    n_workers = max(1, int(args.workers))
    device = args.device
    gpw = max(1, int(args.gpw))

    def one_cycle(executor) -> None:
        model = latest_model(d["models"])
        if model is None:
            _log("no accepted model yet; waiting")
            return
        model_name = os.path.splitext(os.path.basename(model))[0]
        jobs = []
        for i in range(n_workers):
            jobs.append({
                "run": run,
                "graph_key": graphs[(int(time.time()) + i) % len(graphs)],
                "model_path": model,
                "device": device,
                "n_games": gpw,
            })
        futs = [executor.submit(_selfplay_job, j) for j in jobs]
        n_files = 0
        n_samp = 0
        for fut in futs:
            graph_key, samples = fut.result()
            if not samples:
                continue
            write_selfplay_file(d["selfplay"], samples, graph_key, model_name)
            n_files += 1
            n_samp += len(samples)
        _log(f"selfplay {model_name}: wrote {n_files} files, {n_samp} samples")

    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    try:
        while True:
            one_cycle(executor)
            if args.once:
                break
            time.sleep(max(0.05, float(args.poll)))
    finally:
        executor.shutdown(wait=True)


def cmd_shuffle(args) -> None:
    basedir = os.path.abspath(args.basedir)
    d = _ensure_dirs(basedir)
    run = load_run(basedir)
    window = int(run.get("window_files", 32))
    keep_shuf = max(1, int(run.get("shuffled_keep", 2)))

    def once() -> None:
        files = _list_npz(d["selfplay"])
        files.sort(key=os.path.getmtime)
        by_graph: Dict[str, List[str]] = {}
        for p in files:
            try:
                g, _ = unpack_samples(p)
            except Exception as e:  # noqa: BLE001
                _log(f"skip bad selfplay {os.path.basename(p)}: {e}")
                continue
            by_graph.setdefault(g, []).append(p)
        n_out = 0
        for g, paths in by_graph.items():
            use = paths[-window:]
            samples: List[Tuple] = []
            for p in use:
                _, ss = unpack_samples(p)
                samples.extend(ss)
            if len(samples) < 2:
                continue
            rng = random.Random(time.time_ns())
            rng.shuffle(samples)
            name = f"{g}_{time.strftime('%Y%m%d_%H%M%S')}_{len(samples)}.npz"
            dest = os.path.join(d["shuffled"], name)
            _atomic_bytes(dest, pack_samples(samples, g))
            n_out += 1
            dropped_sp = _prune_older_than(paths, window)
            shuf_g = [p for p in _list_npz(d["shuffled"])
                      if _graph_from_shuffled_path(p) == g]
            shuf_g.sort(key=os.path.getmtime)
            dropped_sh = _prune_older_than(shuf_g, keep_shuf)
            _log(f"shuffle {g}: {len(use)} files → {len(samples)} rows → {name}"
                 f" (drop selfplay={dropped_sp} shuffled={dropped_sh})")
        if n_out == 0:
            _log("shuffle: nothing new")

    while True:
        once()
        if args.once:
            break
        time.sleep(max(1.0, float(args.poll)))


def _train_batch_gpu(net, batch) -> Tuple[float, float, float, float]:
    X = np.stack([s[0] for s in batch])
    mask = np.stack([s[1] for s in batch])
    pol = np.stack([s[2] for s in batch])
    z = np.array([s[4] for s in batch], dtype=np.float32)
    own = np.stack([s[5] for s in batch])
    pl, vl, ol, al, tot = net.train_on_batch(X, mask, pol, z, own, stack_aux(batch))
    return pl, vl, ol, tot


def _train_batch_cpu(net, batch) -> Tuple[float, float, float, float]:
    tot = pl = vl = ol = 0.0
    for sample in batch:
        X, mask, pol, _me, zz, own = sample[:6]
        a, b, c, t = net.backward(X, mask, pol, zz, own, aux=aux_from_sample(sample))
        pl += a
        vl += b
        ol += c
        tot += t
    n = max(len(batch), 1)
    return pl / n, vl / n, ol / n, tot / n


def cmd_train(args) -> None:
    basedir = os.path.abspath(args.basedir)
    d = _ensure_dirs(basedir)
    run = load_run(basedir)
    device = args.device
    batch_size = int(run.get("train_batch", 64))
    steps_per_save = int(run.get("steps_per_save", 50))
    lr = float(run.get("lr", 1e-3))
    model = latest_model(d["models"]) or latest_model(d["pending"])
    if model is None:
        raise SystemExit("no model to continue from")
    graphs = _graphs_for(run)
    g0 = get_builtin(graphs[0])
    net = load_net_for_run(model, run, g0, device)
    if hasattr(net, "optimizer"):
        for pg in net.optimizer.param_groups:
            pg["lr"] = lr
    elif hasattr(net, "lr"):
        net.lr = lr
    state_path = os.path.join(basedir, "train_state.json")
    state = {"steps": 0, "rows": 0}
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            state.update(json.load(f))

    buffer: List[Tuple] = []
    buf_graph: Optional[str] = None
    last_graph: Optional[str] = None

    def pull_shuffled() -> None:
        nonlocal buffer, buf_graph, last_graph
        latest = _latest_shuffled_by_graph(d["shuffled"])
        if not latest:
            buffer = []
            buf_graph = None
            return
        keys = list(latest)
        if last_graph in keys and len(keys) > 1:
            keys = [k for k in keys if k != last_graph]
        gkey = random.choice(keys)
        path = latest[gkey]
        gkey, samples = unpack_samples(path)
        if buf_graph == gkey and buffer:
            buffer = buffer + samples
        else:
            buffer = samples
        buf_graph = last_graph = gkey
        g = get_builtin(gkey)
        if hasattr(net, "set_graph"):
            net.set_graph(g)

    def save_pending() -> None:
        ext = ".pt" if _is_gpu(run["net_type"]) else ".npz"
        name = f"s{state['steps']}_d{state['rows']}{ext}"
        dest = os.path.join(d["pending"] if run.get("use_gating") else d["models"], name)
        save_net_for_run(net, dest, run)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        _log(f"saved {dest}")

    gpu = _is_gpu(run["net_type"])
    while True:
        if len(buffer) < batch_size:
            pull_shuffled()
        if len(buffer) < batch_size:
            _log("train: waiting for shuffled data")
            if args.once:
                break
            time.sleep(max(1.0, float(args.poll)))
            continue
        step_losses = []
        for _ in range(steps_per_save):
            if len(buffer) < batch_size:
                break
            idx = random.sample(range(len(buffer)), batch_size)
            batch = [buffer[i] for i in idx]
            if gpu:
                losses = _train_batch_gpu(net, batch)
            else:
                losses = _train_batch_cpu(net, batch)
            if all(np.isfinite(x) for x in losses):
                drop = set(idx)
                buffer = [s for i, s in enumerate(buffer) if i not in drop]
                step_losses.append(losses)
                state["steps"] += 1
                state["rows"] += batch_size
        if step_losses:
            avg = np.mean(step_losses, axis=0)
            _log(f"train graph={buf_graph} steps={state['steps']} "
                 f"ploss={avg[0]:.4f} vloss={avg[1]:.4f} oloss={avg[2]:.4f} "
                 f"total={avg[3]:.4f}")
            save_pending()
        if args.once:
            break
        time.sleep(max(0.05, float(args.poll)))


def cmd_gate(args) -> None:
    basedir = os.path.abspath(args.basedir)
    d = _ensure_dirs(basedir)
    run = load_run(basedir)
    if not run.get("use_gating"):
        _log("gating disabled in run.json; copying pending → models")
    device = args.device

    def once() -> None:
        pending = _list_models(d["pending"])
        accepted = latest_model(d["models"])
        if not pending:
            _log("gate: no pending models")
            return
        new_path = pending[0]
        if accepted is None or not run.get("use_gating"):
            dest = os.path.join(d["models"], os.path.basename(new_path))
            os.replace(new_path, dest)
            _log(f"gate: accepted {os.path.basename(dest)} (no baseline / gating off)")
            return
        graphs = _graphs_for(run)[: max(1, int(args.arena_graphs))]
        arena_progress = os.path.join(basedir, "progress.arena.txt")
        try:
            open(arena_progress, "w", encoding="utf-8").close()
        except OSError:
            pass
        n_games = int(run["arena_games"])
        n_sim = int(run.get("arena_sim", run["sim"]))
        rules = str(run.get("rules", "go"))
        win_length = int(run.get("win_length", 5))
        threshold = float(run.get("arena_threshold", 0.0))
        total_lead, total_games = 0.0, 0
        n_arena = len(graphs)
        _log(f"gate: starting — 1 opponent x {n_arena} graphs x {n_games} games")
        if _is_gpu(run["net_type"]):
            from gkt_gpu import load_net
            from gkt_train_gpu import arena_match
            new_net = load_net(new_path, device=device, graph=get_builtin(graphs[0]))
            old_net = load_net(accepted, device=device, graph=get_builtin(graphs[0]))
            nw, ow = new_net.get_weights(), old_net.get_weights()
            for i, key in enumerate(graphs, 1):
                g = get_builtin(key)
                amax = _max_moves(run, len(g.vertices))
                lead, ng = arena_match(
                    nw, ow, g, n_games, n_sim,
                    amax, new_net.n_features, new_net.hidden_dim, new_net.n_blocks,
                    getattr(new_net, "attn_layer", run.get("attn_layer", 8)),
                    getattr(new_net, "n_heads", run.get("n_heads", 4)),
                    device, new_net.net_type,
                    new_net.num_players, rules, win_length,
                    progress_file=arena_progress, label="accepted",
                    tag=f"gate g{key}")
                total_lead += lead
                total_games += ng
                _log(f"gate: [{i}/{n_arena}] graph {key} vs accepted: "
                     f"lead {lead / max(1, ng):+.4f} ({ng} games)")
        else:
            from gkt_cpu import load_cpu_net
            from gkt_train_cpu import arena_match
            new_net = load_cpu_net(new_path)
            old_net = load_cpu_net(accepted)
            nw, ow = new_net.state_dict(), old_net.state_dict()
            ks = int(run.get("kernel_size", 3))
            cl = int(run.get("conv_layers", 20))
            for i, key in enumerate(graphs, 1):
                g = get_builtin(key)
                amax = _max_moves(run, len(g.vertices))
                lead, ng = arena_match(
                    nw, ow, g, n_games, n_sim,
                    amax, int(run["n_features"]), int(run["hidden"]),
                    run["net_type"], ks, cl,
                    int(run.get("num_players", 2)), rules, win_length,
                    progress_file=arena_progress, label="accepted",
                    tag=f"gate g{key}")
                total_lead += lead
                total_games += ng
                _log(f"gate: [{i}/{n_arena}] graph {key} vs accepted: "
                     f"lead {lead / max(1, ng):+.4f} ({ng} games)")
        avg_lead = total_lead / max(total_games, 1)
        if avg_lead >= threshold:
            dest = os.path.join(d["models"], os.path.basename(new_path))
            os.replace(new_path, dest)
            _log(f"gate: ACCEPTED {os.path.basename(dest)} "
                 f"avg lead {avg_lead:+.4f} ({total_games} games)")
        else:
            dest = os.path.join(d["rejected"], os.path.basename(new_path))
            os.replace(new_path, dest)
            _log(f"gate: REJECTED {os.path.basename(dest)} "
                 f"avg lead {avg_lead:+.4f} ({total_games} games)")

    while True:
        once()
        if args.once:
            break
        time.sleep(max(1.0, float(args.poll)))


class _DistHandler(BaseHTTPRequestHandler):
    basedir = ""
    token = ""

    def log_message(self, fmt, *args):
        _log("http " + (fmt % args))

    def _ok_auth(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("X-Token", "") == self.token

    def _send(self, code: int, obj, ctype="application/json"):
        if isinstance(obj, (dict, list)):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            ctype = "application/json"
        else:
            raw = obj
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self._ok_auth():
            return self._send(401, {"error": "bad token"})
        d = _dirs(self.basedir)
        path = urlparse(self.path).path
        run = load_run(self.basedir)
        if path in ("/", "/api/status"):
            model = latest_model(d["models"])
            return self._send(200, {
                "run": {k: run[k] for k in run if k != "token"},
                "model": os.path.basename(model) if model else None,
                "sha": _file_sha(model) if model else None,
                "graphs": _graphs_for(run),
            })
        if path == "/api/task":
            model = latest_model(d["models"])
            if model is None:
                return self._send(503, {"error": "no accepted model"})
            graphs = _graphs_for(run)
            gkey = random.choice(graphs)
            g = get_builtin(gkey)
            return self._send(200, {
                "model": os.path.basename(model),
                "sha": _file_sha(model),
                "graph": gkey,
                "sim": run["sim"],
                "max_moves": _max_moves(run, len(g.vertices)),
                "net_type": run["net_type"],
                "n_games": 1,
                "rules": run.get("rules", "go"),
                "win_length": int(run.get("win_length", 5)),
            })
        if path.startswith("/api/model/"):
            name = os.path.basename(unquote(path[len("/api/model/"):]))
            fpath = os.path.join(d["models"], name)
            if not os.path.isfile(fpath):
                return self._send(404, {"error": "model not found"})
            with open(fpath, "rb") as f:
                body = f.read()
            ctype = "application/octet-stream"
            return self._send(200, body, ctype=ctype)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._ok_auth():
            return self._send(401, {"error": "bad token"})
        path = urlparse(self.path).path
        if path != "/api/games":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        graph = self.headers.get("X-Graph", "")
        model_name = self.headers.get("X-Model", "unknown")
        try:
            gkey, samples = unpack_samples(body)
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"bad npz: {e}"})
        if graph and graph != gkey:
            return self._send(400, {"error": "graph mismatch"})
        d = _dirs(self.basedir)
        os.makedirs(d["selfplay"], exist_ok=True)
        path_out = write_selfplay_file(d["selfplay"], samples, gkey, model_name)
        return self._send(200, {"ok": True, "file": os.path.basename(path_out),
                                "n": len(samples)})


def cmd_serve(args) -> None:
    basedir = os.path.abspath(args.basedir)
    _ensure_dirs(basedir)
    run = load_run(basedir)
    _DistHandler.basedir = basedir
    _DistHandler.token = args.token if args.token is not None else run.get("token", "")
    host = args.host
    httpd = HTTPServer((host, int(args.port)), _DistHandler)
    _log(f"dist server {host}:{args.port} basedir={basedir} "
         f"(bind 127.0.0.1 unless you pass --host)")
    httpd.serve_forever()


def cmd_contribute(args) -> None:
    import urllib.request

    url = args.url.rstrip("/")
    token = args.token or ""
    device = args.device
    cache = os.path.abspath(args.cache or os.path.join(".", "_dist_cache"))
    os.makedirs(cache, exist_ok=True)

    def req(path, data=None, headers=None, method=None):
        h = dict(headers or {})
        if token:
            h["X-Token"] = token
        r = urllib.request.Request(url + path, data=data, headers=h, method=method)
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.read(), resp.headers

    n_done = 0
    while True:
        raw, _ = req("/api/task")
        task = json.loads(raw.decode("utf-8"))
        model_name = task["model"]
        local = os.path.join(cache, model_name)
        if not os.path.isfile(local) or _file_sha(local) != task["sha"]:
            body, _ = req("/api/model/" + model_name)
            _atomic_bytes(local, body)
            _log(f"downloaded {model_name}")
        run = {
            "net_type": task["net_type"],
            "sim": task["sim"],
            "selfplay_batch": int(args.selfplay_batch),
            "value_target": args.value_target,
            "q_lambda": float(args.q_lambda),
            "rules": task.get("rules", "go"),
            "win_length": int(task.get("win_length", 5)),
        }
        job = {
            "run": run,
            "graph_key": task["graph"],
            "model_path": local,
            "device": device,
            "n_games": int(task.get("n_games", 1)),
            "max_moves": int(task["max_moves"]),
        }
        graph_key, samples = _selfplay_job(job)
        blob = pack_samples(samples, graph_key, extra={"model": model_name})
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Graph": graph_key,
            "X-Model": os.path.splitext(model_name)[0],
        }
        raw, _ = req("/api/games", data=blob, headers=headers, method="POST")
        n_done += 1
        _log(f"uploaded {json.loads(raw.decode())} ({n_done} games this process)")
        if args.once:
            break


def _add_loop_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--basedir", default="../dist_run")
    p.add_argument("--poll", type=float, default=5.0,
                   help="seconds between idle polls")
    p.add_argument("--once", action="store_true",
                   help="run a single cycle then exit")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KataGo-style distributed loop (optional; does nothing until started)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create basedir + run.json + seed accepted model")
    p.add_argument("--basedir", default="../dist_run")
    p.add_argument("--from", dest="src", required=True,
                   help="seed checkpoint (.pt or .npz)")
    p.add_argument("--net", default="gnn", choices=list(GPU_NETS + CPU_NETS))
    p.add_argument("--graphs", default="")
    p.add_argument("--rules", default="go", choices=list(RULES_CHOICES))
    p.add_argument("--win-length", type=int, default=5,
                   help="Gomoku k-in-a-row (ignored for Graph-Go)")
    p.add_argument("--sim", type=int, default=800)
    p.add_argument("--selfplay-batch", type=int, default=64)
    p.add_argument("--value-target", default="mc")
    p.add_argument("--q-lambda", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--attn-layer", type=int, default=8,
                   help="GNN only: insert one global self-attention layer "
                        "before block #N (2DCNN ignores this)")
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--conv-layers", type=int, default=20)
    p.add_argument("--min-moves", type=int, default=40)
    p.add_argument("--max-move-factor", type=float, default=2.0)
    p.add_argument("--gate", action="store_true", help="enable gatekeeper (off = accept every save)")
    p.add_argument("--arena-games", type=int, default=2)
    p.add_argument("--arena-sim", type=int, default=50)
    p.add_argument("--arena-threshold", type=float, default=0.0,
                   help="min average normalized score-lead to accept "
                        "(same units as the self-play trainers; 0.0 = not weaker)")
    p.add_argument("--token", default="")
    p.add_argument("--window-files", type=int, default=32)
    p.add_argument("--shuffled-keep", type=int, default=2,
                   help="shuffled packs to keep per graph (train reads the newest)")
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--steps-per-save", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("selfplay", help="poll accepted models, write selfplay/*.npz")
    _add_loop_flags(p)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpw", type=int, default=16)
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("shuffle", help="window + shuffle selfplay → shuffled/")
    _add_loop_flags(p)

    p = sub.add_parser("train", help="SGD on shuffled npz, write pending or models")
    _add_loop_flags(p)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("gate", help="Arena new vs accepted (KataGo gatekeeper)")
    _add_loop_flags(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--arena-graphs", type=int, default=3)

    p = sub.add_parser("serve", help="HTTP for contribute clients (default bind localhost)")
    p.add_argument("--basedir", default="../dist_run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8877)
    p.add_argument("--token", default=None)

    p = sub.add_parser("contribute", help="like katago contribute: fetch net, play, upload")
    p.add_argument("--url", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", default="")
    p.add_argument("--selfplay-batch", type=int, default=64)
    p.add_argument("--value-target", default="mc")
    p.add_argument("--q-lambda", type=float, default=0.5)
    p.add_argument("--once", action="store_true")

    args = ap.parse_args()
    cmds = {
        "init": cmd_init,
        "selfplay": cmd_selfplay,
        "shuffle": cmd_shuffle,
        "train": cmd_train,
        "gate": cmd_gate,
        "serve": cmd_serve,
        "contribute": cmd_contribute,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
