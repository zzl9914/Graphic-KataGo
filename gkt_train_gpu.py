"""
Cross-graph training for Graph-Go, Gomoku, or Anti-Gomoku.

Trains ONE graph-agnostic network by cycling over the built-in graphs with
vertex count < 400, transferring the SAME weights across graphs. A full pass
over those graphs is one "round"; total = rounds * n_graphs cycles.

The network is parameterized by per-vertex local features (graph-agnostic), so
the same weights apply to any graph regardless of vertex count — this is the
transfer/migration path (see ``ref/implementation.md``).

NOTE ON STRATEGY: this from-zero self-play loop is the THEORETICAL line (the
"compute-abundant" ideal, AlphaZero/KataGo style from random init). The MAIN
line is distillation: ``distill.py`` pretrains the net against KataGo labels
into a base model (「基础培养」, a strong graph-agnostic net), and this trainer
is then used to finetune / transfer those weights via ``--resume``. Do not
expect this from-zero loop alone to escape uniform random on a single small GPU.

Usage (distilled / strong-start regime, matches ``starter/*.bat``):
  python gkt_train_gpu.py --sim 256 --workers 1 --gpw 32 --steps 16 \
      --lr 1e-4 --value-weight 3000 --own-weight 25 --temperature 0.1 \
      --buffer-drop-from-round 5 --device cuda \
      [--rules go|gomoku|antigomoku] [--outdir ../cur_mod_gnn]
"""
from __future__ import annotations
import os
import sys
import json
import re
import time
import argparse
import gc
import random
import numpy as np
import torch

# make sibling modules importable when invoked from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphs import (builtin_graphs, get_builtin,  # noqa: E402
                    GOMOKU_KEYS, GOMOKU_TRAIN_KEYS, default_train_outdir,
                    RULES_CHOICES, is_k_in_row_rules)
from gkt_gpu import (GktTrainer,  # noqa: E402
                       make_net, save_net, load_net, maybe_script_infer,
                       gpu_net_finite)
from gkt import play_eval_match, score_lead, make_move_heartbeat, write_graph_round_summary, reset_worker_progress, feature_dim, load_unused_buffer, save_unused_buffer, drop_legacy_round_npz, drop_oldest_buffer, curriculum_max_moves, REPLAY_ROUNDS, BUFFER_DROP_FROM_ROUND, BUFFER_SNAPSHOT_ROUNDS, MODEL_SNAPSHOT_ROUNDS, auto_buffer_drop, parse_from_round_map, save_buffer_snapshot, prune_buffer_snapshots, prune_model_snapshots, prune_big_snapshots  # noqa: E402

EXCLUDE = {"2", "6"}  # oversized: 3721 / 6859 vertices. 2 is a post-train grid generalization board, not a train key.
GRID_KEYS = ("0", "0.5", "1", "2", "3", "G9", "G15", "G7d", "G9d")  # 2 kept for 2DCNN inference/UI; EXCLUDE drops it from training


def arena_match(new_weights, old_weights, graph, n_games, sim,
                max_moves, n_features, hidden_dim, n_blocks,
                attn_layer, n_heads, device, net_type="gnn",
                num_players=2, rules="go", win_length=5,
                progress_file=None, label="", tag=""):
    """Pit new vs old; return ``(total_lead, games)``.

    Each game records new's *normalized* score-lead over old — a continuous
    score difference (``score_lead``, the exact target the value head learns),
    NOT a 0/1 win/loss. Rotating the new seat across games turns the
    first-move advantage into a sign-flipping additive term that cancels in
    the mean, so the average equals new's strength gap over old regardless of
    komi. A 0/1 win/loss would saturate on high-first-move-advantage graphs
    (e.g. the line graph) — the continuous lead does not.
    """
    def _make(w):
        net = make_net(net_type, n_features, hidden_dim, n_blocks,
                       attn_layer=attn_layer, n_heads=n_heads,
                       graph=graph, device=device, lr=0.0,
                       num_players=num_players)
        net.set_weights(w)
        net.eval()
        return maybe_script_infer(net, graph, device)

    net_new = _make(new_weights)
    net_old = _make(old_weights)
    k = max(2, int(num_players))
    sides = list(range(1, k + 1))
    n = max(1, len(graph.vertices))
    # Balance the first-move advantage exactly: rotate the new seat through
    # every side a whole number of times (an odd n_games would leave a
    # residual first-move bias in the average).
    if n_games % k:
        n_games += k - (n_games % k)
    total = 0.0
    for g in range(n_games):
        seat_new = (g % k) + 1
        nets = {s: (net_new if s == seat_new else net_old) for s in sides}
        hb = None
        if progress_file:
            hb = make_move_heartbeat(
                progress_file, f"{tag} vs {label} game {g + 1}/{n_games}")
        scores = play_eval_match(nets, graph, sim, max_moves,
                                 batch_size=64,
                                 rules=rules, win_length=win_length,
                                 heartbeat=hb)
        total_score = sum(float(scores[s]) for s in sides)
        total += score_lead(float(scores[seat_new]), total_score, n, k)
    return total, n_games


def _load_weight_dict(path: str) -> dict:
    """Load a checkpoint's weights as a numpy dict (same format as get_weights).

    Reads only ``model_state_dict`` straight off disk — no net construction —
    so it works for any architecture without needing a graph, and stays cheap
    even when several historical snapshots are loaded for the Arena panel.
    """
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a GKT GPU checkpoint")
    return {k: v.detach().cpu().numpy().copy()
            for k, v in ckpt["model_state_dict"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=None,
                    help="comma-separated keys (default: Graph-Go set, or "
                         "Gomoku / Anti-Gomoku rotation if --rules gomoku|antigomoku)")
    ap.add_argument("--rules", default="go", choices=list(RULES_CHOICES))
    ap.add_argument("--win-length", type=int, default=5,
                    help="Gomoku k-in-a-row (ignored for Graph-Go)")
    ap.add_argument("--sim", type=int, default=256,
                    help="nominal MCTS simulations per self-play move; each "
                         "search actually draws log-uniform in [sim/4, 4*sim]")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="self-play move temperature (tau): 1.0 = exploratory "
                         "(from-zero); 0.1 = exploitative (distilled start)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpw", type=int, default=16,
                    help="self-play games per worker per cycle")
    ap.add_argument("--rounds", type=int, default=4,
                    help="full passes over all graphs (total = rounds * n_graphs cycles)")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                    help="self-play player count; F matches this (separate 2P/3P/4P nets)")
    ap.add_argument("--n-blocks", type=int, default=20,
                    help="GNN message-passing layers / 2DCNN residual blocks")
    ap.add_argument("--net", default="gnn", choices=["gnn", "2dcnn"],
                    help="GPU network: gnn (any graph) or 2dcnn "
                         "(rectangular .grid; default training omits oversized 2)")
    ap.add_argument("--attn-layer", type=int, default=8,
                    help="GNN only: insert one global self-attention layer "
                         "before block #N (2DCNN ignores this; no n×n attention)")
    ap.add_argument("--attn-heads", type=int, default=4,
                    help="number of heads in the global attention layer")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0,
                    help="SGD weight on value MSE (reported vloss stays unweighted). "
                         "Distilled Go starters pass 3000 to match distill.py")
    ap.add_argument("--own-weight", type=float, default=1.0,
                    help="SGD weight on ownership MSE (reported oloss stays unweighted). "
                         "Distilled Go starters pass 25 to match distill.py")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="training batch size (GNN n×n attention ~64 max on "
                         "6 GB VRAM; 256 OOMs)")
    ap.add_argument("--steps", type=int, default=4,
                    help="SGD steps per graph per cycle (steps * batch-size "
                         "samples consumed per cycle; raise to consume more of "
                         "each round's fresh games)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfplay-device", default="cuda",
                    help="device for self-play net forward (cuda recommended: "
                         "the GNN's dense adjacency matmul is ~12x faster on GPU)")
    ap.add_argument("--selfplay-batch", type=int, default=64,
                    help="leaf-evaluation batch size in MCTS")
    ap.add_argument("--outdir", default=None,
                    help="default ../cur_mod_gnn or ../cur_mod_cnn2d "
                         "(../cur_mod_gomoku_* / ../cur_mod_antigomoku_* for k-in-a-row)")
    ap.add_argument("--min-moves", type=int, default=40,
                    help="curriculum: nominal game length cap at round 1 "
                         "(independent of n); each self-play game draws "
                         "log-uniform in [cap/2, 2*cap]. Ignored when "
                         "--rules gomoku|antigomoku (cap is always n)")
    ap.add_argument("--max-move-factor", type=float, default=2.0,
                    help="curriculum: nominal cap ceiling = n*factor + min-moves. "
                         "Ignored when --rules gomoku|antigomoku")
    ap.add_argument("--curriculum-rounds", type=int, default=0,
                    help="curriculum: rounds to grow the nominal cap linearly "
                         "from min-moves to n*factor+min-moves. 0 = disabled "
                         "(full length from round 1; the distilled / strong-start "
                         "default). Ignored when --rules gomoku|antigomoku")
    ap.add_argument("--replay-rounds", type=int, default=REPLAY_ROUNDS,
                    help="1 = skip unused.npz; >1 = load this graph's unused queue")
    ap.add_argument("--buffer-drop-from-round", type=str,
                    default=str(BUFFER_DROP_FROM_ROUND),
                    help="round at which to start dropping the oldest unused "
                         "samples (= how many generations of replay each graph "
                         "keeps). Accepts a bare int for all graphs, or "
                         "key=round,... per-graph overrides (e.g. 0=20,R1=15).")
    ap.add_argument("--buffer-drop", type=int, default=None,
                    help="oldest unused samples to drop per graph from "
                         "--buffer-drop-from-round onward. 0 = off; positive = "
                         "fixed count; unset = auto (this round's unused "
                         "samples, i.e. new - consumed)")
    ap.add_argument("--buffer-snapshot-rounds", type=int,
                    default=BUFFER_SNAPSHOT_ROUNDS,
                    help="keep a per-round snapshot of each graph's unused "
                         "buffer (rollback point) for the last N rounds; "
                         "0 = off")
    ap.add_argument("--model-snapshot-rounds", type=int,
                    default=MODEL_SNAPSHOT_ROUNDS,
                    help="keep per-round model checkpoints (round*.pt) for the "
                         "last N rounds; 0 = off (keep none); use a very large "
                         "N to keep them all")
    ap.add_argument("--big-snapshot-interval", type=int, default=10,
                    help="save a long-horizon checkpoint (big*.pt) every N "
                         "rounds, independent of the small round*.pt "
                         "snapshots; 0 = off")
    ap.add_argument("--big-snapshot-rounds", type=int, default=5,
                    help="keep the last N big*.pt checkpoints (they double as "
                         "the long-horizon Arena opponents); 0 = keep none")
    ap.add_argument("--infinite", action="store_true",
                    help="loop forever, saving a checkpoint after each round")
    ap.add_argument("--resume", default=None,
                    help="path to a previous new.pt (or other .pt) to continue from")
    ap.add_argument("--arena", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="after each round, pit the new weights vs a panel of "
                         "historical opponents (best-so-far + previous small "
                         "snapshot + last N big snapshots) across the training "
                         "graphs; accept only if the cross-graph average "
                         "normalized score-lead (new vs old, points/n) >= "
                         "--arena-lead-threshold, else roll back. ON by "
                         "default; pass --no-arena to disable")
    ap.add_argument("--arena-games", type=int, default=4,
                    help="games per graph per Arena opponent (colours alternate; "
                         "rounded up to balance both colours)")
    ap.add_argument("--arena-graphs", type=int, default=0,
                    help="how many training graphs to evaluate in Arena "
                         "(0 = all)")
    ap.add_argument("--arena-lead-threshold", type=float, default=0.0,
                    help="min cross-graph average normalized score-lead for "
                         "acceptance; 0.0 = accept as long as new is not "
                         "weaker than old (komi-independent)")
    ap.add_argument("--arena-sim", type=int, default=200,
                    help="MCTS simulations per move during Arena evaluation")
    ap.add_argument("--value-target", default="mc", choices=["mc", "q", "mix"],
                    help="score-lead target: mc = final (my stones − others)/n, "
                         "q = MCTS root Q, mix = blend of both")
    ap.add_argument("--q-lambda", type=float, default=0.5,
                    help="weight of Monte-Carlo z in 'mix' value target "
                         "(1 - q_lambda weights the MCTS root Q)")
    args = ap.parse_args()

    krow = is_k_in_row_rules(args.rules)
    if krow:
        args.players = 2
        if args.graphs is None:
            keys = list(GOMOKU_TRAIN_KEYS)
        else:
            keys = [k.strip() for k in args.graphs.split(",") if k.strip()]
    else:
        if args.graphs is None:
            keys = [k for k in builtin_graphs()
                    if k not in EXCLUDE and k not in GOMOKU_KEYS]
        else:
            keys = [k.strip() for k in args.graphs.split(",") if k.strip()]
            keys = [k for k in keys if k not in EXCLUDE and k not in GOMOKU_KEYS]
    if args.outdir is None:
        args.outdir = default_train_outdir(args.rules, args.net)
    os.makedirs(args.outdir, exist_ok=True)

    lf = open(os.path.join(args.outdir, "gkt_train.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()

    # persistent, graph-agnostic network — weights transfer across graphs.
    # --resume continues from a previous new.pt (live checkpoint in outdir).
    if args.resume:
        W = load_net(args.resume, device=args.device)
        log(f"resumed from checkpoint: {args.resume} "
            f"(net={W.net_type}, {W.num_players}P, F={W.n_features}, "
            f"H={W.hidden_dim}, blocks={W.n_blocks})")
        if int(W.n_features) != feature_dim(W.num_players):
            raise SystemExit(
                f"checkpoint F={W.n_features} != feature_dim({W.num_players})="
                f"{feature_dim(W.num_players)}")
    else:
        W = make_net(args.net, n_features=feature_dim(args.players), hidden_dim=args.hidden,
                     n_blocks=args.n_blocks,
                     attn_layer=args.attn_layer, n_heads=args.attn_heads,
                     device=args.device, lr=args.lr,
                     num_players=args.players)

    if W.net_type == "2dcnn":
        kept, dropped = [], []
        for k in keys:
            (kept if k in GRID_KEYS else dropped).append(k)
        if dropped:
            # 2DCNN reshapes vertices into an m×n image; only graphs with .grid.
            keys = kept
        if not keys:
            log("2DCNN training needs at least one grid graph (0, 0.5, 1, 3).")
            lf.close()
            return
        log(f"2DCNN: training on grid boards only: {', '.join(keys)}"
            + (f" (skipped {', '.join(dropped)})" if dropped else ""))

    rounds_str = "infinite" if args.infinite else str(args.rounds)
    log(f"=== cross-graph training: rules={args.rules} {len(keys)} graphs "
        f"[{', '.join(keys)}], sim={args.sim}, "
        f"{W.num_players}P F={W.n_features}, "
        f"device={args.device}, workers={args.workers}, rounds={rounds_str} ===")
    if krow:
        log(f"{args.rules}: game length cap is n (no Graph-Go move curriculum)")
        ignored = [f for f in ("--min-moves", "--max-move-factor",
                               "--curriculum-rounds")
                   if any(a == f or a.startswith(f + "=") for a in sys.argv)]
        if ignored:
            log(f"{args.rules}: ignoring Graph-Go curriculum flags "
                + ", ".join(ignored))
    else:
        log(f"curriculum: nominal {args.min_moves} moves -> n*{args.max_move_factor}+{args.min_moves} "
            f"over {args.curriculum_rounds} rounds; each game log-uniform in "
            f"[cap/2, 2*cap]")
    args.replay_rounds = max(1, int(args.replay_rounds))
    args.buffer_drop_from_round, args.buffer_drop_from_round_map = \
        parse_from_round_map(args.buffer_drop_from_round)
    if args.buffer_drop is not None:
        args.buffer_drop = max(0, int(args.buffer_drop))  # 0=off, positive=fixed
    args.buffer_snapshot_rounds = max(0, int(args.buffer_snapshot_rounds))
    args.model_snapshot_rounds = max(0, int(args.model_snapshot_rounds))
    args.big_snapshot_interval = max(0, int(args.big_snapshot_interval))
    args.big_snapshot_rounds = max(0, int(args.big_snapshot_rounds))
    # None is kept as a sentinel: the main loop computes the auto drop from the
    # *measured* unused surplus of each graph (see auto_buffer_drop).
    log(f"replay: unused.npz per graph (reload if --replay-rounds>1, "
        f"now {args.replay_rounds}); {os.path.join(args.outdir, 'replay')}")
    if args.buffer_drop is None:
        log(f"buffer drop: from round {args.buffer_drop_from_round}"
            f"{' (per-key ' + str(args.buffer_drop_from_round_map) + ')' if args.buffer_drop_from_round_map else ''}, "
            f"auto = this round's unused samples (new - consumed) per graph")
    elif args.buffer_drop > 0:
        log(f"buffer drop: from round {args.buffer_drop_from_round}, "
            f"{args.buffer_drop} oldest unused per graph (fixed)")
    log(f"SGD aug: random vertex relabel for all nets; GNN also permutes adj")
    log(f"head weights: value={args.value_weight:g} own={args.own_weight:g} "
        f"(reported pl/vl/ol unweighted)")
    log(f"arena: {'ON' if args.arena else 'OFF'} "
        f"(lead threshold {args.arena_lead_threshold:+.4f}, "
        f"{args.arena_games} games/graph/opponent, sim {args.arena_sim}, "
        f"{len(keys) if args.arena_graphs <= 0 else args.arena_graphs} graphs)")
    if args.big_snapshot_interval > 0:
        log(f"big snapshots: every {args.big_snapshot_interval} rounds, "
            f"keep last {args.big_snapshot_rounds}")
    else:
        log("big snapshots: off")

    summary = []
    cycle_no = 0
    t_start = time.time()

    # Per-graph resume: last summary.json row is the next round/graph.
    # Shuffle uses random.Random(rnd), so that round's order is reproducible.
    start_rnd = 1
    start_idx = 0
    if args.resume:
        try:
            with open(os.path.join(args.outdir, "summary.json"), "r",
                      encoding="utf-8") as f:
                prev = json.load(f)
            summary = prev  # keep history; do not overwrite on save
            if prev:
                last = prev[-1]
                cycle_no = int(last["cycle"])
                last_round = int(last["round"])
                lk = last["key"]
                rk = list(keys)
                random.Random(last_round).shuffle(rk)
                if lk in rk:
                    idx = rk.index(lk)
                    if idx < len(rk) - 1:
                        start_rnd, start_idx = last_round, idx + 1
                    else:
                        # The round's last graph is already done. If Arena is
                        # on, re-enter this round's Arena gate first (an empty
                        # graph list drops straight into the gate below) instead
                        # of silently skipping it; otherwise advance a round.
                        if args.arena:
                            start_rnd, start_idx = last_round, len(rk)
                        else:
                            start_rnd, start_idx = last_round + 1, 0
                else:
                    start_rnd, start_idx = last_round + 1, 0
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
            try:
                nums = [int(m.group(1)) for fn in os.listdir(args.outdir)
                        if (m := re.match(r"round(\d+)\.pt$", fn))]
                if nums:
                    start_rnd = max(nums) + 1
            except OSError:
                pass

    # Arena: keep the best-seen weights so a whole round can be rejected if the
    # new net loses to them (guards against policy collapse).
    best_weights = None
    if args.arena:
        best_path = os.path.join(args.outdir, "best.pt")
        if os.path.isfile(best_path):
            try:
                best_weights = _load_weight_dict(best_path)
                log(f"arena: loaded {best_path} as initial best weights")
            except Exception as e:  # noqa: BLE001
                log(f"arena: failed to load {best_path} ({e}); "
                    f"falling back to current weights")
                best_weights = None
        if best_weights is None:
            best_weights = W.get_weights()

    rnd = start_rnd - 1
    while True:
        rnd += 1
        # Shuffle graph order each round (seeded by rnd, so resume can rebuild
        # it). Avoids always training early keys more, and avoids a fixed
        # update order becoming a spurious signal under shared weights.
        round_keys = list(keys)
        random.Random(rnd).shuffle(round_keys)
        if rnd == start_rnd:
            round_keys = round_keys[start_idx:]   # skip graphs already done this round
        if len(round_keys) > 1:
            log(f"round {rnd} training order: {', '.join(round_keys)}")
        for key in round_keys:
            cycle_no += 1
            g = get_builtin(key)
            n = len(g.vertices)
            if krow:
                max_moves = n
            else:
                max_moves = curriculum_max_moves(
                    n, rnd, args.min_moves, args.max_move_factor,
                    args.curriculum_rounds)
            progress_file = os.path.abspath(
                os.path.join(args.outdir, "progress.txt"))
            hb_paths = reset_worker_progress(progress_file, args.workers)
            log("self-play heartbeat → " + ", ".join(hb_paths))
            trainer = GktTrainer(
                g, n_features=W.n_features, hidden_dim=W.hidden_dim,
                n_blocks=W.n_blocks,
                attn_layer=getattr(W, "attn_layer", args.attn_layer),
                n_heads=getattr(W, "n_heads", args.attn_heads),
                device=args.device, lr=args.lr, n_workers=args.workers,
                games_per_worker=args.gpw, n_simulations=args.sim,
                temperature=args.temperature,
                batch_size=args.batch_size, steps_per_cycle=args.steps,
                max_moves=max_moves,
                selfplay_device=args.selfplay_device,
                selfplay_batch=args.selfplay_batch, log_fn=log,
                value_target=args.value_target, q_lambda=args.q_lambda,
                net_type=W.net_type,
                num_players=W.num_players,
                progress_file=progress_file,
                rules=args.rules, win_length=args.win_length,
                value_weight=args.value_weight, own_weight=args.own_weight)
            # transfer the shared weights into this graph's trainer
            trainer.net.set_weights(W.get_weights())
            replay = load_unused_buffer(args.outdir, key, args.replay_rounds)
            t0 = time.time()
            hist = trainer.train(n_cycles=1, seed=0, replay=replay)
            unused = hist.get("unused")
            if unused is None:
                unused = []
            drop_n = args.buffer_drop
            if drop_n is None:
                # This round's *unused* new samples = survivors after SGD =
                # len(unused) - len(replay). Drop only that many so the queue
                # stays constant instead of draining by the consumed count.
                drop_n = auto_buffer_drop(len(unused) - len(replay))
            from_round = args.buffer_drop_from_round_map.get(
                key, args.buffer_drop_from_round)
            if (drop_n > 0 and rnd >= from_round):
                unused, ndrop = drop_oldest_buffer(unused, drop_n)
                if ndrop:
                    log(f"graph {key}: drop {ndrop} oldest unused "
                        f"(round {rnd}, left={len(unused)})")
            save_unused_buffer(args.outdir, key, unused,
                               cap=getattr(trainer, "buffer_capacity", None))
            drop_legacy_round_npz(args.outdir, key)
            # pull the updated weights back into the shared net
            if gpu_net_finite(trainer.net):
                W.set_weights(trainer.net.get_weights())
            else:
                log(f"graph {key}: non-finite net after SGD, keep previous weights")
            ploss = hist["policy_losses"][-1] if hist["policy_losses"] else None
            vloss = hist["value_losses"][-1] if hist["value_losses"] else None
            oloss = hist["own_losses"][-1] if hist.get("own_losses") else None
            dt = time.time() - t0
            ploss_str = f"{ploss:.4f}" if ploss is not None else "n/a"
            vloss_str = f"{vloss:.4f}" if vloss is not None else "n/a"
            oloss_str = f"{oloss:.4f}" if oloss is not None else "n/a"
            buf_n = len(unused)
            log(f"[cycle {cycle_no}] round {rnd} graph {key} (n={n}) "
                f"ploss={ploss_str} vloss={vloss_str} oloss={oloss_str} "
                f"buffer={buf_n} {dt:.1f}s")
            rec = {"cycle": cycle_no, "round": rnd, "key": key,
                   "n_vertices": n, "policy_loss": ploss,
                   "value_loss": vloss, "own_loss": oloss,
                   "seconds": round(dt, 1),
                   "samples": int(hist.get("new_samples") or 0),
                   "buffer": int(buf_n),
                   "sim": int(args.sim), "max_moves": int(max_moves)}
            summary.append(rec)
            write_graph_round_summary(args.outdir, rec)

            # Per-graph checkpoint: overwrite new.pt so resume continues
            # after the last finished graph.
            if gpu_net_finite(W):
                save_net(W, os.path.join(args.outdir, "new.pt"))
                log(f"snapshot saved (latest new.pt after graph {key})")
            else:
                log(f"skip snapshot after graph {key}: non-finite weights")
            with open(os.path.join(args.outdir, "summary.json"), "w",
                      encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Empty CUDA cache between graphs so the next board does not
            # inherit a fragmented caching allocator.
            if args.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        # Round-end artifacts only fire when this round actually trained at least
        # one graph. Resuming right after a round finished leaves an empty graph
        # list; the checkpoint / big / buffer snapshots were already written when
        # the round first completed, so skip them and drop straight into Arena.
        if round_keys:
            # per-round model checkpoint (pruned to the last N rounds below)
            ckpt = os.path.join(args.outdir, f"round{rnd}.pt")
            if gpu_net_finite(W):
                save_net(W, ckpt)
                log(f"round {rnd} done; model checkpoint {ckpt}")
            else:
                log(f"round {rnd}: skip model checkpoint (non-finite weights)")
            # keep only the most recent N round*.pt so disk does not grow
            # forever; new.pt / best.pt are never touched by this prune.
            n_del = prune_model_snapshots(args.outdir, args.model_snapshot_rounds,
                                          ".pt")
            if n_del:
                log(f"round {rnd}: pruned {n_del} old model checkpoint(s) "
                    f"(keep last {args.model_snapshot_rounds})")

            # Long-horizon big snapshot every --big-snapshot-interval rounds,
            # independent of the small round*.pt snapshots. These feed the Arena
            # panel as long-horizon opponents (guards against slow drift).
            if args.big_snapshot_interval > 0 and rnd % args.big_snapshot_interval == 0:
                big = os.path.join(args.outdir, f"big{rnd}.pt")
                if gpu_net_finite(W):
                    save_net(W, big)
                    log(f"round {rnd}: big snapshot {big}")
                else:
                    log(f"round {rnd}: skip big snapshot (non-finite weights)")
                n_bdel = prune_big_snapshots(args.outdir, args.big_snapshot_rounds,
                                             ".pt")
                if n_bdel:
                    log(f"round {rnd}: pruned {n_bdel} old big snapshot(s) "
                        f"(keep last {args.big_snapshot_rounds})")

            # Per-round buffer snapshots: unused.npz only holds the *current*
            # queue (overwritten each visit, oldest dropped), and round*.pt
            # stores weights only, so a corrupt/over-dropped buffer is otherwise
            # unrecoverable. Snapshot every graph's buffer at this round boundary
            # as a rollback point, and prune to the most recent N rounds.
            if args.buffer_snapshot_rounds > 0:
                for skey in keys:
                    sbuf = load_unused_buffer(args.outdir, skey,
                                              max(2, args.replay_rounds))
                    save_buffer_snapshot(args.outdir, skey, rnd, sbuf)
                    prune_buffer_snapshots(args.outdir, skey,
                                           args.buffer_snapshot_rounds)
                log(f"round {rnd}: buffer snapshots saved "
                    f"(keep last {args.buffer_snapshot_rounds} rounds)")

        # Arena: pit the new weights against a panel of historical opponents —
        # best-so-far, the previous round's small snapshot, and the last N big
        # snapshots — across the training graphs (weights are shared, so the
        # gate must be cross-graph to catch catastrophic forgetting on any
        # single graph). The panel spans both short horizon (last round) and
        # long horizon (every --big-snapshot-interval rounds), so a single
        # lucky round or a slow drift can both be caught.
        if args.arena and best_weights is not None:
            panel = [("best", best_weights)]
            if rnd > 1:
                prev_ckpt = os.path.join(args.outdir, f"round{rnd-1}.pt")
                if os.path.isfile(prev_ckpt):
                    try:
                        panel.append((f"round{rnd-1}",
                                      _load_weight_dict(prev_ckpt)))
                    except Exception as e:  # noqa: BLE001
                        log(f"arena: skip small snapshot {prev_ckpt}: {e}")
            big_items = []
            for fn in os.listdir(args.outdir):
                m = re.match(r"big(\d+)\.pt$", fn)
                if m and int(m.group(1)) < rnd:
                    big_items.append((int(m.group(1)), fn))
            big_items.sort(key=lambda t: t[0])
            for bid, fn in big_items[-args.big_snapshot_rounds:]:
                bp = os.path.join(args.outdir, fn)
                try:
                    panel.append((f"big{bid}", _load_weight_dict(bp)))
                except Exception as e:  # noqa: BLE001
                    log(f"arena: skip big snapshot {bp}: {e}")

            arena_keys = keys if args.arena_graphs <= 0 else keys[:args.arena_graphs]
            arena_progress = os.path.abspath(
                os.path.join(args.outdir, "progress.arena.txt"))
            try:
                open(arena_progress, "w", encoding="utf-8").close()
            except OSError:
                pass
            total_lead, total_games = 0.0, 0
            n_arena = len(arena_keys) * len(panel)
            log(f"arena: round {rnd} gate starting — {len(panel)} opponents x "
                f"{len(arena_keys)} graphs x {args.arena_games} games "
                f"({n_arena * args.arena_games} games total)")
            t_arena0 = time.time()
            done = 0
            for k in arena_keys:
                ag = get_builtin(k)
                amax = (len(ag.vertices) if krow
                        else int(len(ag.vertices) * args.max_move_factor) + args.min_moves)
                for label, opp_w in panel:
                    t0 = time.time()
                    lead, ng = arena_match(W.get_weights(), opp_w, ag,
                                           args.arena_games, args.arena_sim,
                                           amax, W.n_features,
                                           W.hidden_dim, W.n_blocks,
                                           getattr(W, "attn_layer", args.attn_layer),
                                           getattr(W, "n_heads", args.attn_heads),
                                           args.device,
                                           W.net_type,
                                           W.num_players,
                                           args.rules, args.win_length,
                                           progress_file=arena_progress,
                                           label=label,
                                           tag=f"arena r{rnd} g{k}")
                    total_lead += lead
                    total_games += ng
                    done += 1
                    log(f"arena: [{done}/{n_arena}] graph {k} vs {label}: "
                        f"lead {lead / max(1, ng):+.4f} "
                        f"({ng} games, {time.time() - t0:.0f}s)")
            log(f"arena: round {rnd} played {total_games} games in "
                f"{time.time() - t_arena0:.0f}s")
            avg_lead = total_lead / max(1, total_games)
            if avg_lead >= args.arena_lead_threshold:
                best_weights = W.get_weights()
                save_net(W, os.path.join(args.outdir, "best.pt"))
                log(f"arena: new accepted (avg lead {avg_lead:+.4f} over "
                    f"{total_games} games vs {len(panel)} opponents on "
                    f"{len(arena_keys)} graphs); best.pt updated")
            else:
                W.set_weights(best_weights)
                save_net(W, os.path.join(args.outdir, "new.pt"))
                log(f"arena: new REJECTED (avg lead {avg_lead:+.4f} over "
                    f"{total_games} games vs {len(panel)} opponents); "
                    f"rolled back to best")

        if not args.infinite and rnd >= args.rounds:
            break

    final_path = os.path.join(args.outdir, "new.pt")
    hrs = (time.time() - t_start) / 3600.0
    log(f"=== done: {cycle_no} cycles in {hrs:.2f} h; final model {final_path} ===")
    lf.close()
    print("Wrote", os.path.join(args.outdir, "summary.json"))


if __name__ == "__main__":
    main()
