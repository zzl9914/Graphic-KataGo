"""CPU-only cross-graph training.

Same loop as `gkt_train_gpu.py`, nets `--net mlp|1dcnn` (default MLP).
Docs: ``ref/algorithm.md``, ``ref/training_method.md``.

Usage (official Go, matches ``starter/*.bat``):
  python gkt_train_cpu.py --infinite
      [--net mlp] [--outdir ../cur_mod_mlp]
      [--rules go|gomoku|antigomoku]

Usage (cultivate2, matches ``base/cultivate2_*.bat``):
  python gkt_train_cpu.py --graphs 0 --rounds 20 --no-arena \
      --freeze-policy-until-round 10 --buffer-drop-from-round 21 \
      --model-snapshot-rounds 25 \
      --resume ../base/mlp/new.npz --outdir ../base/cultivate2/mlp
"""
from __future__ import annotations

import os
import sys
import json
import re
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# make sibling modules importable when invoked from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphs import builtin_graphs, get_builtin, GOMOKU_KEYS, GOMOKU_TRAIN_KEYS, default_train_outdir, RULES_CHOICES, is_k_in_row_rules  # noqa: E402
from gkt import GktSelfPlay, aux_from_sample, play_eval_match, score_lead, write_graph_round_summary, progress_append, make_move_heartbeat, worker_progress_path, reset_worker_progress, feature_dim, load_unused_buffer, save_unused_buffer, drop_oldest_buffer, curriculum_max_moves, shuffle_graph_keys, resume_graph_cursor, apply_krow_train_defaults, REPLAY_ROUNDS, BUFFER_DROP_FROM_ROUND, BUFFER_SNAPSHOT_ROUNDS, MODEL_SNAPSHOT_ROUNDS, auto_buffer_drop, parse_from_round_map, save_buffer_snapshot, prune_buffer_snapshots, prune_model_snapshots, prune_big_snapshots, quarantine_round_artifacts, rollback_unused_after_reject, VALUE_ABS_WEIGHT_DEFAULT, VALUE_RTO_WEIGHT_DEFAULT, VALUE_CONS_WEIGHT_DEFAULT, set_value_loss_attrs, snapshot_tail  # noqa: E402
from gkt_cpu import (MlpPolicyValueNet, Cnn1dPolicyValueNet,  # noqa: E402
                       save_cpu_net, load_cpu_net, _player_count, cpu_net_type,
                       cpu_net_finite)
from grid_sym import augment_vertex_batch  # noqa: E402

EXCLUDE = {"2", "6"}  # oversized examples: 3721 / 6859 vertices


def _make_net(net_type, F, H, lr, kernel_size, conv_layers, seed=0,
              num_players=2, value_weight=1.0, own_weight=1.0,
              value_rto_weight=1.0, value_cons_weight=VALUE_CONS_WEIGHT_DEFAULT):
    """Build a CPU net (mlp or 1dcnn). F = feature_dim(num_players)."""
    k = _player_count(num_players)
    if int(F) != feature_dim(k):
        raise ValueError(f"F={F} must be feature_dim(num_players)={feature_dim(k)}")
    if net_type == "1dcnn":
        return Cnn1dPolicyValueNet(F, H, kernel_size=kernel_size,
                                   n_layers=conv_layers, lr=lr, seed=seed,
                                   num_players=k,
                                   value_weight=value_weight, own_weight=own_weight,
                                   value_rto_weight=value_rto_weight,
                                   value_cons_weight=value_cons_weight)
    if net_type != "mlp":
        raise ValueError(f"unknown CPU net_type {net_type!r}")
    return MlpPolicyValueNet(F, H, lr=lr, seed=seed, num_players=k,
                             value_weight=value_weight, own_weight=own_weight,
                             value_rto_weight=value_rto_weight,
                             value_cons_weight=value_cons_weight)


# ---------------------------------------------------------------------------
# Self-play worker (top-level so it can be pickled for the process pool)
# ---------------------------------------------------------------------------
def _selfplay_worker(graph_key, weights, net_type, F, H, kernel_size, conv_layers,
                     sim, temperature, n_games, max_moves, batch_size,
                     q_lambda, seed, num_players=2,
                     progress_file=None, main_progress_file=None, worker_id=0,
                     rules="go", win_length=5):
    """Run `n_games` self-play games on a CPU copy of the net. Returns samples.

    Each sample is the 13-tuple from `GktSelfPlay.play_one_game`.
    """
    random.seed(seed + os.getpid())
    np.random.seed(seed + os.getpid())
    pid = os.getpid()
    tag = f"w{int(worker_id)} pid={pid}"
    g = get_builtin(graph_key)
    progress_append(progress_file,
                    f"{tag} worker_start graph={graph_key} "
                    f"n={len(g.vertices)} games={n_games} sim={sim} net={net_type}")
    k = max(2, int(num_players))
    net = _make_net(net_type, F, H, lr=0.0, kernel_size=kernel_size,
                    conv_layers=conv_layers, seed=0, num_players=k)
    net.load_state_dict(weights)
    progress_append(progress_file, f"{tag} net_ready starting games")
    driver = GktSelfPlay(g, net, n_simulations=sim, temperature=temperature,
                            max_moves=max_moves, batch_size=batch_size,
                            q_lambda=q_lambda,
                            rules=rules, win_length=win_length)
    samples = []
    for g_i in range(n_games):
        if main_progress_file:
            progress_append(main_progress_file,
                            f"{tag} game {g_i + 1}/{n_games} start")
        hb = make_move_heartbeat(
            progress_file, f"{tag} game {g_i + 1}/{n_games}")
        samples.extend(driver.play_one_game(heartbeat=hb))
        if main_progress_file:
            progress_append(main_progress_file,
                            f"{tag} game {g_i + 1}/{n_games} end "
                            f"({len(samples)} samples)")
        progress_append(progress_file,
                        f"{tag} game {g_i + 1}/{n_games} collected, "
                        f"{len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Arena (lightweight "new vs old" gate, CPU)
# ---------------------------------------------------------------------------
def arena_match(new_weights, old_weights, graph, n_games, sim,
                max_moves, F, H, net_type, kernel_size, conv_layers,
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
    if int(n_games) <= 0:
        return 0.0, 0

    def _make(w):
        net = _make_net(net_type, F, H, lr=0.0, kernel_size=kernel_size,
                        conv_layers=conv_layers, seed=0, num_players=num_players)
        net.load_state_dict(w)
        return net

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
    for g_i in range(n_games):
        seat_new = (g_i % k) + 1
        nets = {s: (net_new if s == seat_new else net_old) for s in sides}
        hb = None
        if progress_file:
            hb = make_move_heartbeat(
                progress_file, f"{tag} vs {label} game {g_i + 1}/{n_games}")
        scores = play_eval_match(nets, graph, sim, max_moves,
                                 batch_size=64,
                                 rules=rules, win_length=win_length,
                                 heartbeat=hb)
        total_score = sum(float(scores[s]) for s in sides)
        total += score_lead(float(scores[seat_new]), total_score, n, k)
    return total, n_games


def _load_weight_dict(path: str) -> dict:
    """Load a CPU checkpoint's weights as a numpy dict (same format as
    ``state_dict``), without constructing a net.

    Reads only the state-dict keys (skipping the ``_*`` metadata) straight
    off disk, so it is cheap even when several historical snapshots are loaded
    for the Arena panel.
    """
    data = np.load(path, allow_pickle=False)
    try:
        return {k: data[k].copy() for k in data.files
                if not str(k).startswith("_")}
    finally:
        data.close()


def _write_round_model_ckpts(args, net, rnd, log):
    """Per-round / big snapshots of the *accepted* (or no-Arena) weights."""
    ckpt = os.path.join(args.outdir, f"round{rnd}.npz")
    if cpu_net_finite(net):
        save_cpu_net(net, ckpt)
        log(f"round {rnd} done; model checkpoint {ckpt}")
    else:
        log(f"round {rnd}: skip model checkpoint (non-finite weights)")
    n_del = prune_model_snapshots(args.outdir, args.model_snapshot_rounds, ".npz")
    if n_del:
        log(f"round {rnd}: pruned {n_del} old model checkpoint(s) "
            f"(keep last {args.model_snapshot_rounds})")
    if args.big_snapshot_interval > 0 and rnd % args.big_snapshot_interval == 0:
        big = os.path.join(args.outdir, f"big{rnd}.npz")
        if cpu_net_finite(net):
            save_cpu_net(net, big)
            log(f"round {rnd}: big snapshot {big}")
        else:
            log(f"round {rnd}: skip big snapshot (non-finite weights)")
        n_bdel = prune_big_snapshots(args.outdir, args.big_snapshot_rounds, ".npz")
        if n_bdel:
            log(f"round {rnd}: pruned {n_bdel} old big snapshot(s) "
                f"(keep last {args.big_snapshot_rounds})")


def main():
    ap = argparse.ArgumentParser(description="CPU-only cross-graph training")
    ap.add_argument("--graphs", default=None,
                    help="comma-separated keys (default: Graph-Go set, or "
                         "Gomoku / Anti-Gomoku rotation if --rules gomoku|antigomoku)")
    ap.add_argument("--rules", default="go", choices=list(RULES_CHOICES))
    ap.add_argument("--win-length", type=int, default=5,
                    help="Gomoku k-in-a-row (ignored for Graph-Go)")
    ap.add_argument("--sim", type=int, default=256,
                    help="nominal MCTS simulations per self-play move; each "
                         "search actually draws log-uniform in [sim/4, 4*sim]")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="self-play move temperature (tau); k-in-a-row fills 1.0 "
                         "if not on the CLI")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpw", type=int, default=32,
                    help="self-play games per worker per cycle; k-in-a-row "
                         "fills 16 if not on the CLI")
    ap.add_argument("--rounds", type=int, default=4,
                    help="full passes over all graphs (total = rounds * n_graphs cycles)")
    ap.add_argument("--net", default="mlp", choices=["mlp", "1dcnn"],
                    help="CPU network: mlp or 1dcnn")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                    help="self-play player count; F matches this (separate 2P/3P/4P nets)")
    ap.add_argument("--kernel-size", type=int, default=3,
                    help="1DCNN kernel size (only with --net 1dcnn)")
    ap.add_argument("--conv-layers", type=int, default=20,
                    help="1DCNN layers (only with --net 1dcnn)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=VALUE_ABS_WEIGHT_DEFAULT,
                    help="SGD weight on value_abs (stone-lead) MSE; reported "
                         "vloss is unweighted abs. k-in-a-row fills 1 if not on CLI")
    ap.add_argument("--value-rto-weight", type=float,
                    default=VALUE_RTO_WEIGHT_DEFAULT,
                    help="SGD weight on value_rto (lead/n, search head) MSE. "
                         "k-in-a-row fills 1 if not on CLI")
    ap.add_argument("--value-cons-weight", type=float,
                    default=VALUE_CONS_WEIGHT_DEFAULT,
                    help="SGD weight on (abs - n*rto)^2 (k-in-a-row: abs - rto). "
                         "Very low so the heads can still disagree")
    ap.add_argument("--own-weight", type=float, default=5.0,
                    help="SGD weight on ownership MSE (reported oloss stays unweighted). "
                         "k-in-a-row fills 1 if not on the CLI")
    ap.add_argument("--freeze-policy-until-round", type=int, default=0,
                    help="1-based: freeze the policy readout for rounds 1..N "
                         "(train trunk + value/own/aux). 0 = never freeze")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="training batch size (per-sample SGD)")
    ap.add_argument("--steps", type=int, default=16,
                    help="SGD steps per graph per cycle; k-in-a-row fills 4 "
                         "if not on the CLI")
    ap.add_argument("--selfplay-batch", type=int, default=32,
                    help="leaf-evaluation batch size in MCTS")
    ap.add_argument("--outdir", default=None,
                    help="default ../cur_mod_mlp or ../cur_mod_cnn1d "
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
                    help="keep per-round model checkpoints (round*.npz) for "
                         "the last N rounds; 0 = off (keep none); use a very "
                         "large N to keep them all")
    ap.add_argument("--big-snapshot-interval", type=int, default=10,
                    help="save a long-horizon checkpoint (big*.npz) every N "
                         "rounds, independent of the small round*.npz "
                         "snapshots; 0 = off")
    ap.add_argument("--big-snapshot-rounds", type=int, default=5,
                    help="keep the last N big*.npz checkpoints (they double as "
                         "the long-horizon Arena opponents); 0 = keep none")
    ap.add_argument("--infinite", action="store_true",
                    help="loop forever, saving a checkpoint after each round")
    ap.add_argument("--resume", default=None,
                    help="path to a previous new.npz (or other .npz) to continue from")
    ap.add_argument("--q-lambda", type=float, default=0.5,
                    help="weight of Monte-Carlo z in the mix value target "
                         "(1 - q_lambda weights the MCTS root Q)")
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
                    help="how many training graphs to evaluate in Arena (0 = all)")
    ap.add_argument("--arena-lead-threshold", type=float, default=0.0,
                    help="min cross-graph average normalized score-lead for "
                         "acceptance; 0.0 = accept as long as new is not "
                         "weaker than old (komi-independent)")
    ap.add_argument("--arena-sim", type=int, default=100)
    args = ap.parse_args()
    krow_applied = apply_krow_train_defaults(args)

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
    keys = [str(k) for k in keys]
    if args.outdir is None:
        args.outdir = default_train_outdir(args.rules, args.net)
    os.makedirs(args.outdir, exist_ok=True)

    lf = open(os.path.join(args.outdir, "gkt_train_cpu.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()

    if args.resume:
        net = load_cpu_net(args.resume)
        net.lr = args.lr
        net_kind = cpu_net_type(net)
        F = net.F
        kernel_size = int(getattr(net, "K", args.kernel_size))
        conv_layers = int(getattr(net, "L", args.conv_layers))
        log(f"resumed from checkpoint: {args.resume} (net={net_kind}, "
            f"{net.num_players}P, F={net.F}, H={net.H})")
        if int(F) != feature_dim(net.num_players):
            raise SystemExit(
                f"checkpoint F={F} != feature_dim={feature_dim(net.num_players)}")
        if args.net != net_kind:
            log(f"--net {args.net} ignored; checkpoint is {net_kind}")
    else:
        F = feature_dim(args.players)
        net_kind = args.net
        kernel_size = args.kernel_size
        conv_layers = args.conv_layers
        net = _make_net(net_kind, F, args.hidden, args.lr,
                        kernel_size, conv_layers, seed=0,
                        num_players=args.players,
                        value_weight=args.value_weight,
                        own_weight=args.own_weight,
                        value_rto_weight=args.value_rto_weight,
                        value_cons_weight=args.value_cons_weight)

    net.value_weight = float(args.value_weight)
    net.value_rto_weight = float(args.value_rto_weight)
    net.own_weight = float(args.own_weight)
    set_value_loss_attrs(
        net, value_weight=args.value_weight,
        value_rto_weight=args.value_rto_weight, own_weight=args.own_weight,
        value_cons_weight=args.value_cons_weight, rules=args.rules)

    rounds_str = "infinite" if args.infinite else str(args.rounds)
    log(f"=== CPU cross-graph training: rules={args.rules} {len(keys)} graphs "
        f"[{', '.join(keys)}], sim={args.sim}, "
        f"workers={args.workers}, net={net_kind}, {net.num_players}P, "
        f"hidden={args.hidden}, rounds={rounds_str}, "
        f"value=mix q_lambda={args.q_lambda:g} ===")
    if krow_applied:
        log(f"{args.rules}: defaulted {', '.join(krow_applied)} (not on CLI)")
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
    log("SGD aug: random vertex relabel (MLP and 1DCNN)")
    log(f"head weights: abs={args.value_weight:g} rto={args.value_rto_weight:g} "
        f"cons={args.value_cons_weight:g} "
        f"own={args.own_weight:g} "
        f"(reported pl/vl/ol unweighted)")
    until = max(0, int(args.freeze_policy_until_round))
    if until > 0:
        log(f"policy freeze: rounds 1-{until} freeze readout, then unfreeze")
    log(f"arena: {'ON' if args.arena else 'OFF'} "
        f"(lead threshold {args.arena_lead_threshold:+.4f}, "
        f"{args.arena_games} games/graph/opponent, sim {args.arena_sim}, "
        f"{len(keys)} graphs)")
    if args.big_snapshot_interval > 0:
        log(f"big snapshots: every {args.big_snapshot_interval} rounds, "
            f"keep last {args.big_snapshot_rounds}")

    summary = []
    cycle_no = 0
    t_start = time.time()

    # Per-graph resume: last summary.json row is the next round/graph.
    # Graph order is shuffle_graph_keys (private Random(rnd)), so this
    # round's remaining graphs can be reconstructed. SGD uses a different RNG.
    start_rnd = 1
    start_idx = 0
    if args.resume:
        try:
            with open(os.path.join(args.outdir, "summary.json"), "r",
                      encoding="utf-8") as f:
                prev = json.load(f)
            summary = prev
            if prev:
                last = prev[-1]
                cycle_no = int(last["cycle"])
                last_round = int(last["round"])
                lk = last["key"]
                start_rnd, start_idx, rk, found = resume_graph_cursor(
                    keys, last_round, lk, args.arena)
                if not found:
                    log(f"resume: last graph {lk!r} not in this run's set "
                        f"[{', '.join(keys)}]; starting round {start_rnd}")
                elif start_idx < len(rk):
                    log(f"resume: round {start_rnd} after graph {lk}; "
                        f"next {', '.join(rk[start_idx:])}")
                elif args.arena and start_rnd == last_round:
                    log(f"resume: round {last_round} graphs done; "
                        f"re-entering Arena")
                else:
                    log(f"resume: round {last_round} graphs done; "
                        f"starting round {start_rnd}")
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
            try:
                nums = [int(m.group(1)) for fn in os.listdir(args.outdir)
                        if (m := re.match(r"round(\d+)\.npz$", fn))]
                if nums:
                    start_rnd = max(nums) + 1
            except OSError:
                pass

    best_weights = None
    if args.arena:
        best_path = os.path.join(args.outdir, "best.npz")
        if os.path.isfile(best_path):
            try:
                best_weights = _load_weight_dict(best_path)
                log(f"arena: loaded {best_path} as initial best weights")
            except Exception as e:  # noqa: BLE001
                log(f"arena: failed to load {best_path} ({e}); "
                    f"falling back to current weights")
                best_weights = None
        if best_weights is None:
            best_weights = net.state_dict()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
    # Minibatch sampling must not share the process-global random with the
    # graph-order RNG. Unseeded: SGD unreproducibility is intentional.
    _sgd_rng = random.Random()

    try:
        rnd = start_rnd - 1
        while True:
            rnd += 1
            want_freeze = until > 0 and rnd <= until
            prev_freeze = bool(getattr(net, "freeze_policy", False))
            net.freeze_policy = want_freeze
            if until > 0 and (rnd == start_rnd or prev_freeze != want_freeze):
                log(f"round {rnd}: policy readout "
                    f"{'FROZEN (train trunk+value/own/aux)' if want_freeze else 'UNFROZEN'}")
            # Shuffle graph order each round (private Random(rnd), so resume
            # can rebuild it). Same helper as gkt_train_gpu.py.
            round_keys = shuffle_graph_keys(keys, rnd)
            if rnd == start_rnd:
                round_keys = round_keys[start_idx:]
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

                # 1. CPU multiprocess self-play with the current weights.
                # One submit per worker (n_games=gpw), same as GPU: pickle
                # full 1DCNN weights once per worker, not once per game.
                weights = net.state_dict()
                futures = [executor.submit(
                    _selfplay_worker, key, weights, net_kind, F, args.hidden,
                    kernel_size, conv_layers, args.sim, args.temperature, args.gpw,
                    max_moves, args.selfplay_batch, args.q_lambda,
                    rnd * 100000 + cycle_no * 1000 + i,
                    net.num_players,
                    worker_progress_path(progress_file, i), progress_file, i,
                    args.rules, args.win_length)
                    for i in range(args.workers)]
                this_round = []
                for fut in futures:
                    this_round.extend(fut.result())
                replay = load_unused_buffer(args.outdir, key,
                                            args.replay_rounds)
                buffer = replay + this_round

                # 2. CPU per-sample SGD, recording policy/value loss separately
                t0 = time.time()
                plosses, vlosses = [], []
                if len(buffer) >= args.batch_size:
                    for _ in range(args.steps):
                        if len(buffer) < args.batch_size:
                            break
                        idx = _sgd_rng.sample(range(len(buffer)), args.batch_size)
                        batch = [buffer[i] for i in idx]
                        Xs = np.stack([s[0] for s in batch])
                        Ms = np.stack([s[1] for s in batch])
                        Ps = np.stack([s[2] for s in batch])
                        owns = np.stack([s[5] for s in batch])
                        extra_p = [np.stack([s[7] for s in batch])]
                        extra_v = [np.stack([s[9] for s in batch])]
                        if args.net in ("mlp", "1dcnn"):
                            Xs, Ms, Ps, owns, ep, ev, _ = augment_vertex_batch(
                                Xs, Ms, Ps, ownership=owns,
                                extra_policy=extra_p, extra_vertex=extra_v)
                            new_batch = []
                            for i, s in enumerate(batch):
                                s = list(s)
                                s[0], s[1], s[2], s[5] = Xs[i], Ms[i], Ps[i], owns[i]
                                s[7] = ep[0][i]
                                s[9] = ev[0][i]
                                new_batch.append(tuple(s))
                            batch = new_batch
                        step_ok = True
                        step_p, step_v = [], []
                        for sample in batch:
                            X, mask, pol, me, zz, own = sample[:6]
                            pl, vl, ol, _ = net.backward(
                                X, mask, pol, zz, own, aux=aux_from_sample(sample))
                            if not (np.isfinite(pl) and np.isfinite(vl)):
                                step_ok = False
                                break
                            step_p.append(pl)
                            step_v.append(vl)
                        if step_ok:
                            drop = set(idx)
                            buffer = [s for i, s in enumerate(buffer) if i not in drop]
                            plosses.extend(step_p)
                            vlosses.extend(step_v)
                drop_n = args.buffer_drop
                if drop_n is None:
                    # This round's *unused* new samples = survivors after SGD =
                    # len(buffer) - len(replay). Drop only that many so the
                    # queue stays constant instead of draining by consumed count.
                    drop_n = auto_buffer_drop(len(buffer) - len(replay))
                from_round = args.buffer_drop_from_round_map.get(
                    key, args.buffer_drop_from_round)
                if (drop_n > 0 and rnd >= from_round):
                    buffer, ndrop = drop_oldest_buffer(buffer, drop_n)
                    if ndrop:
                        log(f"graph {key}: drop {ndrop} oldest unused "
                            f"(round {rnd}, left={len(buffer)})")
                save_unused_buffer(args.outdir, key, buffer)
                avg_p = float(np.mean(plosses)) if plosses else float("nan")
                avg_v = float(np.mean(vlosses)) if vlosses else float("nan")
                dt = time.time() - t0
                log(f"[cycle {cycle_no}] round {rnd} graph {key} (n={n}) "
                    f"ploss={avg_p:.4f} vloss={avg_v:.4f} "
                    f"+{len(this_round)} samples "
                    f"(buffer={len(buffer)} replay={len(replay)}) {dt:.1f}s")
                rec = {"cycle": cycle_no, "round": rnd, "key": key,
                       "n_vertices": n, "policy_loss": avg_p,
                       "value_loss": avg_v, "seconds": round(dt, 1),
                       "samples": int(len(this_round)),
                       "buffer": int(len(buffer)),
                       "sim": int(args.sim), "max_moves": int(max_moves)}
                summary.append(rec)
                write_graph_round_summary(args.outdir, rec)

                # Per-graph checkpoint: overwrite new.npz after each graph.
                if cpu_net_finite(net):
                    save_cpu_net(net, os.path.join(args.outdir, "new.npz"))
                    log(f"snapshot saved (latest new.npz after graph {key})")
                else:
                    log(f"skip snapshot after graph {key}: non-finite weights")
                with open(os.path.join(args.outdir, "summary.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

            # Round-end: buffer snapshots always. Model snapshots wait until
            # Arena accepts (rejected nets must not stay in the panel).
            # ``--no-arena`` writes model snapshots here. Resume after the last
            # graph of a round leaves ``round_keys`` empty and drops into Arena.
            if round_keys:
                if args.buffer_snapshot_rounds > 0:
                    for skey in keys:
                        sbuf = load_unused_buffer(args.outdir, skey,
                                                  max(2, args.replay_rounds))
                        save_buffer_snapshot(args.outdir, skey, rnd, sbuf)
                        prune_buffer_snapshots(args.outdir, skey,
                                               args.buffer_snapshot_rounds)
                    log(f"round {rnd}: buffer snapshots saved "
                        f"(keep last {args.buffer_snapshot_rounds} rounds)")
                if not args.arena:
                    _write_round_model_ckpts(args, net, rnd, log)

            # Arena: pit the new weights against a panel of historical opponents
            # — best-so-far, the previous round's small snapshot, and the last N
            # big snapshots — across the training graphs (weights are shared, so
            # the gate must be cross-graph to catch catastrophic forgetting on
            # any single graph). The panel spans both short horizon (last round)
            # and long horizon (every --big-snapshot-interval rounds).
            if args.arena and best_weights is not None:
                panel = [("best", best_weights)]
                if rnd > 1:
                    prev_ckpt = os.path.join(args.outdir, f"round{rnd-1}.npz")
                    if os.path.isfile(prev_ckpt):
                        try:
                            panel.append((f"round{rnd-1}",
                                          _load_weight_dict(prev_ckpt)))
                        except Exception as e:  # noqa: BLE001
                            log(f"arena: skip small snapshot {prev_ckpt}: {e}")
                big_items = []
                for fn in os.listdir(args.outdir):
                    m = re.match(r"big(\d+)\.npz$", fn)
                    if m and int(m.group(1)) < rnd:
                        big_items.append((int(m.group(1)), fn))
                big_items.sort(key=lambda t: t[0])
                for bid, fn in snapshot_tail(big_items, args.big_snapshot_rounds):
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
                        lead, ng = arena_match(net.state_dict(), opp_w, ag,
                                               args.arena_games, args.arena_sim,
                                               amax, F, args.hidden,
                                               net_kind, kernel_size,
                                               conv_layers,
                                               net.num_players,
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
                if total_games <= 0:
                    log("arena: skip gate (0 games); not accepting, not rolling back")
                elif (total_lead / total_games) >= args.arena_lead_threshold:
                    avg_lead = total_lead / total_games
                    best_weights = net.state_dict()
                    if cpu_net_finite(net):
                        save_cpu_net(net, os.path.join(args.outdir, "best.npz"))
                        best_note = "best.npz updated"
                    else:
                        best_note = "skip best.npz (non-finite weights)"
                    log(f"arena: new accepted (avg lead {avg_lead:+.4f} over "
                        f"{total_games} games vs {len(panel)} opponents on "
                        f"{len(arena_keys)} graphs); {best_note}")
                    _write_round_model_ckpts(args, net, rnd, log)
                else:
                    avg_lead = total_lead / total_games
                    net.load_state_dict(best_weights)
                    if cpu_net_finite(net):
                        save_cpu_net(net, os.path.join(args.outdir, "new.npz"))
                    else:
                        log("arena: skip new.npz rollback write (non-finite)")
                    log(f"arena: new REJECTED (avg lead {avg_lead:+.4f} over "
                        f"{total_games} games vs {len(panel)} opponents); "
                        f"rolled back to best")
                    for dest in quarantine_round_artifacts(args.outdir, rnd, ".npz"):
                        log(f"arena: moved rejected snapshot → {dest}")
                    restored = rollback_unused_after_reject(args.outdir, keys, rnd)
                    for skey, src_rnd in restored.items():
                        if src_rnd:
                            log(f"arena: graph {skey}: restored unused "
                                f"from round {src_rnd} snapshot")
                        else:
                            log(f"arena: graph {skey}: no prior buffer snapshot, "
                                f"cleared unused")

            if not args.infinite and rnd >= args.rounds:
                break
    finally:
        executor.shutdown(wait=True)

    lf.close()
    hrs = (time.time() - t_start) / 3600.0
    print(f"=== done: {cycle_no} cycles in {hrs:.2f} h ===")


if __name__ == "__main__":
    main()
