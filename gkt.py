"""GKT training helpers and C++ self-play / eval dispatch.

Search, legal moves, and scoring live in ``cpp/`` (``gkt_native``).
Network weights live in ``gkt_cpu.py`` / ``gkt_gpu.py``.
Docs: ``ref/rules.md``, ``ref/gomoku.md``, ``ref/algorithm.md``,
``ref/training_method.md``.

These helpers (sample packing, replay buffer, aux targets, eval matches) are
shared by distillation, cultivate2, and official cross-graph training.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import io
import json
import os
import random
import re
import sys
import time
import numpy as np

from graphs import DiGraph, is_k_in_row_rules
from grid_sym import SearchAugNet  # noqa: E402

BLACK = 1
WHITE = 2

SCORE_BINS = 21
SOFT_POLICY_TEMP = 4.0
W_OPP_POLICY = 0.25
W_SOFT_POLICY = 0.15
W_BELIEF_PDF = 0.30
W_BELIEF_CDF = 0.30
W_STDEV = 0.15
W_FUTURE = 0.25
N_EXTRA_FEATURES = 6  # lib1, lib2, lib3+, log1p(group), last-move, just-captured
SQUASH_JAC_FLOOR = 0.2  # min tanh/softplus Jacobian in the backward pass
REPLAY_ROUNDS = 10  # default --replay-rounds; 1 = do not reload unused.npz
BUFFER_DROP_FROM_ROUND = 5  # Go default: start dropping oldest unused this round
VALUE_ABS_WEIGHT_DEFAULT = 0.08  # SGD on stone-lead MSE; ~30/361
VALUE_RTO_WEIGHT_DEFAULT = 5.0   # SGD on search-scale lead/n
VALUE_CONS_WEIGHT_DEFAULT = 0.01  # (abs - n*rto)^2; k-row uses (abs - rto)^2
BUFFER_DROP_RATIO = 1.0  # FIXED (not a knob): auto drop = round(unused_net * this)
BUFFER_SNAPSHOT_ROUNDS = 5  # default --buffer-snapshot-rounds; 0 = off
MODEL_SNAPSHOT_ROUNDS = 5  # default --model-snapshot-rounds; 0 = keep all round*.pt/.npz
DIRICHLET_FRAC = 0.0  # eval/UI/Arena: pure visit-argmax, no root noise (greedy strength test).
# Self-play exploration noise lives in C++ (MCTSConfig.dirichlet_alpha), separate.


def cli_flag_set(*names: str) -> bool:
    """True if any of these flags appear on ``sys.argv`` (bare or ``--flag=``)."""
    flags = {"--" + str(n).lstrip("-").replace("_", "-") for n in names}
    return any(a.split("=", 1)[0] in flags for a in sys.argv[1:])


def apply_krow_train_defaults(args):
    """Fill Gomoku / Anti-Gomoku knobs only when the user did not pass them.

    Graph-Go argparse defaults match ``starter/train_*.bat``. k-in-a-row
    bats only pass ``--rules``; this keeps those bats short without
    changing Go defaults.
    """
    if not is_k_in_row_rules(getattr(args, "rules", "go")):
        return []
    specs = (
        ("sim", 800, ("sim",)),
        ("gpw", 16, ("gpw",)),
        ("steps", 4, ("steps",)),
        ("lr", 1e-3, ("lr",)),
        ("value_weight", 1.0, ("value-weight",)),
        ("value_rto_weight", 1.0, ("value-rto-weight",)),
        ("own_weight", 1.0, ("own-weight",)),
        ("value_cons_weight", VALUE_CONS_WEIGHT_DEFAULT, ("value-cons-weight",)),
        ("temperature", 1.0, ("temperature",)),
        ("arena", False, ("arena", "no-arena")),
        ("buffer_drop_from_round", "11", ("buffer-drop-from-round",)),
    )
    applied = []
    for attr, value, flags in specs:
        if cli_flag_set(*flags):
            continue
        setattr(args, attr, value)
        applied.append(attr)
    return applied


def feature_dim(num_players: int) -> int:
    """Occupancy one-hot plus graph-native extra channels. Must match C++."""
    return int(num_players) + N_EXTRA_FEATURES


def require_feature_dim(n_features, num_players) -> int:
    want = feature_dim(num_players)
    if int(n_features) != want:
        raise ValueError(
            f"n_features={n_features} must be num_players+{N_EXTRA_FEATURES}={want}")
    return want


def occupancy_with_empty(X: np.ndarray, num_players: int) -> np.ndarray:
    """Occupancy one-hot, derived empty, then extra channels. X is (..., F)."""
    X = np.asarray(X, dtype=np.float32)
    k = int(num_players)
    occ = X[..., :k]
    extra = X[..., k:]
    empty = np.clip(1.0 - occ.sum(axis=-1, keepdims=True), 0.0, 1.0)
    return np.concatenate([occ, empty, extra], axis=-1)


def score_lead(my: float, total: float, n: int, num_players: int) -> float:
    """Komi-free lead in [-1, 1]: (k * my - S) / ((k-1) * n). Search / Arena."""
    k = max(int(num_players), 2)
    denom = (k - 1) * max(int(n), 1)
    return float(np.clip((k * float(my) - float(total)) / denom, -1.0, 1.0))


def score_lead_abs(my: float, total: float, num_players: int) -> float:
    """Komi-free stone lead: (k * my - S) / (k-1). 2P is my - opp. No /n."""
    k = max(int(num_players), 2)
    return float((k * float(my) - float(total)) / (k - 1))


def sample_value_abs(sample) -> float:
    return float(sample[4])


def sample_value_rto(sample) -> float:
    if len(sample) < 13:
        raise ValueError("self-play sample must be a 13-tuple with value_rto")
    return float(sample[12])


def value_cons_scale(n_verts, mul_n: bool) -> float:
    """Go: identity is abs ≈ n·rto. k-in-a-row: both heads already in [-1, 1]."""
    return float(n_verts) if mul_n else 1.0


def set_value_loss_attrs(net, *, value_weight, value_rto_weight, own_weight,
                         value_cons_weight, rules: str = "go"):
    net.value_weight = float(value_weight)
    net.value_rto_weight = float(value_rto_weight)
    net.own_weight = float(own_weight)
    net.value_cons_weight = float(value_cons_weight)
    net.value_cons_mul_n = not is_k_in_row_rules(rules)


def curriculum_max_moves(n: int, rnd: int, min_moves: int = 40,
                         max_move_factor: float = 2.0,
                         curriculum_rounds: int = 100) -> int:
    """Nominal Graph-Go self-play length cap at 1-based training round ``rnd``.

    ``curriculum_rounds <= 0`` disables the curriculum and returns the full
    length ``int(n * max_move_factor) + min_moves`` from round 1 (the distilled
    / strong-start regime). Otherwise round 1 is ``min_moves`` and by
    ``curriculum_rounds`` it reaches the full cap. Each game then draws
    log-uniform in ``[cap/2, 2*cap]`` inside C++ (not here). k-in-a-row ignores
    this and uses cap ``n``.
    """
    min_moves = int(min_moves)
    max_cap = int(int(n) * float(max_move_factor)) + min_moves
    if int(curriculum_rounds) <= 0:
        return max_cap   # curriculum disabled: full length from round 1
    step = (max_cap - min_moves) / max(int(curriculum_rounds) - 1, 1)
    return min(min_moves + int(round(step * (int(rnd) - 1))), max_cap)


def shuffle_graph_keys(keys, rnd) -> List[str]:
    """Deterministic graph order for training round ``rnd``.

    Uses a *private* ``random.Random(rnd)``, never the process-global
    ``random`` module (CPU SGD samples from that). Same key list + same
    ``rnd`` ⇒ same permutation, so resume can skip finished graphs without
    repeating or dropping any. This is the one training RNG that must be
    reproducible; it is for visit-uniformity across graphs, not for
    replicating a run.
    """
    order = [str(k) for k in keys]
    random.Random(int(rnd)).shuffle(order)
    return order


def resume_graph_cursor(keys, last_round, last_key, arena: bool):
    """``(start_rnd, start_idx)`` after finishing ``last_key`` in ``last_round``.

    ``start_idx == len(shuffled)`` means the round's graphs are done: with
    Arena on, re-enter the gate (empty graph list); otherwise start the
    next round. If ``last_key`` is not in this run's set, skip to the next
    round (caller should log).
    """
    rk = shuffle_graph_keys(keys, last_round)
    lk = str(last_key)
    if lk not in rk:
        return int(last_round) + 1, 0, rk, False
    idx = rk.index(lk)
    if idx < len(rk) - 1:
        return int(last_round), idx + 1, rk, True
    if arena:
        return int(last_round), len(rk), rk, True
    return int(last_round) + 1, 0, rk, True


def forbid_pass_after(n: int, max_moves: int) -> int:
    """Self-play MCTS drops pass for this many plies if a stone move exists.

    Matches ``cpp/src/selfplay.cpp``: ``min(cap, max(n/8, 16))``.
    """
    return min(int(max_moves), max(int(n) // 8, 16))


def soft_policy_target(policy: np.ndarray, mask: np.ndarray,
                       temp: float = SOFT_POLICY_TEMP) -> np.ndarray:
    """KataGo auxiliary: policy^(1/T), re-normalized on legal moves."""
    p = np.asarray(policy, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    q = np.where(m > 0, np.power(np.clip(p, 1e-8, 1.0), 1.0 / max(temp, 1e-6)), 0.0)
    s = float(q.sum())
    if s <= 0:
        return p
    return (q / s).astype(np.float32)


def lead_belief(lead: float, n_bins: int = SCORE_BINS) -> np.ndarray:
    """Soft histogram of score-lead / n on [-1, 1] (n-invariant score pdf)."""
    t = np.zeros(n_bins, dtype=np.float32)
    x = (float(np.clip(lead, -1.0, 1.0)) + 1.0) * 0.5 * (n_bins - 1)
    i = int(np.floor(x))
    i = min(max(i, 0), n_bins - 2)
    f = float(x - i)
    t[i] = 1.0 - f
    t[i + 1] = f
    return t


def progress_append(path: Optional[str], msg: str, fsync: bool = True) -> None:
    """Append one flushed line to the training heartbeat file."""
    if not path:
        return
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
    except OSError:
        pass


def worker_progress_path(progress_file: Optional[str], worker_id: int) -> Optional[str]:
    """`progress.txt` → `progress.w0.txt` (one file per worker)."""
    if not progress_file:
        return None
    directory = os.path.dirname(os.path.abspath(progress_file))
    stem, ext = os.path.splitext(os.path.basename(progress_file))
    if not ext:
        ext = ".txt"
    return os.path.join(directory, f"{stem}.w{int(worker_id)}{ext}")


def reset_worker_progress(progress_file: Optional[str], n_workers: int):
    """Truncate `progress.w0.txt` … `progress.w{n-1}.txt` for a new graph cycle."""
    paths = []
    if not progress_file or n_workers <= 0:
        return paths
    for i in range(int(n_workers)):
        p = worker_progress_path(progress_file, i)
        paths.append(p)
        try:
            open(p, "w", encoding="utf-8").close()
        except OSError:
            pass
    try:
        open(progress_file, "w", encoding="utf-8").close()
    except OSError:
        pass
    return paths


def make_move_heartbeat(path: Optional[str], prefix: str):
    """C++ callback: game_start / move / batch (after each MCTS eval) / game_end.

    `batch` lines are throttled to ~1s (plus the last batch of a search) so
    the heartbeat file does not fsync hundreds of times per ply.
    """
    t0 = time.time()
    last_batch = [0.0]

    def hb(event, move, max_moves, n_sim, batch_done=0, batch_total=0):
        dt = time.time() - t0
        do_sync = event != "batch"
        extra = ""
        if event == "batch":
            now = time.time()
            finished = batch_total > 0 and batch_done >= batch_total
            if not finished and (now - last_batch[0]) < 1.0:
                return
            last_batch[0] = now
            extra = f" batch={batch_done}/{batch_total} rem={n_sim}"
        else:
            extra = f" sim={n_sim}"
        progress_append(
            path,
            f"{prefix} {event} ply={move}/{max_moves}{extra} t={dt:.1f}s",
            fsync=do_sync)
    return hb


def aux_from_sample(sample) -> Dict:
    """KataGo-style aux fields from a 13-tuple self-play sample."""
    return {
        "q": sample[6],
        "opp": sample[7],
        "opp_w": sample[8],
        "future": sample[9],
        "lead": sample[10],
        "weight": float(sample[11]),
        "value_rto": sample_value_rto(sample),
    }


def stack_aux(batch) -> Dict:
    """Stack aux targets from a list of 13-tuple samples."""
    return {
        "q": np.array([s[6] for s in batch], dtype=np.float32),
        "opp": np.stack([s[7] for s in batch]),
        "opp_w": np.array([s[8] for s in batch], dtype=np.float32),
        "future": np.stack([s[9] for s in batch]),
        "lead": np.array([s[10] for s in batch], dtype=np.float32),
        "weight": np.array([s[11] for s in batch], dtype=np.float32),
        "value_rto": np.array([sample_value_rto(s) for s in batch], dtype=np.float32),
    }


def pack_samples(samples: List[Tuple], graph_key: str,
                 extra: Optional[Dict] = None) -> bytes:
    """Compress 13-tuple self-play rows to an npz blob (one graph)."""
    if not samples:
        raise ValueError("no samples to pack")
    payload = {
        "X": np.stack([s[0] for s in samples]),
        "mask": np.stack([s[1] for s in samples]),
        "policy": np.stack([s[2] for s in samples]),
        "me": np.array([s[3] for s in samples], dtype=np.int32),
        "value": np.array([s[4] for s in samples], dtype=np.float32),
        "own": np.stack([s[5] for s in samples]),
        "q": np.array([s[6] for s in samples], dtype=np.float32),
        "opp": np.stack([s[7] for s in samples]),
        "opp_w": np.array([s[8] for s in samples], dtype=np.float32),
        "future": np.stack([s[9] for s in samples]),
        "lead": np.array([s[10] for s in samples], dtype=np.float32),
        "weight": np.array([s[11] for s in samples], dtype=np.float32),
        "value_rto": np.array([sample_value_rto(s) for s in samples],
                             dtype=np.float32),
        "_graph": np.asarray(graph_key),
    }
    if extra:
        for k, v in extra.items():
            payload["_" + k] = np.asarray(v)
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def unpack_samples(data) -> Tuple[str, List[Tuple]]:
    """Read `pack_samples` bytes or a path."""
    if isinstance(data, (str, os.PathLike)):
        z = np.load(data, allow_pickle=False)
    else:
        z = np.load(io.BytesIO(data), allow_pickle=False)
    try:
        graph = str(np.asarray(z["_graph"]).item())
        # Materialize every array ONCE. NpzFile.__getitem__ re-decompresses the
        # whole array on each access (numpy >= 2.x does not cache), so the old
        # `z["X"][i]` loop re-decompressed X ~n times and blew peak memory
        # (measured: 27 GB for a 5 MB buffer). Grab the arrays first, then slice
        # the in-memory views.
        X = z["X"]
        mask = z["mask"]
        policy = z["policy"]
        me = z["me"]
        value = z["value"]
        own = z["own"]
        q = z["q"]
        opp = z["opp"]
        opp_w = z["opp_w"]
        future = z["future"]
        lead = z["lead"]
        weight = z["weight"]
        if "value_rto" not in z:
            raise ValueError("npz missing value_rto; 12-tuple buffers are not supported")
        value_rto = z["value_rto"]
        n = X.shape[0]
        samples = []
        for i in range(n):
            samples.append((
                X[i], mask[i], policy[i], int(me[i]),
                float(value[i]), own[i], float(q[i]),
                opp[i], float(opp_w[i]), future[i],
                float(lead[i]), float(weight[i]),
                float(value_rto[i]),
            ))
        return graph, samples
    finally:
        z.close()


def _replay_key(key: str) -> str:
    return str(key).replace("/", "_").replace("\\", "_")


def unused_buffer_path(outdir: str, key: str) -> str:
    return os.path.join(outdir, "replay", _replay_key(key), "unused.npz")


def load_unused_buffer(outdir: str, key: str, n_rounds: int = REPLAY_ROUNDS,
                       cap: Optional[int] = None) -> List[Tuple]:
    """Unused self-play for this graph. ``n_rounds <= 1`` skips the disk queue."""
    if int(n_rounds) <= 1:
        return []
    path = unused_buffer_path(outdir, key)
    rows: List[Tuple] = []
    if os.path.isfile(path):
        try:
            _, rows = unpack_samples(path)
        except (OSError, ValueError, KeyError, TypeError):
            rows = []
    if cap is not None and int(cap) > 0 and len(rows) > int(cap):
        rows = rows[-int(cap):]
    return rows


def auto_buffer_drop(unused_net: int) -> int:
    """Drop this round's *unused* new samples so the queue stays constant.

    Each round self-play produces ``new_samples`` fresh games and SGD then
    consumes ``steps*batch_size`` of the mixed queue (removing each successful
    batch). The surplus that survived SGD is ``unused_net = new_samples -
    consumed`` — equivalently ``len(unused_after_sgd) - len(replay_before)``.
    Dropping exactly that many keeps the queue at a constant size (neither
    draining nor growing), instead of draining it by the consumed count each
    round (which is what dropping the full ``new_samples`` did).

    The ratio is FIXED at ``BUFFER_DROP_RATIO`` (1.0): drop = round(unused_net).
    How many generations are retained is governed by ``--buffer-drop-from-
    round`` (when dropping starts), NOT by this ratio — that round is the
    per-graph knob.
    """
    return int(round(max(0, int(unused_net)) * float(BUFFER_DROP_RATIO)))


def parse_from_round_map(spec, default=BUFFER_DROP_FROM_ROUND):
    """Parse ``--buffer-drop-from-round`` into a per-key lookup.

    ``spec`` is either a bare integer (``"11"``) applied to every graph, or a
    comma-separated ``key=round`` list (``"0=20,R1=15"``) overriding specific
    keys; unlisted keys fall back to ``default``.

    Returns ``(global_default, overrides)`` where ``global_default`` is an int
    and ``overrides`` maps key -> int.
    """
    global_default = max(1, int(default))
    overrides = {}
    spec = (spec or "").strip()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            overrides[k.strip()] = max(1, int(v.strip()))
        else:
            global_default = max(1, int(part))
    return global_default, overrides


def drop_oldest_buffer(samples: List[Tuple], n: int) -> Tuple[List[Tuple], int]:
    """Drop the oldest ``n`` unused samples (front of the per-graph queue)."""
    n = int(n)
    if n <= 0 or not samples:
        return list(samples), 0
    dropped = min(n, len(samples))
    return list(samples)[dropped:], dropped


def save_unused_buffer(outdir: str, key: str, samples: List[Tuple],
                       cap: Optional[int] = None) -> Optional[str]:
    """Write the unused queue. Empty list deletes the file."""
    path = unused_buffer_path(outdir, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cap is not None and int(cap) > 0 and len(samples) > int(cap):
        samples = samples[-int(cap):]
    if not samples:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(pack_samples(samples, str(key), extra={"unused": 1}))
    os.replace(tmp, path)
    return path


def buffer_snapshot_path(outdir: str, key: str, rnd: int) -> str:
    """Path of a per-round snapshot of this graph's unused buffer.

    Lives in ``snapshots/`` so it never collides with ``unused.npz``.
    """
    return os.path.join(outdir, "replay", _replay_key(key), "snapshots",
                        "round%06d.npz" % int(rnd))


def save_buffer_snapshot(outdir: str, key: str, rnd: int,
                         samples: List[Tuple]) -> Optional[str]:
    """Write a rollback snapshot of this graph's unused buffer for ``rnd``.

    ``unused.npz`` only ever holds the *current* queue (it is overwritten each
    visit, and oldest samples are dropped from it), so a corrupt or
    over-dropped ``unused.npz`` cannot be reconstructed from ``round*.pt``
    (weights only, no samples). These snapshots give the per-round buffer
    state needed to roll back and re-run from a round boundary.
    """
    if not samples:
        return None
    path = buffer_snapshot_path(outdir, key, rnd)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(pack_samples(samples, str(key), extra={"snapshot": 1}))
    os.replace(tmp, path)
    return path


def prune_buffer_snapshots(outdir: str, key: str, keep: int) -> int:
    """Keep only the most recent ``keep`` buffer snapshots for this graph."""
    folder = os.path.join(outdir, "replay", _replay_key(key), "snapshots")
    if not os.path.isdir(folder):
        return 0
    names = sorted(n for n in os.listdir(folder)
                   if re.match(r"round\d+\.npz$", n))
    keep = int(keep)
    doomed = names if keep <= 0 else names[:-keep]
    removed = 0
    for name in doomed:
        try:
            os.remove(os.path.join(folder, name))
            removed += 1
        except OSError:
            pass
    return removed


def load_buffer_snapshot(outdir: str, key: str, rnd: int) -> List[Tuple]:
    """Read one per-round unused-buffer snapshot, or ``[]`` if missing."""
    path = buffer_snapshot_path(outdir, key, rnd)
    if not os.path.isfile(path):
        return []
    try:
        _, rows = unpack_samples(path)
        return rows
    except (OSError, ValueError, KeyError, TypeError):
        return []


def restore_unused_from_prior_snapshot(
        outdir: str, key: str, rnd: int,
        cap: Optional[int] = None) -> int:
    """Replace ``unused.npz`` with the latest snapshot strictly before ``rnd``.

    Returns the snapshot round loaded, or 0 if the queue was cleared.
    """
    folder = os.path.join(outdir, "replay", _replay_key(key), "snapshots")
    prior = []
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            m = re.match(r"round(\d+)\.npz$", name)
            if m and int(m.group(1)) < int(rnd):
                prior.append(int(m.group(1)))
    if not prior:
        save_unused_buffer(outdir, key, [])
        return 0
    src_rnd = max(prior)
    rows = load_buffer_snapshot(outdir, key, src_rnd)
    save_unused_buffer(outdir, key, rows, cap=cap)
    return src_rnd


def drop_buffer_snapshot(outdir: str, key: str, rnd: int) -> bool:
    """Delete this round's buffer snapshot (rejected-net samples)."""
    path = buffer_snapshot_path(outdir, key, rnd)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def quarantine_ckpt(outdir: str, filename: str) -> Optional[str]:
    """Move ``outdir/filename`` into ``outdir/rejected/`` if it exists."""
    src = os.path.join(outdir, filename)
    if not os.path.isfile(src):
        return None
    dest_dir = os.path.join(outdir, "rejected")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.isfile(dest):
        stem, ext = os.path.splitext(filename)
        dest = os.path.join(dest_dir, f"{stem}_{int(time.time())}{ext}")
    os.replace(src, dest)
    return dest


def quarantine_round_artifacts(outdir: str, rnd: int, ext: str) -> List[str]:
    """Move this round's ``roundN`` / ``bigN`` checkpoints into ``rejected/``."""
    ext = "." + str(ext).lstrip(".")
    moved = []
    for prefix in ("round", "big"):
        dest = quarantine_ckpt(outdir, f"{prefix}{int(rnd)}{ext}")
        if dest:
            moved.append(dest)
    return moved


def rollback_unused_after_reject(
        outdir: str, keys, rnd: int,
        cap: Optional[int] = None) -> Dict[str, int]:
    """Restore each graph's unused queue from the last accepted snapshot.

    Also drops this round's buffer snapshot so rejected-net samples are not
    the next rollback point. Values are the snapshot round loaded (0 = cleared).
    """
    restored = {}
    for key in keys:
        src = restore_unused_from_prior_snapshot(outdir, key, rnd, cap=cap)
        drop_buffer_snapshot(outdir, key, rnd)
        restored[str(key)] = src
    return restored


def _prune_numbered_snapshots(outdir: str, prefix: str, keep: int,
                               ext: str) -> int:
    """Keep the most recent ``keep`` ``<prefix><n>``+``ext`` files in ``outdir``.

    Sorts by the *numeric* id (``round10`` must come after ``round9``, so a
    plain string sort — which puts ``round10`` < ``round2`` — is wrong).
    ``keep <= 0`` deletes every matching checkpoint.
    """
    ext = "." + str(ext).lstrip(".")
    if not os.path.isdir(outdir):
        return 0
    pat = re.compile(re.escape(prefix) + r"(\d+)\." + re.escape(ext[1:]) + r"$")
    items = []
    for n in os.listdir(outdir):
        m = pat.match(n)
        if m:
            items.append((int(m.group(1)), n))
    items.sort(key=lambda t: t[0])
    names = [n for _, n in items]
    keep = int(keep)
    doomed = names if keep <= 0 else names[:-keep]
    removed = 0
    for name in doomed:
        try:
            os.remove(os.path.join(outdir, name))
            removed += 1
        except OSError:
            pass
    return removed


def prune_model_snapshots(outdir: str, keep: int, ext: str = ".pt") -> int:
    """Keep only the most recent ``keep`` per-round model checkpoints.

    Matches ``round\\d+`` + ``ext`` at the top level of ``outdir`` only, so
    ``new.pt`` / ``best.pt`` / ``big*.pt`` (and their ``.npz`` counterparts)
    are never touched. ``keep <= 0`` deletes every per-round checkpoint (they
    are pure history once ``new.*``/``best.*`` hold the live/best weights).
    """
    return _prune_numbered_snapshots(outdir, "round", keep, ext)


def snapshot_tail(items, keep: int):
    """Last ``keep`` items of a sequence already in order.

    ``keep <= 0`` is empty. Do not use ``xs[-0:]``: in Python that is ``xs[0:]``.
    """
    k = int(keep)
    if k <= 0:
        return []
    return list(items)[-k:]


def prune_big_snapshots(outdir: str, keep: int, ext: str = ".pt") -> int:
    """Keep only the most recent ``keep`` long-horizon big checkpoints.

    Matches ``big\\d+`` + ``ext`` at the top level of ``outdir`` only (the
    every-N-round checkpoints used as Arena opponents), independent of the
    small per-round ``round*.pt`` snapshots. ``keep <= 0`` deletes them all.
    """
    return _prune_numbered_snapshots(outdir, "big", keep, ext)


def play_eval_match(nets, graph: DiGraph,
                    n_simulations: int, max_moves: int,
                    dirichlet_frac: float = DIRICHLET_FRAC, batch_size: int = 64,
                    rules: str = "go", win_length: int = 5,
                    heartbeat=None
                    ) -> Dict[int, float]:
    """Empty-board game. `nets` maps side (1..k) to a net or None (uniform).

    Search uses a fixed ``n_simulations`` (no log-uniform draw). Root Dirichlet
    defaults to 0 (``DIRICHLET_FRAC``) so the move is a pure visit-argmax with
    no temperature sample — a deterministic strength test.

    ``heartbeat``, if given, matches the self-play callback signature
    ``hb(event, move, max_moves, n_sim, batch_done, batch_total)``. It fires
    game_start / move / batch (per MCTS eval batch, via the native search
    on_batch hook) / game_end, so Arena can stream a per-move / per-batch
    heartbeat exactly like a self-play worker.
    """
    from gkt_cpp import require_native, py_graph_to_native
    native = require_native()
    k = len(nets)
    ng = py_graph_to_native(graph)
    game = native.Game(ng, k, rules=rules, win_length=int(win_length))
    if heartbeat:
        heartbeat("game_start", 0, max_moves, n_simulations, 0, 0)
    for _ in range(max_moves):
        if game.game_over():
            break
        me = game.position.to_move
        net = nets[me]
        if net is None:
            legal = list(game.legal_moves())
            if not legal:
                break
            action = legal[random.randrange(len(legal))]
        else:
            wrapped = SearchAugNet(net, graph)
            wrapped.begin_search()
            ply = int(game.position.move_no)
            if heartbeat:
                heartbeat("move", ply, max_moves, n_simulations, 0, 0)
            on_batch = None
            if heartbeat:
                on_batch = (lambda done, total, rem, _ply=ply:
                            heartbeat("batch", _ply, max_moves, rem,
                                      done, total))
            legal, counts, _ = native.search(
                game.position, wrapped, max(1, n_simulations),
                min(batch_size, max(1, n_simulations)),
                dirichlet_frac, 1.5, random.getrandbits(32), False,
                on_batch)
            if not legal:
                break
            action = legal[int(np.argmax(counts))]
        played = game.play(action)
        if not played.legal:
            raise RuntimeError(
                f"eval play illegal: {played.reason} action={action}")
    if heartbeat:
        heartbeat("game_end", int(game.position.move_no), max_moves,
                  n_simulations, 0, 0)
    scores, _ = game.finalize()
    out = {int(s): float(scores[s]) for s in game.sides}
    for i, side in enumerate(game.sides):
        out[side] += game.komi_schedule[i]
    return out


def standing_unit(scores: Dict[int, float], seat: int, sides) -> float:
    """Map `seat`'s standing to [0, 1]; equal strength averages 0.5."""
    k = len(sides)
    if k <= 1:
        return 1.0
    my = float(scores.get(seat, 0.0))
    better = sum(1 for s in sides if float(scores.get(s, 0.0)) > my)
    tied = sum(1 for s in sides if float(scores.get(s, 0.0)) == my)
    rank = better + (tied + 1) / 2.0
    return (k - rank) / (k - 1)


def write_graph_round_summary(outdir: str, rec: Dict) -> str:
    """Append this graph's round stats to ``progress/summary_{key}.json``.

    The file is a list, one object per visit (normally one per round).
    A lone JSON object is read as a one-row list. Re-writing the same
    ``cycle`` replaces the last row instead of duplicating.
    """
    folder = os.path.join(outdir, "progress")
    os.makedirs(folder, exist_ok=True)
    key = str(rec.get("key", "unknown")).replace("/", "_").replace("\\", "_")
    path = os.path.join(folder, f"summary_{key}.json")
    rows: List[Dict] = []
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                rows = [data]
        except (OSError, ValueError, json.JSONDecodeError):
            rows = []
    if rows and rows[-1].get("cycle") == rec.get("cycle"):
        rows[-1] = rec
    else:
        rows.append(rec)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def play_eval_game(net_black, net_white, graph: DiGraph,
                   n_simulations: int, max_moves: int,
                   dirichlet_frac: float = DIRICHLET_FRAC, batch_size: int = 64,
                   rules: str = "go", win_length: int = 5) -> float:
    """Two-player wrapper: Black's payoff in {1.0, 0.5, 0.0}."""
    scores = play_eval_match({BLACK: net_black, WHITE: net_white}, graph,
                             n_simulations, max_moves, dirichlet_frac,
                             batch_size, rules=rules, win_length=win_length)
    return standing_unit(scores, BLACK, (BLACK, WHITE))


class GktSelfPlay:
    def __init__(self, graph: DiGraph, net,
                 n_simulations: int = 800, temperature: float = 1.0,
                 max_moves: int = None,
                 batch_size: int = 32, log_fn=print,
                 q_lambda: float = 0.5,
                 randomize_sim: bool = True,
                 relabel: bool = True,
                 rules: str = "go", win_length: int = 5):
        self.graph = graph
        self.net = net
        self.n_sim = n_simulations
        self.batch_size = batch_size
        self.temperature = temperature
        self.log = log_fn
        self.q_lambda = q_lambda
        self.randomize_sim = randomize_sim
        self.rules = str(rules)
        self.win_length = int(win_length)
        if is_k_in_row_rules(self.rules):
            self.max_moves = len(graph.vertices)
        else:
            self.max_moves = max_moves or (len(graph.vertices) * 2 + 40)
        nf = getattr(net, "n_features", None)
        if nf is None:
            nf = getattr(net, "F", None)
        if nf is None:
            raise TypeError(
                f"{type(net).__name__} has neither n_features nor F")
        nf = int(nf)
        k = int(net.num_players)
        require_feature_dim(nf, k)
        self.num_players = k
        # Index-superstition defense: relabel vertices on EVERY leaf eval the
        # net reads (fresh perm per predict_batch), so self-play data generation
        # cannot latch onto a fixed vertex numbering.
        if relabel:
            self.net = SearchAugNet(net, graph, fresh_each_batch=True)

    def play_one_game(self, heartbeat=None):
        """Self-play samples: 13-tuple from C++ `play_one_game`."""
        from gkt_cpp import play_one_game as cpp_play
        return cpp_play(
            self.graph, self.net, n_simulations=self.n_sim,
            temperature=self.temperature, max_moves=self.max_moves,
            batch_size=self.batch_size,
            q_lambda=self.q_lambda, randomize_sim=self.randomize_sim,
            seed=random.randrange(1 << 30), heartbeat=heartbeat,
            rules=self.rules, win_length=self.win_length)
