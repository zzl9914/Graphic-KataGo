"""KataGo distillation: supervised pretrain of a graph-agnostic net.

Stage 1 of the Go pipeline (``ref/training_method.md``): pretrain against
KataGo teacher labels so the value head gets a real signal from step one,
instead of collapsing under from-zero self-play on a small GPU.

Output is the distilled checkpoint (「基础培养」) on 19x19. Next: cultivate2,
then official cross-graph training. From-zero remains only as M0.

Supported architectures (``--net``):

    gnn     GPU  GnnPolicyValueNet        (gkt_gpu.py)   -> new.pt
    2dcnn   GPU  Cnn2dPolicyValueNet      (gkt_gpu.py)   -> new.pt
    mlp     CPU  MlpPolicyValueNet        (gkt_cpu.py)   -> new.npz
    1dcnn   CPU  Cnn1dPolicyValueNet      (gkt_cpu.py)   -> new.npz

Only THREE heads get distilled targets, because those are the heads KataGo has
a trustworthy, well-defined signal for:

    policy     KataGo MCTS visit distribution (soft CE)
    value      KataGo score lead (MSE, mapped to score_lead in [-1, 1])
    ownership  KataGo ownership (MSE, mapped to mover-relative [-1, 1])

The aux heads (opp / soft / belief / stdev / future) are left at random init;
they train later during self-play. Distillation runs three stages of
``--epochs`` each (default 10+10+10). Stage 1 is joint
``policy + 30*value + 5*own`` with **no freeze**. Stages 2–3 freeze the
trunk and train own / value alone (Adam lr ×25 / ×100, unweighted MSE).

Input (JSONL, one position per line, already in *gkt vertex order* — see
``gen_katago_data.py`` / ``distill_katago.py`` for the KataGo -> gkt conversion):

    {"to_move": 1, "board": [0/1/2 x n], "policy": [x (n+1)],
     "scoreLead": 3.14, "ownership": [x n]}

Output: a standard gkt checkpoint ``<outdir>/new.pt`` (GPU) or ``<outdir>/new.npz``
(CPU) that the self-play trainers load with ``--resume``.

Run:  python distill.py --net gnn --data data.jsonl --graph-key 0 --outdir ../base/gnn
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from graphs import get_builtin
import gkt_cpp
from gkt import progress_append  # noqa: E402
from gkt_gpu import (  # noqa: E402
    make_net, save_net, _masked_ce, _permute_adj_batch,
    gpu_net_finite, gpu_net_type, gpu_net_label,
)
from gkt_cpu import (  # noqa: E402
    MlpPolicyValueNet, Cnn1dPolicyValueNet, save_cpu_net, NumpyAdam,
)
from grid_sym import augment_vertex_batch, augment_graph_batch  # noqa: E402

GPU_NETS = ("gnn", "2dcnn")
DISTILL_STAGES = ("policy", "own", "value")
STAGE_LR_MULT = {"policy": 1.0, "own": 25.0, "value": 100.0}


def build_sample(rec, graph, ng, native):
    """One JSONL record -> the 12-tuple sample the GNN expects.

    Feature vectors are computed by the *native engine* (``extract_features``)
    so distillation uses exactly the same F=8 graph-agnostic encoding as
    self-play. ``policy``/``ownership`` arrive in gkt vertex order (row-major
    A_{i*n+j}); pass sits at index ``n``.
    """
    n = len(graph.vertices)
    board = np.asarray(rec["board"], dtype=np.int8).reshape(-1)
    if board.shape[0] != n:
        raise ValueError(f"board length {board.shape[0]} != n={n}")
    to_move = int(rec["to_move"])

    game = native.Game(ng, 2, [], to_move, board, "go", 5)
    pos = game.position
    me = int(pos.to_move)

    X = np.asarray(native.extract_features(pos, me), dtype=np.float32)  # (n, F)

    legal = list(game.legal_moves())  # GraphGo: empty vertices + pass (n)
    mask = np.zeros(n + 1, dtype=np.float32)
    for a in legal:
        if 0 <= int(a) <= n:
            mask[int(a)] = 1.0

    pol = np.asarray(rec["policy"], dtype=np.float32).reshape(-1)
    if pol.shape[0] != n + 1:
        raise ValueError(f"policy length {pol.shape[0]} != n+1={n + 1}")
    pol = pol * mask              # zero anything illegal, keep pass if legal
    s = float(pol.sum())
    if s > 0:
        pol = pol / s

    # KataGo scoreLead (mover-perspective points) -> score_lead = (my-opp)/n.
    # KataGo "points" ~ stone-count lead here, so /n lands in [-1, 1] and
    # matches the self-play value target's scale exactly.
    value = float(np.clip(float(rec["scoreLead"]) / max(n, 1), -1.0, 1.0))

    # KataGo ownership is black-relative [-1, 1]; self-play own is
    # mover-relative (mine - (rest - mine)). Flip sign when white to move.
    own = np.asarray(rec["ownership"], dtype=np.float32).reshape(-1)
    if own.shape[0] != n:
        raise ValueError(f"ownership length {own.shape[0]} != n={n}")
    own = own * (1.0 if me == 1 else -1.0)

    # Aux targets: no trustworthy KataGo signal, so neutral values. Distill
    # (GPU and CPU) trains only policy / value / own; aux stays at init.
    q = value
    opp = np.zeros(n + 1, dtype=np.float32)
    opp_w = 0.0
    future = own.copy()
    lead = value
    weight = 1.0

    return (X, mask, pol, me, value, own, q, opp, opp_w, future, lead, weight)


def build_batch(records, graph, ng, native):
    """Stack a list of records into the 5 tensors distillation actually uses.

    Builds samples on the fly (native ``extract_features`` is ~0.000s/sample)
    so the full dataset's feature tensors are never materialized in memory —
    only the lightweight JSON records stay resident.
    """
    xs, ms, ps, vs, os_ = [], [], [], [], []
    for rec in records:
        X, mask, pol, _me, value, own = build_sample(rec, graph, ng, native)[:6]
        xs.append(X)
        ms.append(mask)
        ps.append(pol)
        vs.append(value)
        os_.append(own)
    return (np.stack(xs), np.stack(ms), np.stack(ps),
            np.asarray(vs, dtype=np.float32), np.stack(os_))


def _aug_train_batch(net, graph, X, mask, pol, own, enabled=True):
    """Train-only relabel / board symmetry. Val must not call this.

    GNN / MLP / 1DCNN: random S_n (GNN also permutes adj so the graph is the
    same). 2DCNN: D4 / Klein / torus shift via ``augment_graph_batch``. Value
    is a scalar and is not permuted. ``enabled=False`` is identity (curriculum:
    learn the original numbering first).
    """
    if not enabled:
        return X, mask, pol, own, None, None
    nt = getattr(net, "net_type", "")
    if nt == "2dcnn":
        X, mask, pol, own, _, _ = augment_graph_batch(
            X, mask, pol, graph, ownership=own)
        return X, mask, pol, own, None, None
    X, mask, pol, own, _, _, perms = augment_vertex_batch(
        X, mask, pol, ownership=own)
    if nt == "gnn":
        perms_t = torch.from_numpy(np.asarray(perms, np.int64)).long().to(net.device)
        adj_in = _permute_adj_batch(net.adj_in, perms_t)
        adj_out = _permute_adj_batch(net.adj_out, perms_t)
        return X, mask, pol, own, adj_in, adj_out
    return X, mask, pol, own, None, None


def distill_train_on_batch(net, X, mask, policy, value, own,
                           stage="policy", adj_in=None, adj_out=None):
    """One supervised Adam step on a single distill stage (GPU).

    All three losses are computed for the log. Stage 1 (``policy``) backprops
    ``pl + value_weight*vl + own_weight*ol`` through every parameter. Own /
    value stages backprop one unweighted head with the trunk frozen.
    """
    X_t = torch.from_numpy(np.asarray(X, np.float32)).float().to(net.device)
    m_t = torch.from_numpy(np.asarray(mask) > 0).bool().to(net.device)
    p_t = torch.from_numpy(np.asarray(policy, np.float32)).float().to(net.device)
    v_t = torch.from_numpy(np.asarray(value, np.float32)).float().to(net.device)
    o_t = torch.from_numpy(np.asarray(own, np.float32)).float().to(net.device)

    if stage == "policy":
        net.train()
    else:
        net.eval()
    if adj_in is not None:
        out = net.forward_adj(X_t, adj_in, adj_out)
    else:
        out = net.forward(X_t)
    policy_loss = _masked_ce(out["policy"], p_t, m_t).mean()
    value_loss = (out["value"].squeeze(-1) - v_t).pow(2).mean()
    own_loss = (out["own"] - o_t).pow(2).mean(dim=-1).mean()
    vw = float(getattr(net, "value_weight", 1.0))
    ow = float(getattr(net, "own_weight", 1.0))
    if stage == "policy":
        loss = policy_loss + vw * value_loss + ow * own_loss
    elif stage == "own":
        loss = own_loss
    elif stage == "value":
        loss = value_loss
    else:
        raise ValueError(f"unknown distill stage {stage!r}")

    net.optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return (float("nan"),) * 4
    loss.backward()
    trainable = [p for p in net.parameters() if p.requires_grad]
    if trainable:
        nn.utils.clip_grad_norm_(trainable, 1.0)
    net.optimizer.step()
    return tuple(float(x.detach().item())
                 for x in (policy_loss, value_loss, own_loss, loss))


@torch.no_grad()
def eval_batch(net, X, mask, policy, value, own):
    X_t = torch.from_numpy(np.asarray(X, np.float32)).float().to(net.device)
    m_t = torch.from_numpy(np.asarray(mask) > 0).bool().to(net.device)
    p_t = torch.from_numpy(np.asarray(policy, np.float32)).float().to(net.device)
    v_t = torch.from_numpy(np.asarray(value, np.float32)).float().to(net.device)
    o_t = torch.from_numpy(np.asarray(own, np.float32)).float().to(net.device)
    net.eval()
    out = net.forward(X_t)
    pl = _masked_ce(out["policy"], p_t, m_t).mean()
    vl = (out["value"].squeeze(-1) - v_t).pow(2).mean()
    ol = (out["own"] - o_t).pow(2).mean(dim=-1).mean()
    return float(pl.item()), float(vl.item()), float(ol.item())


def _batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _progress_path(outdir):
    return os.path.abspath(os.path.join(outdir, "progress.txt"))


class _DistillHeartbeat:
    """``outdir/progress.txt``: one line per train/val step; truncated each epoch.

    ``fsync`` is throttled to ~1s (plus epoch/phase boundaries) so a fast MLP
    epoch does not fsync thousands of times.
    """

    def __init__(self, path, tag):
        self.path = path
        self.tag = tag
        self.t0 = time.time()
        self._last_sync = 0.0

    def line(self, body, fsync=False):
        now = time.time()
        if fsync or (now - self._last_sync) >= 1.0:
            fsync = True
            self._last_sync = now
        dt = now - self.t0
        progress_append(
            self.path,
            f"distill {self.tag} {body} t={dt:.1f}s",
            fsync=fsync)

    def step(self, phase, i, n_steps, n_batch, pl, vl, ol):
        self.line(
            f"{phase} step={i}/{n_steps} n={n_batch} "
            f"pl={pl:.4f} vl={vl:.4f} ol={ol:.4f}",
            fsync=(i >= n_steps))


def _begin_epoch_hb(outdir, tag, n_train, n_val, batch_size, extra=""):
    path = _progress_path(outdir)
    try:
        open(path, "w", encoding="utf-8").close()
    except OSError:
        pass
    hb = _DistillHeartbeat(path, tag)
    bs = max(int(batch_size), 1)
    n_tr = max(1, math.ceil(n_train / bs))
    n_va = max(1, math.ceil(n_val / bs))
    more = f" {extra}" if extra else ""
    hb.line(
        f"epoch_start train_samples={n_train} train_steps={n_tr} "
        f"val_samples={n_val} val_steps={n_va} batch={bs}{more}",
        fsync=True)
    return hb, n_tr, n_va


def _latest_round(outdir, ext):
    """Scan outdir for ``round{N}<ext>``; return (N, path) of the largest N."""
    best, best_path = 0, None
    if os.path.isdir(outdir):
        for name in os.listdir(outdir):
            if name.startswith("round") and name.endswith(ext):
                stem = name[len("round"):-len(ext)]
                if stem.isdigit():
                    n = int(stem)
                    if n > best:
                        best, best_path = n, os.path.join(outdir, name)
    return best, best_path


def _total_epochs(stage_epochs):
    return 3 * max(int(stage_epochs), 1)


def _stage_of_epoch(epoch, stage_epochs):
    """1-based global epoch → (stage, epoch-in-stage, total-epochs)."""
    se = max(int(stage_epochs), 1)
    total = _total_epochs(se)
    epoch = int(epoch)
    if epoch < 1 or epoch > total:
        return None, 0, total
    idx = (epoch - 1) // se
    return DISTILL_STAGES[idx], (epoch - 1) % se + 1, total


def _hb_tag(stage, sepoch, stage_epochs, epoch, total):
    return f"stage={stage} e{sepoch}/{stage_epochs} g{epoch}/{total}"


def _aug_on(epoch, aug_from_epoch):
    """Train-time shuffle. ``aug_from_epoch<=0`` never; ``1`` from the start."""
    af = int(aug_from_epoch)
    if af <= 0:
        return False
    return int(epoch) >= af


def _hb_extra(lr, do_aug, stage, args):
    extra = f"lr={lr:g} aug={'on' if do_aug else 'off'}"
    if stage == "policy":
        extra += f" vw={args.value_weight:g} ow={args.own_weight:g}"
    return extra


def _gpu_stage_prefixes(_net, stage):
    """Trainable-name prefixes. ``None`` = every parameter (stage-1 joint)."""
    if stage == "policy":
        return None
    own = ("own_head",)
    value = ("value_attn", "value_fc1", "value_fc2")
    if stage == "own":
        return own
    if stage == "value":
        return value
    raise ValueError(f"unknown distill stage {stage!r}")


def _enter_gpu_stage(net, stage, base_lr):
    prefixes = _gpu_stage_prefixes(net, stage)
    if prefixes is None:
        for p in net.parameters():
            p.requires_grad = True
    else:
        for name, p in net.named_parameters():
            p.requires_grad = any(name == pref or name.startswith(pref + ".")
                                  for pref in prefixes)
    lr = float(base_lr) * STAGE_LR_MULT[stage]
    wd = 1e-4
    if getattr(net, "optimizer", None) is not None and net.optimizer.param_groups:
        wd = float(net.optimizer.param_groups[0].get("weight_decay", 1e-4))
    trainable = [p for p in net.parameters() if p.requires_grad]
    net.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    n_param = sum(p.numel() for p in trainable)
    extra = ""
    if stage == "policy":
        extra = (f" joint P+{float(getattr(net, 'value_weight', 1.0)):g}V"
                 f"+{float(getattr(net, 'own_weight', 1.0)):g}O, no freeze")
    else:
        extra = " head only, trunk frozen"
    print(f"[distill] stage={stage} Adam lr={lr:g} trainable={n_param}{extra}")
    return lr


def _save_gpu_distill(net, path, stage, epoch, stage_epochs):
    if not gpu_net_finite(net):
        raise ValueError("refusing to save non-finite GPU weights")
    net_type = gpu_net_type(net.net_type)
    ckpt = {
        "net_type": net_type,
        "label": gpu_net_label(net_type),
        "num_players": int(net.num_players),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": net.optimizer.state_dict(),
        "n_features": int(net.n_features),
        "hidden_dim": int(net.hidden_dim),
        "n_blocks": int(net.n_blocks),
        "attn_layer": int(net.attn_layer),
        "n_heads": int(net.n_heads),
        "distill_stage": stage,
        "distill_epoch": int(epoch),
        "distill_stage_epochs": int(stage_epochs),
    }
    torch.save(ckpt, path)


def _enter_cpu_stage(net, stage, base_lr):
    lr = float(base_lr) * STAGE_LR_MULT[stage]
    net.lr = lr
    net._distill_opt = NumpyAdam(lr)
    if stage == "policy":
        extra = (f"joint P+{float(getattr(net, 'value_weight', 1.0)):g}V"
                 f"+{float(getattr(net, 'own_weight', 1.0)):g}O, no freeze")
    else:
        extra = "head only, trunk frozen"
    print(f"[distill] stage={stage} Adam lr={lr:g} ({extra})")
    return lr


def _cpu_adam_path(ckpt_path):
    root, _ext = os.path.splitext(ckpt_path)
    return root + ".adam.npz"


def _save_cpu_adam(ckpt_path, opt, stage, epoch):
    path = _cpu_adam_path(ckpt_path)
    np.savez(path, _stage=np.asarray(stage), _epoch=np.asarray(epoch),
             **opt.state_dict())


def _load_cpu_adam(ckpt_path):
    path = _cpu_adam_path(ckpt_path)
    if not os.path.isfile(path):
        return None, None
    data = np.load(path, allow_pickle=False)
    try:
        stage = str(np.asarray(data["_stage"]).item())
        payload = {k: np.asarray(data[k]).copy() for k in data.files}
        return stage, payload
    finally:
        data.close()


def _train_gpu(net, args, records, train_idx, val_idx, graph, ng, native,
               start_epoch=1, resume_opt=None, resume_stage=None):
    """Torch loop: joint PVO → own → value. Freeze trunk after stage 1."""
    def run_batch(indices):
        return build_batch([records[i] for i in indices], graph, ng, native)

    os.makedirs(args.outdir, exist_ok=True)
    total = _total_epochs(args.epochs)
    hb_path = _progress_path(args.outdir)
    print(f"[distill] heartbeat → {hb_path} (truncated each epoch)")
    if start_epoch > total:
        out = os.path.join(args.outdir, "new.pt")
        save_net(net, out)
        print(f"[distill] already finished {total} epochs "
              f"(latest checkpoint is epoch {start_epoch - 1}); saved {out}")
        return out
    current_stage = None
    for epoch in range(start_epoch, total + 1):
        stage, sepoch, _tot = _stage_of_epoch(epoch, args.epochs)
        if stage != current_stage:
            lr = _enter_gpu_stage(net, stage, args.lr)
            current_stage = stage
            if (resume_opt is not None and resume_stage == stage
                    and epoch == start_epoch):
                try:
                    net.optimizer.load_state_dict(resume_opt)
                    for st in net.optimizer.state.values():
                        for k, v in st.items():
                            if torch.is_tensor(v):
                                st[k] = v.to(net.device)
                    print(f"[distill] restored Adam state for stage={stage}")
                except (ValueError, KeyError) as e:
                    print(f"[distill] optimizer not restored ({e})")
                resume_opt = None
        else:
            lr = float(args.lr) * STAGE_LR_MULT[stage]
        random.shuffle(train_idx)
        do_aug = _aug_on(epoch, args.aug_from_epoch)
        tag = _hb_tag(stage, sepoch, args.epochs, epoch, total)
        hb, n_tr, n_va = _begin_epoch_hb(
            args.outdir, tag, len(train_idx), len(val_idx), args.batch_size,
            extra=_hb_extra(lr, do_aug, stage, args))
        pl_s = vl_s = ol_s = 0.0
        steps = 0
        for i, batch_idx in enumerate(_batched(train_idx, args.batch_size), 1):
            X, mask, pol, val, own = run_batch(batch_idx)
            X, mask, pol, own, adj_in, adj_out = _aug_train_batch(
                net, graph, X, mask, pol, own, enabled=do_aug)
            pl, vl, ol, _loss = distill_train_on_batch(
                net, X, mask, pol, val, own, stage=stage,
                adj_in=adj_in, adj_out=adj_out)
            hb.step("train", i, n_tr, len(batch_idx), pl, vl, ol)
            if math.isfinite(pl):
                pl_s += pl
                vl_s += vl
                ol_s += ol
                steps += 1
        if steps == 0:
            hb.line("epoch_abort all train steps non-finite", fsync=True)
            print(f"[distill] epoch {epoch}: all steps non-finite, abort")
            break
        pl_m, vl_m, ol_m = pl_s / steps, vl_s / steps, ol_s / steps

        vpl = vvl = vol = 0.0
        hb.line(f"val_start steps={n_va}", fsync=True)
        for i, batch_idx in enumerate(_batched(val_idx, args.batch_size), 1):
            X, mask, pol, val, own = run_batch(batch_idx)
            p, v, o = eval_batch(net, X, mask, pol, val, own)
            hb.step("val", i, n_va, len(batch_idx), p, v, o)
            vpl += p
            vvl += v
            vol += o
        nvb = max(1, math.ceil(len(val_idx) / args.batch_size))
        hb.line(
            f"epoch_end train pl={pl_m:.4f} vl={vl_m:.4f} ol={ol_m:.4f} | "
            f"val pl={vpl / nvb:.4f} vl={vvl / nvb:.4f} ol={vol / nvb:.4f}",
            fsync=True)
        print(f"[distill] {tag} "
              f"train pl={pl_m:.4f} vl={vl_m:.4f} ol={ol_m:.4f} | "
              f"val pl={vpl / nvb:.4f} vl={vvl / nvb:.4f} ol={vol / nvb:.4f}")

        ckpt = os.path.join(args.outdir, f"round{epoch}.pt")
        _save_gpu_distill(net, ckpt, stage, epoch, args.epochs)
        print(f"[distill] checkpoint {ckpt}")

    out = os.path.join(args.outdir, "new.pt")
    save_net(net, out)
    print(f"[distill] saved {out}")
    return out


def _train_cpu(net, args, records, train_idx, val_idx, graph, ng, native,
               start_epoch=1, resume_adam=None, resume_stage=None):
    """NumPy loop: joint PVO → own → value. Freeze trunk after stage 1."""
    def run_batch(indices):
        return build_batch([records[i] for i in indices], graph, ng, native)

    os.makedirs(args.outdir, exist_ok=True)
    total = _total_epochs(args.epochs)
    print(f"[distill] CPU mini-batch size={args.batch_size}")
    hb_path = _progress_path(args.outdir)
    print(f"[distill] heartbeat → {hb_path} (truncated each epoch)")
    if start_epoch > total:
        out = os.path.join(args.outdir, "new.npz")
        save_cpu_net(net, out)
        print(f"[distill] already finished {total} epochs "
              f"(latest checkpoint is epoch {start_epoch - 1}); saved {out}")
        return out
    current_stage = None
    for epoch in range(start_epoch, total + 1):
        stage, sepoch, _tot = _stage_of_epoch(epoch, args.epochs)
        if stage != current_stage:
            lr = _enter_cpu_stage(net, stage, args.lr)
            current_stage = stage
            if (resume_adam is not None and resume_stage == stage
                    and epoch == start_epoch):
                try:
                    net._distill_opt.load_state_dict(resume_adam)
                    net._distill_opt.lr = lr
                    print(f"[distill] restored Adam state for stage={stage}")
                except (KeyError, ValueError, TypeError) as e:
                    print(f"[distill] optimizer not restored ({e})")
                resume_adam = None
        else:
            lr = float(args.lr) * STAGE_LR_MULT[stage]
        random.shuffle(train_idx)
        do_aug = _aug_on(epoch, args.aug_from_epoch)
        tag = _hb_tag(stage, sepoch, args.epochs, epoch, total)
        hb, n_tr, n_va = _begin_epoch_hb(
            args.outdir, tag, len(train_idx), len(val_idx), args.batch_size,
            extra=_hb_extra(lr, do_aug, stage, args))
        pl_s = vl_s = ol_s = 0.0
        steps = 0
        for i, batch_idx in enumerate(_batched(train_idx, args.batch_size), 1):
            X, mask, pol, val, own = run_batch(batch_idx)
            X, mask, pol, own, _, _ = _aug_train_batch(
                net, graph, X, mask, pol, own, enabled=do_aug)
            pl, vl, ol, _loss = net.distill_train_on_batch(
                X, mask, pol, val, own, stage=stage)
            hb.step("train", i, n_tr, len(batch_idx), pl, vl, ol)
            if math.isfinite(pl):
                pl_s += pl
                vl_s += vl
                ol_s += ol
                steps += 1
        if steps == 0:
            hb.line("epoch_abort all train steps non-finite", fsync=True)
            print(f"[distill] epoch {epoch}: all steps non-finite, abort")
            break
        pl_m, vl_m, ol_m = pl_s / steps, vl_s / steps, ol_s / steps

        vpl = vvl = vol = 0.0
        hb.line(f"val_start steps={n_va}", fsync=True)
        for i, batch_idx in enumerate(_batched(val_idx, args.batch_size), 1):
            X, mask, pol, val, own = run_batch(batch_idx)
            p, v, o = net.distill_eval_batch(X, mask, pol, val, own)
            hb.step("val", i, n_va, len(batch_idx), p, v, o)
            vpl += p
            vvl += v
            vol += o
        nvb = max(1, math.ceil(len(val_idx) / args.batch_size))
        hb.line(
            f"epoch_end train pl={pl_m:.4f} vl={vl_m:.4f} ol={ol_m:.4f} | "
            f"val pl={vpl / nvb:.4f} vl={vvl / nvb:.4f} ol={vol / nvb:.4f}",
            fsync=True)
        print(f"[distill] {tag} "
              f"train pl={pl_m:.4f} vl={vl_m:.4f} ol={ol_m:.4f} | "
              f"val pl={vpl / nvb:.4f} vl={vvl / nvb:.4f} ol={vol / nvb:.4f}")

        ckpt = os.path.join(args.outdir, f"round{epoch}.npz")
        save_cpu_net(net, ckpt)
        _save_cpu_adam(ckpt, net._distill_opt, stage, epoch)
        print(f"[distill] checkpoint {ckpt}")

    out = os.path.join(args.outdir, "new.npz")
    save_cpu_net(net, out)
    print(f"[distill] saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Distill KataGo into a GKT net")
    ap.add_argument("--net", default="gnn",
                    choices=["gnn", "2dcnn", "mlp", "1dcnn"],
                    help="student architecture (gnn/2dcnn = GPU, mlp/1dcnn = CPU)")
    ap.add_argument("--data", default=None,
                    help="JSONL of analyzed positions (gkt vertex order)")
    ap.add_argument("--graph-key", default="0",
                    help="builtin graph key ('0'=19x19, '0.5'=7x7)")
    ap.add_argument("--outdir", default="../base/gnn")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=20)
    ap.add_argument("--kernel-size", type=int, default=3,
                    help="1DCNN kernel size (only with --net 1dcnn)")
    ap.add_argument("--conv-layers", type=int, default=20,
                    help="1DCNN layers (only with --net 1dcnn)")
    ap.add_argument("--attn-layer", type=int, default=8)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10,
                    help="epochs per stage (3 stages: policy, own, value)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="base Adam lr for stage 1; own uses 25x, value 100x")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-weight", type=float, default=30.0,
                    help="stage-1 value MSE multiplier (own/value stages ignore this)")
    ap.add_argument("--own-weight", type=float, default=5.0,
                    help="stage-1 ownership MSE multiplier (own/value stages ignore this)")
    ap.add_argument("--aug-from-epoch", type=int, default=4,
                    help="1-based global epoch when train shuffle starts "
                         "(0=never, 1=from epoch 1). Default 4 = 3 identity epochs first")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest round*.pt/.npz checkpoint in outdir")
    args = ap.parse_args()

    if not args.data:
        ap.error("need --data <file.jsonl>")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    native = gkt_cpp.require_native()
    graph = get_builtin(args.graph_key)
    ng = gkt_cpp.py_graph_to_native(graph)
    n = len(graph.vertices)
    print(f"[distill] net={args.net} graph {args.graph_key}: n={n} vertices")

    # Load records (lightweight dicts only; feature tensors are built per-batch).
    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[distill] loaded {len(records)} records from {args.data}")
    if not records:
        print("[distill] no records; abort")
        return

    n_val = max(1, int(len(records) * args.val_frac))
    val_idx = list(range(n_val))
    train_idx = list(range(n_val, len(records)))
    print(f"[distill] train={len(train_idx)} val={len(val_idx)} "
          f"(val = first {args.val_frac:.0%}, fixed order, no aug)")
    af = int(args.aug_from_epoch)
    if af <= 0:
        print("[distill] train aug: OFF (identity numbering)")
    else:
        kind = ("2DCNN board symmetry (D4/Klein/torus)" if args.net == "2dcnn"
                else "random S_n + adj permute" if args.net == "gnn"
                else "random S_n (features/policy/own)")
        if af <= 1:
            print(f"[distill] train aug from epoch 1: {kind}")
        else:
            print(f"[distill] train aug: identity for epochs 1-{af - 1}, "
                  f"then {kind} from epoch {af}")
    print(f"[distill] stages: joint P+{args.value_weight:g}V+{args.own_weight:g}O "
          f"{args.epochs}ep @ lr={args.lr:g} (no freeze) → "
          f"own {args.epochs}ep @ lr={args.lr * 25:g} → "
          f"value {args.epochs}ep @ lr={args.lr * 100:g} "
          f"(trunk frozen after stage 1)")
    print(f"[distill] value_weight={args.value_weight:g} "
          f"own_weight={args.own_weight:g} (stage-1 loss only)")

    if args.net in GPU_NETS:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        net = make_net(args.net, n_features=None, hidden_dim=args.hidden,
                       n_blocks=args.n_blocks, graph=graph, device=device,
                       lr=args.lr, attn_layer=args.attn_layer, n_heads=args.attn_heads,
                       num_players=2, zero_value_heads=False,
                       value_weight=args.value_weight, own_weight=args.own_weight)
        print(f"[distill] net: {net.net_type} H={net.hidden_dim} "
              f"blocks={net.n_blocks} F={net.n_features} device={device}")
        start_epoch = 1
        resume_opt = resume_stage = None
        if args.resume:
            last_epoch, path = _latest_round(args.outdir, ".pt")
            if path:
                ckpt = torch.load(path, map_location=device)
                net.load_state_dict(ckpt["model_state_dict"], strict=True)
                resume_opt = ckpt.get("optimizer_state_dict")
                resume_stage = ckpt.get("distill_stage")
                start_epoch = last_epoch + 1
                print(f"[distill] resumed from {path} (epoch {last_epoch}"
                      f"{', stage=' + str(resume_stage) if resume_stage else ''}); "
                      f"continuing at epoch {start_epoch}")
            else:
                print(f"[distill] --resume set but no round*.pt in "
                      f"{args.outdir}; starting at epoch 1")
        out = _train_gpu(net, args, records, train_idx, val_idx, graph, ng,
                         native, start_epoch, resume_opt=resume_opt,
                         resume_stage=resume_stage)
    else:
        if args.net == "mlp":
            net = MlpPolicyValueNet(n_features=None, hidden_dim=args.hidden,
                                    lr=args.lr, num_players=2,
                                    zero_value_heads=False,
                                    value_weight=args.value_weight,
                                    own_weight=args.own_weight)
        else:
            net = Cnn1dPolicyValueNet(n_features=None, hidden_dim=args.hidden,
                                      kernel_size=args.kernel_size,
                                      n_layers=args.conv_layers, lr=args.lr,
                                      num_players=2,
                                      zero_value_heads=False,
                                      value_weight=args.value_weight,
                                      own_weight=args.own_weight)
        print(f"[distill] net: {args.net} H={net.H} F={net.F}")
        start_epoch = 1
        resume_adam = resume_stage = None
        if args.resume:
            last_epoch, path = _latest_round(args.outdir, ".npz")
            if path:
                data = np.load(path, allow_pickle=False)
                weights = {k: data[k] for k in data.files
                           if not str(k).startswith("_")}
                net.load_state_dict(weights)
                data.close()
                resume_stage, resume_adam = _load_cpu_adam(path)
                start_epoch = last_epoch + 1
                print(f"[distill] resumed from {path} (epoch {last_epoch}"
                      f"{', stage=' + str(resume_stage) if resume_stage else ''}); "
                      f"continuing at epoch {start_epoch}")
            else:
                print(f"[distill] --resume set but no round*.npz in "
                      f"{args.outdir}; starting at epoch 1")
        out = _train_cpu(net, args, records, train_idx, val_idx, graph, ng,
                         native, start_epoch, resume_adam=resume_adam,
                         resume_stage=resume_stage)

    meta = os.path.join(args.outdir, "distill_meta.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"source": args.data, "graph_key": args.graph_key,
                   "net": args.net, "stage_epochs": args.epochs,
                   "total_epochs": _total_epochs(args.epochs),
                   "stages": list(DISTILL_STAGES),
                   "stage_lr_mult": STAGE_LR_MULT,
                   "n_samples": len(records), "lr": args.lr,
                   "value_weight": args.value_weight,
                   "own_weight": args.own_weight}, f, indent=2)
    print(f"[distill] wrote {meta}")


if __name__ == "__main__":
    main()
