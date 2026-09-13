"""CPU policy-value nets (NumPy MLP / 1D conv).

Pairs with ``gkt_gpu.py`` (no PyTorch here). Search/self-play live in ``cpp/``.
Cross-graph loop: ``gkt_train_cpu.py``.
Docs: ``ref/rules.md``, ``ref/gomoku.md``, ``ref/implementation.md``.
"""
from __future__ import annotations

import os
import sys
import math
from typing import Tuple, Dict

import numpy as np

# make sibling modules importable when invoked from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gkt import (  # noqa: E402
    SCORE_BINS, SOFT_POLICY_TEMP,
    W_OPP_POLICY, W_SOFT_POLICY, W_BELIEF_PDF, W_BELIEF_CDF, W_STDEV, W_FUTURE,
    lead_belief, soft_policy_target, occupancy_with_empty,
    feature_dim, require_feature_dim, SQUASH_JAC_FLOOR,
)


def _load_arrays_strict(obj, d: Dict[str, np.ndarray]):
    have = {k for k, v in obj.__dict__.items() if isinstance(v, np.ndarray)}
    got = set(d)
    if have != got:
        raise ValueError(
            f"weight key mismatch: missing={sorted(have - got)[:12]} "
            f"extra={sorted(got - have)[:12]}")
    for k, v in d.items():
        setattr(obj, k, np.asarray(v).copy())


def _dpre_mse(pred, target, n_elem=1.0, weight=1.0, jac=None):
    """d(mean-MSE)/d(pre) with leaky squash Jacobian (floor on tanh/softplus)."""
    y = np.asarray(pred, dtype=np.float32)
    g = weight * 2.0 * (y - np.asarray(target, dtype=np.float32)) / n_elem
    if jac is None:
        jac = np.maximum(1.0 - y * y, SQUASH_JAC_FLOOR)
    return g * jac


def _init_cpu_aux(obj, rng, hidden_dim: int) -> None:
    s = math.sqrt(2.0 / hidden_dim)
    s256 = math.sqrt(2.0 / 256)
    obj.W_opp = (rng.standard_normal((hidden_dim, 1)) * s).astype(np.float32)
    obj.b_opp = np.zeros(1, dtype=np.float32)
    obj.W_opp_pass = (rng.standard_normal((hidden_dim, 1)) * s).astype(np.float32)
    obj.b_opp_pass = np.zeros(1, dtype=np.float32)
    obj.W_soft = (rng.standard_normal((hidden_dim, 1)) * s).astype(np.float32)
    obj.b_soft = np.zeros(1, dtype=np.float32)
    obj.W_soft_pass = (rng.standard_normal((hidden_dim, 1)) * s).astype(np.float32)
    obj.b_soft_pass = np.zeros(1, dtype=np.float32)
    obj.W_fut = np.zeros((hidden_dim, 1), dtype=np.float32)
    obj.b_fut = np.zeros(1, dtype=np.float32)
    obj.W_belief = (rng.standard_normal((256, SCORE_BINS)) * s256).astype(np.float32)
    obj.b_belief = np.zeros(SCORE_BINS, dtype=np.float32)
    obj.W_stdev = np.zeros((256, 1), dtype=np.float32)
    obj.b_stdev = np.zeros(1, dtype=np.float32)


def _cpu_policy_like(h, W, b, W_pass, b_pass, mask, target, weight):
    """CE on a pass-aware policy head. Returns (ce*w, dh, dW, db, dWp, dbp).

    Pass uses the same (W, b) as vertices (mean pool). W_pass / b_pass sit
    in the npz; dWp and dbp are zero.
    """
    n = h.shape[0]
    hm = h.mean(axis=0)
    logits = (h @ W + b).reshape(-1)
    pass_logit = float((hm @ W + b)[0])
    logits = np.concatenate([logits, [pass_logit]])
    logits = np.where(mask > 0, logits, -1e9)
    logits = logits - logits.max()
    ex = np.exp(logits)
    p = ex / ex.sum() if ex.sum() > 0 else np.ones_like(ex) / len(ex)
    w = float(weight)
    dlogit = np.where(mask > 0, (p - target) * w, 0.0)
    dv, dp = dlogit[:-1], dlogit[-1]
    dW = h.T @ dv.reshape(-1, 1) + hm.reshape(-1, 1) * dp
    db = np.array([dv.sum() + dp], dtype=np.float32)
    dWp = np.zeros_like(W_pass)
    dbp = np.zeros_like(b_pass)
    dh = dv.reshape(-1, 1) @ W.T + (dp / n) * W.reshape(1, -1)
    ce = float(-np.sum(target * np.log(p + 1e-9)) * w)
    return ce, dh, dW, db, dWp, dbp


def _cpu_aux_grads(obj, h, hv, mask, policy, aux):
    """Aux losses + grads into trunk `h` and pooled `hv`. Also SGD-updates heads."""
    dh = np.zeros_like(h)
    d_hv = np.zeros_like(hv)
    aloss = 0.0
    lr = obj.lr * max(float(aux.get("weight", 1.0)), 0.0)
    mask = np.asarray(mask, dtype=np.float32)
    # opponent next-move policy (legal set differs from the current mask)
    ce, dhi, dW, db, dWp, dbp = _cpu_policy_like(
        h, obj.W_opp, obj.b_opp, obj.W_opp_pass, obj.b_opp_pass,
        np.ones_like(mask), np.asarray(aux["opp"], np.float32),
        W_OPP_POLICY * float(aux["opp_w"]))
    dh += dhi
    obj.W_opp -= lr * dW
    obj.b_opp -= lr * db
    obj.W_opp_pass -= lr * dWp
    obj.b_opp_pass -= lr * dbp
    aloss += ce
    # soft policy
    soft = soft_policy_target(policy, mask, SOFT_POLICY_TEMP)
    ce, dhi, dW, db, dWp, dbp = _cpu_policy_like(
        h, obj.W_soft, obj.b_soft, obj.W_soft_pass, obj.b_soft_pass,
        mask, soft, W_SOFT_POLICY)
    dh += dhi
    obj.W_soft -= lr * dW
    obj.b_soft -= lr * db
    obj.W_soft_pass -= lr * dWp
    obj.b_soft_pass -= lr * dbp
    aloss += ce
    # future occupancy
    fut = np.tanh(h @ obj.W_fut + obj.b_fut).reshape(-1)
    tf = np.asarray(aux["future"], dtype=np.float32).reshape(-1)
    d_pre = _dpre_mse(fut, tf, fut.shape[0], W_FUTURE)
    dh += d_pre.reshape(-1, 1) @ obj.W_fut.T
    obj.W_fut -= lr * (h.T @ d_pre.reshape(-1, 1))
    obj.b_fut -= lr * np.array([d_pre.sum()], dtype=np.float32)
    aloss += W_FUTURE * float(np.mean((fut - tf) ** 2))
    # score belief pdf + cdf
    t = lead_belief(float(aux["lead"]))
    z = hv @ obj.W_belief + obj.b_belief
    zc = z - z.max()
    e = np.exp(zc)
    p = e / e.sum()
    pdf = float(-np.sum(t * np.log(p + 1e-9)))
    cdf_p = np.cumsum(p)
    cdf_t = np.cumsum(t)
    nb = p.shape[0]
    cdf_mse = float(np.mean((cdf_p - cdf_t) ** 2))
    d_cdf_dp = np.zeros_like(p)
    acc = 0.0
    for k in range(nb - 1, -1, -1):
        acc += (2.0 / nb) * (cdf_p[k] - cdf_t[k])
        d_cdf_dp[k] = acc
    d_z = W_BELIEF_PDF * (p - t) + W_BELIEF_CDF * (p * (d_cdf_dp - np.dot(p, d_cdf_dp)))
    d_hv += obj.W_belief @ d_z
    obj.W_belief -= lr * (hv.reshape(-1, 1) * d_z.reshape(1, -1))
    obj.b_belief -= lr * d_z
    aloss += W_BELIEF_PDF * pdf + W_BELIEF_CDF * cdf_mse
    # stdev: |lead - q|
    w = float((hv @ obj.W_stdev + obj.b_stdev)[0])
    y = w if w > 20 else math.log1p(math.exp(min(max(w, -40.0), 20.0)))
    tgt = abs(float(aux["lead"]) - float(aux["q"]))
    g = float(_dpre_mse(y, tgt, 1.0, W_STDEV,
                        jac=max(1.0 / (1.0 + math.exp(-min(max(w, -40.0), 40.0))),
                                SQUASH_JAC_FLOOR)))
    d_hv += g * obj.W_stdev.reshape(-1)
    obj.W_stdev -= lr * (hv.reshape(-1, 1) * g)
    obj.b_stdev -= lr * np.array([g], dtype=np.float32)
    aloss += W_STDEV * (y - tgt) ** 2
    return aloss, dh, d_hv


def _ensure_batch5(X, mask, policy, value, own):
    """Promote a single sample to ``(B, ...)`` so distill can share one path."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
        mask = np.asarray(mask, dtype=np.float32)[None]
        policy = np.asarray(policy, dtype=np.float32)[None]
        own = np.asarray(own, dtype=np.float32)[None]
        value = np.asarray(value, dtype=np.float32).reshape(1)
    elif X.ndim == 3:
        b = X.shape[0]
        mask = np.asarray(mask, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        own = np.asarray(own, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).reshape(b)
    else:
        raise ValueError(f"expected X (n,F) or (B,n,F); got {X.shape}")
    return X, mask, policy, value, own


def _pvown_forward_batch(obj, h, mask):
    """Policy / value / own heads on a batched trunk ``h`` of shape ``(B, n, H)``."""
    h = np.asarray(h, dtype=np.float32)
    b, n, _hid = h.shape
    logits_v = (h @ obj.Wp + obj.bp).reshape(b, n)
    hm = h.mean(axis=1)
    pass_logit = (hm @ obj.Wp + obj.bp).reshape(b)
    logits = np.concatenate([logits_v, pass_logit[:, None]], axis=1)
    legal = np.asarray(mask, dtype=np.float32).reshape(b, n + 1) > 0
    logits = np.where(legal, logits, np.float32(-1e9))
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    p = ex / np.maximum(ex.sum(axis=1, keepdims=True), np.float32(1e-12))
    z_attn = (h @ obj.W_attn + obj.b_attn).reshape(b, n)
    z_attn = z_attn - z_attn.max(axis=1, keepdims=True)
    e_attn = np.exp(z_attn)
    a = e_attn / np.maximum(e_attn.sum(axis=1, keepdims=True), np.float32(1e-12))
    v_pool = (h * a[:, :, None]).sum(axis=1)
    z1 = v_pool @ obj.W_v1 + obj.b_v1
    hv = np.maximum(z1, 0.0)
    v = np.tanh(hv @ obj.W_v2 + obj.b_v2).reshape(b)
    own = np.tanh(h @ obj.W_own + obj.b_own).reshape(b, n)
    return p, legal, hm, a, v_pool, z1, hv, v, own


def _pvown_losses(p, v, own, policy, value, own_t):
    """Unweighted mean policy CE / value MSE / own MSE (same reduction as GPU)."""
    policy = np.asarray(policy, dtype=np.float32).reshape(p.shape)
    value = np.asarray(value, dtype=np.float32).reshape(v.shape)
    own_t = np.asarray(own_t, dtype=np.float32).reshape(own.shape)
    pl = float(np.mean(-np.sum(policy * np.log(p + 1e-9), axis=1)))
    vl = float(np.mean((v - value) ** 2))
    ol = float(np.mean((own - own_t) ** 2))
    return pl, vl, ol, policy, value, own_t


def _cpu_policy_head_grads(obj, h, hm, p, policy, legal):
    b, n, hid = h.shape
    dlogit = (p - policy) / b
    dlogit = np.where(legal, dlogit, 0.0)
    dlogit_v = dlogit[:, :n]
    dlogit_pass = dlogit[:, n]
    dWp = h.reshape(b * n, hid).T @ dlogit_v.reshape(b * n, 1)
    dbp = np.array([dlogit_v.sum() + dlogit_pass.sum()], dtype=np.float32)
    dh = dlogit_v[:, :, None] * obj.Wp.reshape(1, 1, hid)
    dWp = dWp + hm.T @ dlogit_pass.reshape(b, 1)
    dh = dh + (dlogit_pass / n)[:, None, None] * obj.Wp.reshape(1, 1, hid)
    return dh, [("Wp", obj.Wp, dWp), ("bp", obj.bp, dbp)]


def _cpu_own_head_grads(obj, h, own, own_t, scale=1.0):
    b, n, hid = h.shape
    jac_o = np.maximum(1.0 - own * own, SQUASH_JAC_FLOOR)
    d_pre = np.float32(scale) * 2.0 * (own - own_t) / (n * b) * jac_o
    dW_own = h.reshape(b * n, hid).T @ d_pre.reshape(b * n, 1)
    db_own = np.array([d_pre.sum()], dtype=np.float32)
    dh = d_pre[:, :, None] * obj.W_own.reshape(1, 1, hid)
    return dh, [("W_own", obj.W_own, dW_own), ("b_own", obj.b_own, db_own)]


def _cpu_value_head_grads(obj, h, a, v_pool, z1, hv, v, value, scale=1.0):
    b, n, hid = h.shape
    jac_v = np.maximum(1.0 - v * v, SQUASH_JAC_FLOOR)
    d_z2 = np.float32(scale) * 2.0 * (v - value) / b * jac_v
    dW_v2 = hv.T @ d_z2.reshape(b, 1)
    db_v2 = np.asarray(d_z2.sum(), dtype=np.float32).reshape(1)
    d_hv = d_z2[:, None] * obj.W_v2.reshape(1, -1)
    d_z1 = d_hv * (z1 > 0)
    dW_v1 = v_pool.T @ d_z1
    db_v1 = d_z1.sum(axis=0)
    d_pool = d_z1 @ obj.W_v1.T
    dh = a[:, :, None] * d_pool[:, None, :]
    d_a = (h * d_pool[:, None, :]).sum(axis=-1)
    d_z_attn = a * (d_a - (a * d_a).sum(axis=1, keepdims=True))
    dW_attn = h.reshape(b * n, hid).T @ d_z_attn.reshape(b * n, 1)
    db_attn = np.array([d_z_attn.sum()], dtype=np.float32)
    dh = dh + d_z_attn[:, :, None] * obj.W_attn.reshape(1, 1, hid)
    named = [
        ("W_attn", obj.W_attn, dW_attn), ("b_attn", obj.b_attn, db_attn),
        ("W_v1", obj.W_v1, dW_v1), ("b_v1", obj.b_v1, db_v1),
        ("W_v2", obj.W_v2, dW_v2), ("b_v2", obj.b_v2, db_v2),
    ]
    return dh, named


def _cpu_distill_from_trunk(obj, h, mask, policy, value, own_t, stage="policy"):
    """Losses + grads for one distill stage. Returns ``(pl, vl, ol, dh, named)``.

    Stage 1 (``policy``) is joint ``pl + vw*vl + ow*ol``: ``named`` holds all
    three heads and ``dh`` is the trunk gradient. Own / value stages return
    one unweighted head and ``dh is None`` so the caller skips a frozen trunk.
    """
    p, legal, hm, a, v_pool, z1, hv, v, own = _pvown_forward_batch(obj, h, mask)
    pl, vl, ol, policy, value, own_t = _pvown_losses(
        p, v, own, policy, value, own_t)
    if not (math.isfinite(pl) and math.isfinite(vl) and math.isfinite(ol)):
        return float("nan"), float("nan"), float("nan"), None, None
    if stage == "policy":
        vw = float(getattr(obj, "value_weight", 1.0))
        ow = float(getattr(obj, "own_weight", 1.0))
        dh_p, named_p = _cpu_policy_head_grads(obj, h, hm, p, policy, legal)
        dh_o, named_o = _cpu_own_head_grads(obj, h, own, own_t, ow)
        dh_v, named_v = _cpu_value_head_grads(
            obj, h, a, v_pool, z1, hv, v, value, vw)
        return pl, vl, ol, dh_p + dh_o + dh_v, named_p + named_o + named_v
    if stage == "own":
        _, named = _cpu_own_head_grads(obj, h, own, own_t, 1.0)
        return pl, vl, ol, None, named
    if stage != "value":
        raise ValueError(f"unknown distill stage {stage!r}")
    _, named = _cpu_value_head_grads(
        obj, h, a, v_pool, z1, hv, v, value, 1.0)
    return pl, vl, ol, None, named


class NumpyAdam:
    """Adam on named NumPy parameter views (CPU distillation)."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = {}
        self.v = {}

    def state_dict(self):
        d = {"_t": np.asarray(self.t, dtype=np.int64),
             "_lr": np.asarray(self.lr, dtype=np.float64)}
        for k, arr in self.m.items():
            d[f"m.{k}"] = np.asarray(arr)
        for k, arr in self.v.items():
            d[f"v.{k}"] = np.asarray(arr)
        return d

    def load_state_dict(self, d):
        self.t = int(np.asarray(d["_t"]).item())
        if "_lr" in d:
            self.lr = float(np.asarray(d["_lr"]).item())
        self.m, self.v = {}, {}
        for k, arr in d.items():
            ks = str(k)
            if ks.startswith("m."):
                self.m[ks[2:]] = np.asarray(arr, dtype=np.float32).copy()
            elif ks.startswith("v."):
                self.v[ks[2:]] = np.asarray(arr, dtype=np.float32).copy()

    def step(self, named_pairs):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        step = self.lr * math.sqrt(1.0 - b2 ** self.t) / (1.0 - b1 ** self.t)
        eps = self.eps
        for name, p, g in named_pairs:
            g = np.asarray(g, dtype=np.float32)
            m = self.m.get(name)
            if m is None:
                m = np.zeros_like(p, dtype=np.float32)
                vv = np.zeros_like(p, dtype=np.float32)
                self.m[name] = m
                self.v[name] = vv
            else:
                vv = self.v[name]
            m *= b1
            m += (1.0 - b1) * g
            vv *= b2
            vv += (1.0 - b2) * (g * g)
            p -= (step * m / (np.sqrt(vv) + eps)).astype(p.dtype, copy=False)


def _cpu_adam_clip_apply(opt, named_pairs, max_norm=1.0):
    """Clip named grads to L2 ``max_norm``, then one Adam step."""
    sq = 0.0
    for _n, _p, g in named_pairs:
        sq += float(np.square(g).sum())
    nrm = math.sqrt(sq)
    if max_norm > 0.0 and nrm > max_norm:
        scale = np.float32(max_norm / (nrm + 1e-12))
        named_pairs = [(n, p, g * scale) for n, p, g in named_pairs]
    opt.step(named_pairs)


# ---------------------------------------------------------------------------
# Policy-Value Networks — NumPy implementations (CPU-friendly)
# ---------------------------------------------------------------------------

class MlpPolicyValueNet:
    """Two-layer MLP over per-vertex occupancy + empty + graph extras.

    Ablation: no trunk gpool. Policy is (n+1,) with Graph-Go pass at index n;
    value is a scalar in [-1, 1]. Attention-pooled value / per-vertex
    ownership match the GPU nets, so weights depend on F and H, not n.
    """

    def __init__(self, n_features=None, hidden_dim: int = 512,
                 lr: float = 1e-3, seed: int = 0, num_players: int = 2,
                 zero_value_heads: bool = True,
                 value_weight: float = 1.0, own_weight: float = 1.0):
        rng = np.random.default_rng(seed)
        self.num_players = int(num_players)
        self.F = feature_dim(self.num_players) if n_features is None else int(n_features)
        require_feature_dim(self.F, self.num_players)
        self.n_features = self.F
        self.H = hidden_dim
        self.lr = lr
        # Distillation and distilled-start self-play re-weight value/own
        # (their MSE is ~1e-2 of the policy CE). Default 1.0 is from-zero /
        # Gomoku; Go starters pass 3000 / 25 to match distill.py.
        self.value_weight = float(value_weight)
        self.own_weight = float(own_weight)
        fin = self.F + 1
        self.W1 = (rng.standard_normal((fin, hidden_dim))
                   * math.sqrt(2.0 / fin)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, hidden_dim))
                   * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        # policy head
        self.Wp = (rng.standard_normal((hidden_dim, 1))
                   * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.bp = np.zeros(1, dtype=np.float32)
        # pass: same Linear as vertices on mean(h); W_pass kept for checkpoints
        self.W_pass = (rng.standard_normal((hidden_dim, 1))
                       * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_pass = np.zeros(1, dtype=np.float32)
        # value head: attention pooling over vertices (softmax attention, then
        # two FC layers) — aligned with gkt_gpu.GnnPolicyValueNet's value head.
        # The softmax keeps the pool n-invariant, so it stays graph-agnostic.
        self.W_attn = (rng.standard_normal((hidden_dim, 1))
                       * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_attn = np.zeros(1, dtype=np.float32)
        self.W_v1 = (rng.standard_normal((hidden_dim, 256))
                     * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_v1 = np.zeros(256, dtype=np.float32)
        if zero_value_heads:
            self.W_v2 = np.zeros((256, 1), dtype=np.float32)
            self.b_v2 = np.zeros(1, dtype=np.float32)
        else:
            # non-zero init so the value head receives gradient from step 1
            self.W_v2 = (rng.standard_normal((256, 1))
                         * math.sqrt(2.0 / 256)).astype(np.float32)
            self.b_v2 = np.zeros(1, dtype=np.float32)
        # ownership head (per-vertex tanh); same n-invariant Linear as GPU
        if zero_value_heads:
            self.W_own = np.zeros((hidden_dim, 1), dtype=np.float32)
            self.b_own = np.zeros(1, dtype=np.float32)
        else:
            self.W_own = (rng.standard_normal((hidden_dim, 1))
                          * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
            self.b_own = np.zeros(1, dtype=np.float32)
        _init_cpu_aux(self, rng, hidden_dim)

    def forward(self, X: np.ndarray, legal_mask: np.ndarray = None
                ) -> Tuple[np.ndarray, float]:
        """X: (n, F). Returns (policy_probs (n+1,), value (scalar in [-1,1]).

        The last policy entry (index n) is Graph-Go pass (masked out in k-in-a-row).
        """
        # trunk
        Xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(Xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        # policy logits (n,) + pass logit (scalar) -> (n+1,)
        logits = (h2 @ self.Wp + self.bp).reshape(-1)
        pass_logit = float((h2.mean(axis=0) @ self.Wp + self.bp)[0])
        logits = np.concatenate([logits, [pass_logit]])
        if legal_mask is not None:
            logits = np.where(legal_mask > 0, logits, -1e9)
        # softmax
        logits = logits - logits.max()
        ex = np.exp(logits)
        policy = ex / ex.sum() if ex.sum() > 0 else np.ones_like(ex) / len(ex)
        # value head: attention pooling over vertices (n-invariant, transferable)
        z_attn = h2 @ self.W_attn + self.b_attn        # (n, 1)
        z_attn = z_attn - z_attn.max()
        e_attn = np.exp(z_attn)
        a = e_attn / e_attn.sum(axis=0, keepdims=True)  # (n, 1) softmax over vertices
        v_pool = (h2 * a).sum(axis=0)                   # (H,)
        v = np.tanh(np.maximum(v_pool @ self.W_v1 + self.b_v1, 0.0)
                    @ self.W_v2 + self.b_v2)[0]
        return policy, float(v)

    def backward(self, X: np.ndarray, legal_mask: np.ndarray,
                 target_policy: np.ndarray, target_value: float,
                 target_own: np.ndarray, aux
                 ) -> Tuple[float, float, float, float]:
        """SGD step: policy CE + score MSE + own MSE + KataGo aux.

        Returns (policy_loss, value_loss, own_loss, total_loss).
        """
        policy, value = self.forward(X, legal_mask)
        dlogit = policy - target_policy
        dlogit = np.where(legal_mask > 0, dlogit, 0.0)
        Xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(Xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        n = h2.shape[0]
        dlogit_vertex = dlogit[:-1]
        dlogit_pass = dlogit[-1]
        dWp = h2.T @ dlogit_vertex.reshape(-1, 1)
        dbp = dlogit_vertex.sum() + dlogit_pass
        dh2 = dlogit_vertex.reshape(-1, 1) @ self.Wp.T
        dW_pass = np.zeros_like(self.W_pass)
        db_pass = np.zeros_like(self.b_pass)
        hm = h2.mean(axis=0)
        dWp = dWp + hm.reshape(-1, 1) * dlogit_pass
        dh2 += (dlogit_pass / n) * self.Wp.reshape(1, -1)
        z_attn = h2 @ self.W_attn + self.b_attn
        z_attn = z_attn - z_attn.max()
        e_attn = np.exp(z_attn)
        a = e_attn / e_attn.sum(axis=0, keepdims=True)
        v_pool = (h2 * a).sum(axis=0)
        z1 = v_pool @ self.W_v1 + self.b_v1
        hv = np.maximum(z1, 0.0)
        d_z2 = float(_dpre_mse(value, target_value, weight=self.value_weight))
        dW_v2 = hv.reshape(-1, 1) * d_z2
        db_v2 = np.array([d_z2], dtype=np.float32)
        d_hv = d_z2 * self.W_v2.reshape(-1)
        own = np.tanh(h2 @ self.W_own + self.b_own).reshape(-1)
        to = np.asarray(target_own, dtype=np.float32).reshape(-1)
        d_pre = _dpre_mse(own, to, own.shape[0], weight=self.own_weight)
        dW_own = h2.T @ d_pre.reshape(-1, 1)
        db_own = np.array([d_pre.sum()], dtype=np.float32)
        dh2 += d_pre.reshape(-1, 1) @ self.W_own.T
        own_mse = float(np.mean((own - to) ** 2))
        aloss, dh_a, dhv_a = _cpu_aux_grads(self, h2, hv, legal_mask, target_policy, aux)
        dh2 += dh_a
        d_hv = d_hv + dhv_a
        d_z1 = d_hv * (z1 > 0)
        dW_v1 = v_pool.reshape(-1, 1) * d_z1.reshape(1, -1)
        db_v1 = d_z1
        d_pool = d_z1 @ self.W_v1.T
        d_h2_v = a * d_pool.reshape(1, -1)
        d_a = (h2 @ d_pool).reshape(-1, 1)
        d_z_attn = a * (d_a - (a * d_a).sum(axis=0, keepdims=True))
        dW_attn = h2.T @ d_z_attn
        db_attn = d_z_attn.sum(axis=0)
        dh2 += d_h2_v + d_z_attn @ self.W_attn.T
        dh2 = dh2 * (1 - h2 ** 2)
        dh1 = dh2 @ self.W2.T
        dh1 = dh1 * (1 - h1 ** 2)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dW1 = Xe.T @ dh1
        db1 = dh1.sum(axis=0)
        lr = self.lr * max(float(aux.get("weight", 1.0)), 0.0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.Wp -= lr * dWp
        self.bp -= lr * dbp
        self.W_pass -= lr * dW_pass
        self.b_pass -= lr * db_pass
        self.W_attn -= lr * dW_attn
        self.b_attn -= lr * db_attn
        self.W_v1 -= lr * dW_v1
        self.b_v1 -= lr * db_v1
        self.W_v2 -= lr * dW_v2
        self.b_v2 -= lr * db_v2
        self.W_own -= lr * dW_own
        self.b_own -= lr * db_own
        eps = 1e-9
        ce = -np.sum(target_policy * np.log(policy + eps))
        mse = (value - target_value) ** 2
        return float(ce), float(mse), float(own_mse), float(ce + mse + own_mse + aloss)

    def distill_eval_batch(self, X, mask, policy, value, own):
        """Batched forward losses for distillation val (no aug, no SGD)."""
        X, mask, policy, value, own = _ensure_batch5(X, mask, policy, value, own)
        xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        p, _legal, _hm, _a, _vp, _z1, _hv, v, opred = _pvown_forward_batch(
            self, h2, mask)
        pl, vl, ol, _, _, _ = _pvown_losses(p, v, opred, policy, value, own)
        return pl, vl, ol

    def distill_train_on_batch(self, X, mask, policy, value, own, stage="policy"):
        """Mini-batch Adam on one distill stage (policy / own / value).

        Stage 1 jointly updates trunk + policy/value/own heads. Own / value
        stages freeze the trunk.
        """
        X, mask, policy, value, own = _ensure_batch5(X, mask, policy, value, own)
        xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        pl, vl, ol, dh2, named = _cpu_distill_from_trunk(
            self, h2, mask, policy, value, own, stage=stage)
        if named is None:
            return (float("nan"),) * 4
        if stage == "policy":
            b, n, hid = h2.shape
            dh2 = dh2 * (1.0 - h2 * h2)
            dh1 = (dh2 @ self.W2.T) * (1.0 - h1 * h1)
            dW2 = h1.reshape(b * n, hid).T @ dh2.reshape(b * n, hid)
            db2 = dh2.sum(axis=(0, 1))
            dW1 = xe.reshape(b * n, xe.shape[-1]).T @ dh1.reshape(b * n, hid)
            db1 = dh1.sum(axis=(0, 1))
            named = named + [
                ("W1", self.W1, dW1), ("b1", self.b1, db1),
                ("W2", self.W2, dW2), ("b2", self.b2, db2),
            ]
        opt = getattr(self, "_distill_opt", None)
        if opt is None:
            raise RuntimeError("CPU distill Adam is not attached")
        _cpu_adam_clip_apply(opt, named)
        return pl, vl, ol, pl + vl + ol

    def own_map(self, X: np.ndarray) -> np.ndarray:
        Xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(Xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return np.tanh(h2 @ self.W_own + self.b_own).reshape(-1)

    # serialization for migration learning
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}

    def predict(self, X: np.ndarray, legal_mask: np.ndarray = None
                ) -> Tuple[np.ndarray, float]:
        """Single-position interface used by MCTS: (features, legal_mask) ->
        (policy_probs (n+1,), value scalar). Index n is the Graph-Go pass
        dimension (masked out in k-in-a-row). Alias of forward()."""
        return self.forward(X, legal_mask)

    def predict_batch(self, X_batch: np.ndarray, masks: np.ndarray = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched interface. Returns (policy (B, n+1), value (B,))."""
        X = np.asarray(X_batch, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        xe = occupancy_with_empty(X, self.num_players)
        h1 = np.tanh(xe @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        b, n, _hid = h2.shape
        if masks is None:
            masks = np.ones((b, n + 1), dtype=np.float32)
        p, _legal, _hm, _a, _vp, _z1, _hv, v, _own = _pvown_forward_batch(
            self, h2, masks)
        return p.astype(np.float32), v.astype(np.float32)

    def load_state_dict(self, d: Dict[str, np.ndarray]):
        _load_arrays_strict(self, d)


class Cnn1dPolicyValueNet:
    """CPU 1DCNN: convolve the n vertices as a 1D sequence in index order.

    Occupancy + empty + graph extras, then `n_layers` 1D convolutions along
    vertex index (not graph edges). Ablation: no trunk gpool. On a grid,
    index neighbors coincide with same-row neighbors, so it captures
    horizontal locality but misses vertical and diagonal edges. Baseline for
    how much the GNN's real graph structure helps vs the two-layer MLP.

    CPU (NumPy) only. Value head matches MLP / GPU (attention pool).
    """

    def __init__(self, n_features=None, hidden_dim: int = 512,
                 kernel_size: int = 3, n_layers: int = 20,
                 lr: float = 1e-3, seed: int = 0, num_players: int = 2,
                 zero_value_heads: bool = True,
                 value_weight: float = 1.0, own_weight: float = 1.0):
        rng = np.random.default_rng(seed)
        self.num_players = int(num_players)
        self.F = feature_dim(self.num_players) if n_features is None else int(n_features)
        require_feature_dim(self.F, self.num_players)
        self.n_features = self.F
        self.H = hidden_dim
        self.K = kernel_size
        self.L = n_layers
        self.pad = kernel_size // 2
        self.lr = lr
        self.value_weight = float(value_weight)
        self.own_weight = float(own_weight)
        fin = self.F + 1
        self.W_enc = (rng.standard_normal((fin, hidden_dim))
                      * math.sqrt(2.0 / fin)).astype(np.float32)
        self.b_enc = np.zeros(hidden_dim, dtype=np.float32)
        self.conv_W = (rng.standard_normal((n_layers, kernel_size, hidden_dim, hidden_dim))
                       * math.sqrt(2.0 / (kernel_size * hidden_dim))).astype(np.float32)
        self.conv_b = np.zeros((n_layers, hidden_dim), dtype=np.float32)
        self.Wp = (rng.standard_normal((hidden_dim, 1))
                   * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.bp = np.zeros(1, dtype=np.float32)
        # pass head (global, from mean-pooled trunk; mean pool keeps it n-invariant)
        self.W_pass = (rng.standard_normal((hidden_dim, 1))
                       * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_pass = np.zeros(1, dtype=np.float32)
        self.W_attn = (rng.standard_normal((hidden_dim, 1))
                       * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_attn = np.zeros(1, dtype=np.float32)
        self.W_v1 = (rng.standard_normal((hidden_dim, 256))
                     * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_v1 = np.zeros(256, dtype=np.float32)
        if zero_value_heads:
            self.W_v2 = np.zeros((256, 1), dtype=np.float32)
            self.b_v2 = np.zeros(1, dtype=np.float32)
        else:
            self.W_v2 = (rng.standard_normal((256, 1))
                         * math.sqrt(2.0 / 256)).astype(np.float32)
            self.b_v2 = np.zeros(1, dtype=np.float32)
        if zero_value_heads:
            self.W_own = np.zeros((hidden_dim, 1), dtype=np.float32)
            self.b_own = np.zeros(1, dtype=np.float32)
        else:
            self.W_own = (rng.standard_normal((hidden_dim, 1))
                          * math.sqrt(2.0 / hidden_dim)).astype(np.float32)
            self.b_own = np.zeros(1, dtype=np.float32)
        _init_cpu_aux(self, rng, hidden_dim)

    def _im2col(self, h: np.ndarray) -> np.ndarray:
        """Unfold a 1D conv window.

        ``h`` is ``(n, H)`` or ``(B, n, H)``. Returns ``(n, K*H)`` or
        ``(B, n, K*H)``. Zero-pad ``pad`` on both ends of the vertex axis.
        Uses ``sliding_window_view`` (no Python loop over n).
        """
        h = np.asarray(h, dtype=np.float32)
        k = int(self.K)
        if h.ndim == 2:
            n, hid = h.shape
            hp = np.pad(h, ((self.pad, self.pad), (0, 0)), mode="constant")
            win = np.lib.stride_tricks.sliding_window_view(hp, k, axis=0)[:n]
            # (n, H, K) -> (n, K, H) -> (n, K*H). [:n] for even K (2*pad = K).
            return np.ascontiguousarray(
                np.transpose(win, (0, 2, 1)).reshape(n, k * hid))
        if h.ndim != 3:
            raise ValueError(f"_im2col expected (n,H) or (B,n,H); got {h.shape}")
        b, n, hid = h.shape
        hp = np.pad(h, ((0, 0), (self.pad, self.pad), (0, 0)), mode="constant")
        win = np.lib.stride_tricks.sliding_window_view(hp, k, axis=1)[:, :n]
        # (B, n, H, K) -> (B, n, K, H) -> (B, n, K*H)
        return np.ascontiguousarray(
            np.transpose(win, (0, 1, 3, 2)).reshape(b, n, k * hid))

    def _conv1d(self, h: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """h: (n, H) or (B, n, H); W: (K, H, H); b: (H,)."""
        cols = self._im2col(h)
        return cols @ W.reshape(self.K * self.H, self.H) + b

    def forward(self, X: np.ndarray, legal_mask: np.ndarray = None
                ) -> Tuple[np.ndarray, float]:
        """X: (n, F). Returns (policy_probs (n+1,), value (scalar in [-1,1]).

        The last policy entry (index n) is Graph-Go pass (masked out in k-in-a-row).
        """
        Xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(Xe @ self.W_enc + self.b_enc)
        for l in range(self.L):
            h = np.tanh(self._conv1d(h, self.conv_W[l], self.conv_b[l]))
        logits = (h @ self.Wp + self.bp).reshape(-1)              # (n,)
        pass_logit = float((h.mean(axis=0) @ self.Wp + self.bp)[0])
        logits = np.concatenate([logits, [pass_logit]])           # (n+1,)
        if legal_mask is not None:
            logits = np.where(legal_mask > 0, logits, -1e9)
        logits = logits - logits.max()
        ex = np.exp(logits)
        policy = ex / ex.sum() if ex.sum() > 0 else np.ones_like(ex) / len(ex)
        z_attn = h @ self.W_attn + self.b_attn
        z_attn = z_attn - z_attn.max()
        e_attn = np.exp(z_attn)
        a = e_attn / e_attn.sum(axis=0, keepdims=True)
        v_pool = (h * a).sum(axis=0)
        v = np.tanh(np.maximum(v_pool @ self.W_v1 + self.b_v1, 0.0)
                    @ self.W_v2 + self.b_v2)[0]
        return policy, float(v)

    def backward(self, X: np.ndarray, legal_mask: np.ndarray,
                 target_policy: np.ndarray, target_value: float,
                 target_own: np.ndarray, aux
                 ) -> Tuple[float, float, float, float]:
        """SGD step: policy CE + score MSE + own MSE + KataGo aux."""
        policy, value = self.forward(X, legal_mask)

        dlogit = policy - target_policy
        dlogit = np.where(legal_mask > 0, dlogit, 0.0)

        Xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(Xe @ self.W_enc + self.b_enc)
        acts = [h]
        cols_list, zs = [], []
        for l in range(self.L):
            cols = self._im2col(h)
            z = cols @ self.conv_W[l].reshape(self.K * self.H, self.H) + self.conv_b[l]
            cols_list.append(cols)
            zs.append(z)
            h = np.tanh(z)
            acts.append(h)
        h_out = h
        n = h_out.shape[0]

        dlogit_vertex = dlogit[:-1]
        dlogit_pass = dlogit[-1]
        dWp = h_out.T @ dlogit_vertex.reshape(-1, 1)
        dbp = dlogit_vertex.sum() + dlogit_pass
        dh = dlogit_vertex.reshape(-1, 1) @ self.Wp.T
        dW_pass = np.zeros_like(self.W_pass)
        db_pass = np.zeros_like(self.b_pass)
        hm = h_out.mean(axis=0)
        dWp = dWp + hm.reshape(-1, 1) * dlogit_pass
        dh += (dlogit_pass / n) * self.Wp.reshape(1, -1)

        z_attn = h_out @ self.W_attn + self.b_attn
        z_attn = z_attn - z_attn.max()
        e_attn = np.exp(z_attn)
        a = e_attn / e_attn.sum(axis=0, keepdims=True)
        v_pool = (h_out * a).sum(axis=0)
        z1 = v_pool @ self.W_v1 + self.b_v1
        hv = np.maximum(z1, 0.0)
        d_z2 = float(_dpre_mse(value, target_value, weight=self.value_weight))
        dW_v2 = hv.reshape(-1, 1) * d_z2
        db_v2 = np.array([d_z2], dtype=np.float32)
        d_hv = d_z2 * self.W_v2.reshape(-1)
        own = np.tanh(h_out @ self.W_own + self.b_own).reshape(-1)
        to = np.asarray(target_own, dtype=np.float32).reshape(-1)
        d_pre = _dpre_mse(own, to, own.shape[0], weight=self.own_weight)
        dW_own = h_out.T @ d_pre.reshape(-1, 1)
        db_own = np.array([d_pre.sum()], dtype=np.float32)
        dh += d_pre.reshape(-1, 1) @ self.W_own.T
        own_mse = float(np.mean((own - to) ** 2))
        aloss, dh_a, dhv_a = _cpu_aux_grads(
            self, h_out, hv, legal_mask, target_policy, aux)
        dh += dh_a
        d_hv = d_hv + dhv_a
        d_z1 = d_hv * (z1 > 0)
        dW_v1 = v_pool.reshape(-1, 1) * d_z1.reshape(1, -1)
        db_v1 = d_z1
        d_pool = d_z1 @ self.W_v1.T
        d_h2_v = a * d_pool.reshape(1, -1)
        d_a = (h_out @ d_pool).reshape(-1, 1)
        d_z_attn = a * (d_a - (a * d_a).sum(axis=0, keepdims=True))
        dW_attn = h_out.T @ d_z_attn
        db_attn = d_z_attn.sum(axis=0)
        dh += d_h2_v + d_z_attn @ self.W_attn.T

        # conv layers (reverse order)
        dconv_W = np.zeros_like(self.conv_W)
        dconv_b = np.zeros_like(self.conv_b)
        for l in reversed(range(self.L)):
            dh = dh * (1.0 - np.tanh(zs[l]) ** 2)
            dconv_W[l] = (cols_list[l].T @ dh).reshape(self.K, self.H, self.H)
            dconv_b[l] = dh.sum(axis=0)
            dcols = dh @ self.conv_W[l].reshape(self.K * self.H, self.H).T
            dh = self._unpad_grad(dcols, n)

        # encoder
        dh = dh * (1.0 - acts[0] ** 2)
        dW_enc = Xe.T @ dh
        db_enc = dh.sum(axis=0)

        # SGD update
        lr = self.lr * max(float(aux.get("weight", 1.0)), 0.0)
        self.W_enc -= lr * dW_enc
        self.b_enc -= lr * db_enc
        self.conv_W -= lr * dconv_W
        self.conv_b -= lr * dconv_b
        self.Wp -= lr * dWp
        self.bp -= lr * dbp
        self.W_pass -= lr * dW_pass
        self.b_pass -= lr * db_pass
        self.W_attn -= lr * dW_attn
        self.b_attn -= lr * db_attn
        self.W_v1 -= lr * dW_v1
        self.b_v1 -= lr * db_v1
        self.W_v2 -= lr * dW_v2
        self.b_v2 -= lr * db_v2
        self.W_own -= lr * dW_own
        self.b_own -= lr * db_own

        eps = 1e-9
        ce = -np.sum(target_policy * np.log(policy + eps))
        mse = (value - target_value) ** 2
        return float(ce), float(mse), float(own_mse), float(ce + mse + own_mse + aloss)

    def _unpad_grad(self, dcols: np.ndarray, n: int) -> np.ndarray:
        """Fold im2col grads back. ``dcols`` is ``(n, K*H)`` or ``(B, n, K*H)``.

        Loop is over kernel width K (typically 3), not over n.
        """
        k = int(self.K)
        hid = int(self.H)
        if dcols.ndim == 2:
            dwin = dcols.reshape(n, k, hid)
            dh = np.zeros((n + 2 * self.pad, hid), dtype=dcols.dtype)
            for t in range(k):
                dh[t:t + n] += dwin[:, t, :]
            return dh[self.pad:self.pad + n]
        b = dcols.shape[0]
        dwin = dcols.reshape(b, n, k, hid)
        dh = np.zeros((b, n + 2 * self.pad, hid), dtype=dcols.dtype)
        for t in range(k):
            dh[:, t:t + n, :] += dwin[:, :, t, :]
        return dh[:, self.pad:self.pad + n, :]

    def distill_eval_batch(self, X, mask, policy, value, own):
        """Batched forward losses for distillation val (no aug, no SGD)."""
        X, mask, policy, value, own = _ensure_batch5(X, mask, policy, value, own)
        xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(xe @ self.W_enc + self.b_enc)
        for l in range(self.L):
            h = np.tanh(self._conv1d(h, self.conv_W[l], self.conv_b[l]))
        p, _legal, _hm, _a, _vp, _z1, _hv, v, opred = _pvown_forward_batch(
            self, h, mask)
        pl, vl, ol, _, _, _ = _pvown_losses(p, v, opred, policy, value, own)
        return pl, vl, ol

    def distill_train_on_batch(self, X, mask, policy, value, own, stage="policy"):
        """Mini-batch Adam on one distill stage (policy / own / value).

        Own / value stages freeze the conv trunk, so im2col backward is skipped.
        Stage 1 jointly updates the trunk + policy/value/own heads.
        """
        X, mask, policy, value, own = _ensure_batch5(X, mask, policy, value, own)
        xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(xe @ self.W_enc + self.b_enc)
        acts = [h] if stage == "policy" else None
        for l in range(self.L):
            h = np.tanh(self._conv1d(h, self.conv_W[l], self.conv_b[l]))
            if acts is not None:
                acts.append(h)
        pl, vl, ol, dh, named = _cpu_distill_from_trunk(
            self, h, mask, policy, value, own, stage=stage)
        if named is None:
            return (float("nan"),) * 4
        if stage == "policy":
            b, n, hid = h.shape
            kh = self.K * self.H
            dconv_w = np.zeros_like(self.conv_W)
            dconv_b = np.zeros_like(self.conv_b)
            for l in reversed(range(self.L)):
                dh = dh * (1.0 - acts[l + 1] ** 2)
                cols = self._im2col(acts[l])
                dconv_w[l] = (cols.reshape(b * n, kh).T @ dh.reshape(b * n, hid)).reshape(
                    self.K, self.H, self.H)
                dconv_b[l] = dh.sum(axis=(0, 1))
                dcols = dh @ self.conv_W[l].reshape(kh, self.H).T
                dh = self._unpad_grad(dcols, n)
            dh = dh * (1.0 - acts[0] ** 2)
            dW_enc = xe.reshape(b * n, xe.shape[-1]).T @ dh.reshape(b * n, hid)
            db_enc = dh.sum(axis=(0, 1))
            named = named + [
                ("W_enc", self.W_enc, dW_enc), ("b_enc", self.b_enc, db_enc),
                ("conv_W", self.conv_W, dconv_w), ("conv_b", self.conv_b, dconv_b),
            ]
        opt = getattr(self, "_distill_opt", None)
        if opt is None:
            raise RuntimeError("CPU distill Adam is not attached")
        _cpu_adam_clip_apply(opt, named)
        return pl, vl, ol, pl + vl + ol

    def own_map(self, X: np.ndarray) -> np.ndarray:
        Xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(Xe @ self.W_enc + self.b_enc)
        for l in range(self.L):
            h = np.tanh(self._conv1d(h, self.conv_W[l], self.conv_b[l]))
        return np.tanh(h @ self.W_own + self.b_own).reshape(-1)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}

    def load_state_dict(self, d: Dict[str, np.ndarray]):
        _load_arrays_strict(self, d)

    def predict(self, X: np.ndarray, legal_mask: np.ndarray = None
                ) -> Tuple[np.ndarray, float]:
        """Same as forward: policy (n+1,) with pass at index n, value scalar."""
        return self.forward(X, legal_mask)

    def predict_batch(self, X_batch: np.ndarray, masks: np.ndarray = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched interface. Returns (policy (B, n+1), value (B,))."""
        X = np.asarray(X_batch, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        xe = occupancy_with_empty(X, self.num_players)
        h = np.tanh(xe @ self.W_enc + self.b_enc)
        for l in range(self.L):
            h = np.tanh(self._conv1d(h, self.conv_W[l], self.conv_b[l]))
        b, n, _hid = h.shape
        if masks is None:
            masks = np.ones((b, n + 1), dtype=np.float32)
        p, _legal, _hm, _a, _vp, _z1, _hv, v, _own = _pvown_forward_batch(
            self, h, masks)
        return p.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

CPU_NET_LABELS = {"mlp": "MLP", "1dcnn": "1DCNN"}


def _player_count(k) -> int:
    k = int(k)
    if k not in (2, 3, 4):
        raise ValueError(f"num_players must be 2, 3, or 4; got {k}")
    return k


def tagged_num_players(src) -> int:
    """Player-count tag on a net or npz dict (required)."""
    if isinstance(src, dict):
        if "_num_players" in src:
            return _player_count(np.asarray(src["_num_players"]).item())
        if "num_players" in src:
            return _player_count(src["num_players"])
        raise ValueError("checkpoint missing _num_players")
    return _player_count(src.num_players)


def cpu_net_type(net_or_kind) -> str:
    if isinstance(net_or_kind, Cnn1dPolicyValueNet):
        return "1dcnn"
    if isinstance(net_or_kind, MlpPolicyValueNet):
        return "mlp"
    t = str(net_or_kind).strip().lower()
    if t not in CPU_NET_LABELS:
        raise ValueError(f"unknown CPU net_type {net_or_kind!r}; expected 'mlp' or '1dcnn'")
    return t


def cpu_net_label(net_or_kind) -> str:
    return CPU_NET_LABELS[cpu_net_type(net_or_kind)]


def save_cpu_net(net, path: str):
    kind = cpu_net_type(net)
    np.savez(path,
             _net_type=np.asarray(kind),
             _label=np.asarray(CPU_NET_LABELS[kind]),
             _num_players=np.asarray(int(net.num_players)),
             **net.state_dict())


def peek_cpu_tags(path: str):
    """Return (architecture label, num_players)."""
    data = np.load(path, allow_pickle=False)
    try:
        if "_net_type" not in data.files or "_num_players" not in data.files:
            raise ValueError(f"{path} is not a GKT CPU checkpoint")
        lab = cpu_net_label(np.asarray(data["_net_type"]).item())
        if "_label" in data.files:
            got = str(np.asarray(data["_label"]).item()).strip().upper()
            if got != lab:
                raise ValueError(f"{path} label {got!r} != {lab}")
        k = _player_count(np.asarray(data["_num_players"]).item())
        return lab, k
    finally:
        data.close()


def peek_cpu_label(path: str) -> str:
    """Read MLP / 1DCNN · kP from a .npz without constructing the net."""
    lab, k = peek_cpu_tags(path)
    return f"{lab} · {k}P"


def load_cpu_net(path: str):
    """Load a CPU checkpoint saved by `save_cpu_net` (requires `_net_type`)."""
    data = np.load(path, allow_pickle=False)
    try:
        if "_net_type" not in data.files or "_num_players" not in data.files:
            raise ValueError(f"{path} is not a GKT CPU checkpoint")
        kind = cpu_net_type(np.asarray(data["_net_type"]).item())
        npl = _player_count(np.asarray(data["_num_players"]).item())
        weights = {key: data[key] for key in data.files
                   if not str(key).startswith("_")}
        if kind == "1dcnn":
            enc_in = int(weights["W_enc"].shape[0])
            hidden_dim = int(weights["W_enc"].shape[1])
            n_layers = int(weights["conv_W"].shape[0])
            kernel_size = int(weights["conv_W"].shape[1])
            if enc_in != feature_dim(npl) + 1:
                raise ValueError(
                    f"{path} encoder in={enc_in} != feature_dim+1={feature_dim(npl) + 1}")
            net = Cnn1dPolicyValueNet(feature_dim(npl), hidden_dim,
                                      kernel_size=kernel_size, n_layers=n_layers,
                                      num_players=npl)
        elif kind == "mlp":
            enc_in = int(weights["W1"].shape[0])
            hidden_dim = int(weights["W1"].shape[1])
            if enc_in != feature_dim(npl) + 1:
                raise ValueError(
                    f"{path} encoder in={enc_in} != feature_dim+1={feature_dim(npl) + 1}")
            net = MlpPolicyValueNet(feature_dim(npl), hidden_dim, num_players=npl)
        else:
            raise ValueError(f"unknown CPU net_type {kind!r}")
        net.load_state_dict(weights)
        return net
    finally:
        data.close()
