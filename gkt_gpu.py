"""GKT GPU nets (GNN / 2DCNN) and the parallel trainer ``GktTrainer``.

Search and self-play samples: ``cpp/`` via ``gkt.py`` / ``gkt_cpp.py``.
NumPy nets: ``gkt_cpu.py``. Cross-graph loop: ``gkt_train_gpu.py``.
Docs: ``ref/algorithm.md``, ``ref/training_method.md``.
"""

from __future__ import annotations
import os
import sys
import random
import time
import warnings
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphs import DiGraph, is_k_in_row_rules  # noqa: E402
from gkt import (  # noqa: E402
    GktSelfPlay, stack_aux,
    SCORE_BINS,
    W_OPP_POLICY, W_SOFT_POLICY, W_BELIEF_PDF, W_BELIEF_CDF,
    W_STDEV, W_FUTURE, lead_belief, soft_policy_target,
    progress_append, make_move_heartbeat, worker_progress_path,
    feature_dim, require_feature_dim, SQUASH_JAC_FLOOR,
)
from grid_sym import augment_vertex_batch, augment_graph_batch  # noqa: E402


# Illegal-move logit mask. Use finite -1e9, not -inf, so 0 * log_softmax(-inf)
# cannot yield NaN (the NumPy nets use -1e9 too).
MASK_VALUE = -1e9
LOGIT_CLAMP = 20.0
STDEV_MAX = 8.0
GPOOL_SQRT_N_CAP = 8.0


def _cat_shared_linear_pass(head: nn.Linear, h: torch.Tensor) -> torch.Tensor:
    """(B, n, H) → (B, n+1): same Linear on each vertex and on mean(h).

    Sharing the vertex head keeps pass = typical vertex when embeddings
    collapse, so softmax stays ~1/(n+1) until the pooled trunk differs.
    """
    vert = head(h).squeeze(-1)
    pss = head(h.mean(dim=1, keepdim=True)).squeeze(-1)
    return torch.cat([vert, pss], dim=-1)


def _cat_shared_conv_pass(conv: nn.Conv2d, h: torch.Tensor) -> torch.Tensor:
    """(B, C, R, C) → (B, n+1): same 1×1 conv on the grid and on spatial mean."""
    B = h.size(0)
    n = h.size(2) * h.size(3)
    vert = conv(h).reshape(B, n)
    pss = conv(h.mean(dim=(2, 3), keepdim=True)).reshape(B, 1)
    return torch.cat([vert, pss], dim=-1)


def _clamp_logits(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)


def gpu_net_finite(net) -> bool:
    with torch.no_grad():
        for p in net.parameters():
            if p.numel() and not torch.isfinite(p).all():
                return False
    return True


def _grads_finite(net) -> bool:
    for p in net.parameters():
        if p.grad is not None and p.grad.numel() and not torch.isfinite(p.grad).all():
            return False
    return True


def _loss_floats(*ts):
    out = []
    for t in ts:
        x = float(t.detach().item())
        out.append(x if math.isfinite(x) else float("nan"))
    return tuple(out)


def encode_features_torch(x: torch.Tensor, num_players: int) -> torch.Tensor:
    k = int(num_players)
    occ = x[..., :k]
    extra = x[..., k:]
    empty = (1.0 - occ.sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
    return torch.cat([occ, empty, extra], dim=-1)


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _zero_squash_heads(net, value_heads: bool = True) -> None:
    """Zero the squash heads' output layers.

    ``value_fc2`` / ``own_head`` are zeroed so value & ownership start at a
    constant 0 — correct for from-zero self-play, where an early random value
    would mislead MCTS. For supervised distillation they must NOT be zeroed:
    the value/own gradient is already ~1e-2 of the policy gradient, and a zero
    init welds those heads shut (no gradient reaches the trunk until the last
    layer grows a non-zero weight). Pass ``value_heads=False`` there.

    ``fut_head`` / ``stdev_head`` are always zeroed (no trustworthy target in
    either regime).
    """
    names = ["fut_head", "stdev_head"]
    if value_heads:
        names = ["value_fc2", "own_head"] + names
    for name in names:
        _zero_linear(getattr(net, name))


class _ScaleJac(torch.autograd.Function):
    """Forward `y`; backward multiplies by `jac` (leaky squash)."""

    @staticmethod
    def forward(ctx, pre, y, jac):
        ctx.save_for_backward(jac)
        return y

    @staticmethod
    def backward(ctx, grad):
        jac, = ctx.saved_tensors
        return grad * jac, None, None


def _squash_tanh(pre):
    y = torch.tanh(pre)
    if not torch.is_grad_enabled():
        return y
    jac = (1.0 - y * y).clamp(min=SQUASH_JAC_FLOOR)
    return _ScaleJac.apply(pre, y, jac)


def _squash_softplus(pre):
    y = F.softplus(pre)
    if not torch.is_grad_enabled():
        return y
    jac = torch.sigmoid(pre).clamp(min=SQUASH_JAC_FLOOR)
    return _ScaleJac.apply(pre, y, jac)


def _pool_heads(net, h, hv):
    return {
        "value": _squash_tanh(net.value_fc2(hv)),
        "own": _squash_tanh(net.own_head(h).squeeze(-1)),
        "future": _squash_tanh(net.fut_head(h).squeeze(-1)),
        "belief": _clamp_logits(net.belief_head(hv)),
        "stdev": _squash_softplus(net.stdev_head(hv).squeeze(-1)).clamp(max=STDEV_MAX),
    }


def _sample_w(aux, B, device):
    w = aux.get("weight")
    if w is None:
        t = torch.ones(B, device=device)
    else:
        t = torch.from_numpy(np.asarray(w, dtype=np.float32)).float().to(device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)
    return t / t.mean().clamp(min=1e-6)


def _masked_ce(logits, target, mask_bool):
    logits = logits.masked_fill(~mask_bool, MASK_VALUE)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _permute_adj_batch(adj, perms):
    """adj (n, n), perms (B, n) gather indices → (B, n, n) with A'[i,j]=A[π(i),π(j)]."""
    return adj[perms[:, :, None], perms[:, None, :]]


def _gpu_train_on_batch(net, X, mask, policy, value, ownership, aux):
    aux = dict(aux)
    nt = getattr(net, "net_type", "")
    adj_in = adj_out = None
    if nt == "2dcnn":
        # 2DCNN keeps board geometry: only board symmetries (D4 / torus shift).
        X, mask, policy, ownership, ep, ev = augment_graph_batch(
            X, mask, policy, getattr(net, "_aug_graph", None), ownership=ownership,
            extra_policy=[aux["opp"]], extra_vertex=[aux["future"]])
    else:
        # GNN / MLP / 1DCNN: random S_n relabel (GNN also permutes adj).
        X, mask, policy, ownership, ep, ev, perms = augment_vertex_batch(
            X, mask, policy, ownership=ownership,
            extra_policy=[aux["opp"]], extra_vertex=[aux["future"]])
        if nt == "gnn":
            perms_t = torch.from_numpy(np.asarray(perms, dtype=np.int64)).long().to(
                net.device)
            adj_in = _permute_adj_batch(net.adj_in, perms_t)
            adj_out = _permute_adj_batch(net.adj_out, perms_t)
    aux["opp"], aux["future"] = ep[0], ev[0]

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(net.device)
    m_t = torch.from_numpy(np.asarray(mask) > 0).bool().to(net.device)
    p_t = torch.from_numpy(np.asarray(policy, dtype=np.float32)).float().to(net.device)
    v_t = torch.from_numpy(np.asarray(value, dtype=np.float32)).float().to(net.device)
    o_t = torch.from_numpy(np.asarray(ownership, dtype=np.float32)).float().to(net.device)

    net.train()
    if adj_in is not None:
        out = net.forward_adj(X_t, adj_in, adj_out)
    else:
        out = net.forward(X_t)
    w = _sample_w(aux, X_t.shape[0], net.device)
    policy_loss = (w * _masked_ce(out["policy"], p_t, m_t)).mean()
    value_loss = (w * (out["value"].squeeze(-1) - v_t).pow(2)).mean()
    own_loss = (w * (out["own"] - o_t).pow(2).mean(dim=-1)).mean()

    opp_t = torch.from_numpy(np.asarray(aux["opp"], dtype=np.float32)).float().to(net.device)
    opp_w = torch.from_numpy(np.asarray(aux["opp_w"], dtype=np.float32)).float().to(net.device)
    # Next-move (opp) visits can sit on points illegal *now* (recapture after
    # capture, superko, turn salt). Do not apply the current legal mask.
    opp_ce = -(opp_t * F.log_softmax(out["opp"], dim=-1)).sum(dim=-1)
    aux_loss = W_OPP_POLICY * (w * opp_ce * opp_w).mean()
    soft_np = np.stack([soft_policy_target(policy[i], mask[i])
                        for i in range(policy.shape[0])])
    soft_t = torch.from_numpy(soft_np).float().to(net.device)
    aux_loss = aux_loss + W_SOFT_POLICY * (w * _masked_ce(out["soft"], soft_t, m_t)).mean()
    leads = np.asarray(aux["lead"], dtype=np.float32).reshape(-1)
    bel_np = np.stack([lead_belief(float(x)) for x in leads])
    bel_t = torch.from_numpy(bel_np).float().to(net.device)
    logb = F.log_softmax(out["belief"], dim=-1)
    pdf = (w * -(bel_t * logb).sum(dim=-1)).mean()
    cdf_p = F.softmax(out["belief"], dim=-1).cumsum(dim=-1)
    cdf_t = bel_t.cumsum(dim=-1)
    cdf_mse = (w * (cdf_p - cdf_t).pow(2).mean(dim=-1)).mean()
    aux_loss = aux_loss + W_BELIEF_PDF * pdf + W_BELIEF_CDF * cdf_mse
    q_t = torch.from_numpy(np.asarray(aux["q"], dtype=np.float32)).float().to(net.device)
    lead_t = torch.from_numpy(leads).float().to(net.device)
    aux_loss = aux_loss + W_STDEV * (w * (out["stdev"] - (lead_t - q_t).abs()).pow(2)).mean()
    fut_t = torch.from_numpy(np.asarray(aux["future"], dtype=np.float32)).float().to(net.device)
    aux_loss = aux_loss + W_FUTURE * (w * (out["future"] - fut_t).pow(2).mean(dim=-1)).mean()

    vw = float(getattr(net, "value_weight", 1.0))
    ow = float(getattr(net, "own_weight", 1.0))
    loss = policy_loss + vw * value_loss + ow * own_loss + aux_loss
    net.optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return _loss_floats(policy_loss, value_loss, own_loss, aux_loss, loss)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if (not math.isfinite(float(gn))) or (not _grads_finite(net)):
        net.optimizer.zero_grad(set_to_none=True)
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    net.optimizer.step()
    if not gpu_net_finite(net):
        net.optimizer.zero_grad(set_to_none=True)
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return _loss_floats(policy_loss, value_loss, own_loss, aux_loss, loss)


def _set_weights_strict(net, weights: Dict[str, np.ndarray]):
    sd = net.state_dict()
    missing = set(sd) - set(weights)
    extra = set(weights) - set(sd)
    if missing or extra:
        raise ValueError(
            f"weight key mismatch: missing={sorted(missing)[:12]} "
            f"extra={sorted(extra)[:12]}")
    for k, arr in weights.items():
        t = torch.from_numpy(np.asarray(arr)).to(sd[k].dtype).to(sd[k].device)
        sd[k].copy_(t)


class _GpuNetIO:
    """Shared numpy I/O for GNN / 2DCNN (MCTS + trainer)."""

    def train_on_batch(self, X, mask, policy, value, ownership, aux):
        return _gpu_train_on_batch(self, X, mask, policy, value, ownership, aux)

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy()
                for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        _set_weights_strict(self, weights)

    @torch.no_grad()
    def predict(self, X, legal_mask=None) -> Tuple[np.ndarray, float]:
        x = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().unsqueeze(0).to(self.device)
        out = self.forward(x)
        logits = out["policy"][0]
        if legal_mask is not None:
            m = torch.from_numpy(np.asarray(legal_mask) > 0).bool().to(self.device)
            logits = logits.masked_fill(~m, MASK_VALUE)
        policy = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        return policy, float(out["value"].item())

    @torch.no_grad()
    def predict_batch(self, X_batch, masks=None):
        x = torch.from_numpy(np.asarray(X_batch, dtype=np.float32)).float().to(self.device)
        out = self.forward(x)
        logits = out["policy"]
        if masks is not None:
            m = torch.from_numpy(np.asarray(masks) > 0).bool().to(self.device)
            logits = logits.masked_fill(~m, MASK_VALUE)
        policy = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        return (policy,
                out["value"].squeeze(-1).cpu().numpy(),
                out["stdev"].reshape(-1).detach().cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------------------
# PyTorch policy-value network (graph-agnostic parameterization)
# ---------------------------------------------------------------------------

class _GlobalPoolBias(nn.Module):
    """Broadcast mean / max / scaled-mean as a per-channel bias. Zero-init."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * hidden_dim, hidden_dim)
        _zero_linear(self.proj)

    def forward(self, h):
        mean = h.mean(dim=1)
        mx = h.amax(dim=1)
        # sqrt(n) from the tensor itself so jit.trace does not do float(h.shape[1]).
        scale = torch.sqrt(torch.ones_like(h[:, :, 0]).sum(dim=1, keepdim=True))
        scale = scale.clamp(max=GPOOL_SQRT_N_CAP)
        g = torch.cat([mean, mx, mean * scale], dim=-1)
        return h + self.proj(g).unsqueeze(1)


class _SpatialPoolBias(nn.Module):
    """Same gpool for NCHW feature maps. Zero-init."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(3 * channels, channels)
        _zero_linear(self.proj)

    def forward(self, h):
        mean = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        scale = torch.sqrt(torch.ones_like(h[:, 0]).flatten(1).sum(dim=1))
        scale = scale.clamp(max=GPOOL_SQRT_N_CAP)
        g = torch.cat([mean, mx, mean * scale.unsqueeze(-1)], dim=-1)
        return h + self.proj(g)[:, :, None, None]


class _GNNBlock(nn.Module):
    """Directed message passing: in/out neighbour sums plus log-degree, then MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.gpool = _GlobalPoolBias(hidden_dim)

    def forward(self, h, adj_in, adj_out):
        msg_in = adj_in @ h
        msg_out = adj_out @ h
        B = h.shape[0]
        din = adj_in.sum(dim=-1)
        dout = adj_out.sum(dim=-1)
        if din.dim() == 1:
            din = din.unsqueeze(0).expand(B, -1)
            dout = dout.unsqueeze(0).expand(B, -1)
        log_din = torch.log1p(din).unsqueeze(-1)
        log_dout = torch.log1p(dout).unsqueeze(-1)
        update = self.mlp(torch.cat([h, msg_in, msg_out, log_din, log_dout], dim=-1))
        return self.gpool(self.norm(h + self.act(update)))


class _AttentionBlock(nn.Module):
    """Global multi-head self-attention; weights depend on H, not n."""

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h):
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        nh, hd = self.n_heads, self.head_dim
        q = q.unflatten(-1, (nh, hd)).transpose(1, 2)
        k = k.unflatten(-1, (nh, hd)).transpose(1, 2)
        v = v.unflatten(-1, (nh, hd)).transpose(1, 2)
        scale = hd ** -0.5
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).flatten(-2)
        out = self.out(out)
        return self.norm(h + self.act(out))


class GnnPolicyValueNet(nn.Module, _GpuNetIO):
    """Graph-neural-network policy-value net for generalized Go.

    Input (B, n, F): occupancy one-hot plus graph extras (`feature_dim`).
    Encoder adds a derived empty channel. Trunk: in/out sum-aggregation MLP
    residuals with per-block gpool, plus one global self-attention block
    before `attn_layer`. Heads: per-vertex policy (pass is the same Linear
    on mean(h)), attention-pooled value, ownership / future / belief / stdev.

    Weights depend only on F and H. Adjacency is 0/1 data from `set_graph()`.
    Graph-Go pass is index n (k-in-a-row masks it).
    """

    def __init__(self, n_features: Optional[int] = None, hidden_dim: int = 512,
                 n_blocks: int = 20, graph: Optional[DiGraph] = None,
                 device: str = "cpu", lr: float = 1e-3,
                 weight_decay: float = 1e-4, seed: Optional[int] = 0,
                 attn_layer: int = 8, n_heads: int = 4,
                 num_players: int = 2, zero_value_heads: bool = True):
        super().__init__()
        self.num_players = int(num_players)
        self.n_features = (feature_dim(self.num_players)
                           if n_features is None else int(n_features))
        require_feature_dim(self.n_features, self.num_players)
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.net_type = "gnn"
        self.device = device
        # One global self-attention block, inserted before block #attn_layer.
        self.attn_layer = min(max(attn_layer, 0), n_blocks - 1)
        self.n_heads = n_heads

        self.encoder = nn.Linear(self.n_features + 1, hidden_dim)
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([_GNNBlock(hidden_dim)
                                     for _ in range(n_blocks)])
        self.attn_block = _AttentionBlock(hidden_dim, n_heads)
        self.policy_head = nn.Linear(hidden_dim, 1)   # vertices + shared pass
        self.pass_head = nn.Linear(hidden_dim, 1)     # in state_dict; not in forward
        self.value_attn = nn.Linear(hidden_dim, 1)    # -> (B, n) per-vertex attention logits
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.own_head = nn.Linear(hidden_dim, 1)      # -> (B, n) per-vertex ownership
        self.opp_head = nn.Linear(hidden_dim, 1)
        self.opp_pass = nn.Linear(hidden_dim, 1)      # in state_dict; not in forward
        self.soft_head = nn.Linear(hidden_dim, 1)
        self.soft_pass = nn.Linear(hidden_dim, 1)     # in state_dict; not in forward
        self.fut_head = nn.Linear(hidden_dim, 1)
        self.belief_head = nn.Linear(256, SCORE_BINS)
        self.stdev_head = nn.Linear(256, 1)
        _zero_squash_heads(self, value_heads=zero_value_heads)

        # adjacency — 0/1 data, not weights
        self.adj_in = None
        self.adj_out = None
        self._aug_graph = None

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        if seed is not None:
            torch.manual_seed(seed)
        if graph is not None:
            self.set_graph(graph)

    def set_graph(self, graph: DiGraph) -> "GnnPolicyValueNet":
        """Build 0/1 in/out adjacency for `graph`."""
        n = len(graph.vertices)
        A_in = torch.zeros(n, n)
        A_out = torch.zeros(n, n)
        for u_lbl in graph.vertices:
            u = graph.index_of(u_lbl)
            for v_lbl in graph.out_adj[u_lbl]:
                v = graph.index_of(v_lbl)
                A_out[u, v] = 1.0
                A_in[v, u] = 1.0
        self.adj_in = A_in.to(self.device)
        self.adj_out = A_out.to(self.device)
        self._aug_graph = graph
        return self

    def forward_adj(self, x: torch.Tensor, adj_in, adj_out
                    ) -> Dict[str, torch.Tensor]:
        h = self.enc_norm(F.relu(self.encoder(encode_features_torch(x, self.num_players))))
        for i, blk in enumerate(self.blocks):
            if i == self.attn_layer:
                h = self.attn_block(h)
            h = blk(h, adj_in, adj_out)
        return self._decode_heads_linear(h)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, n, F) -> dict of heads. Search uses policy, value, stdev."""
        assert self.adj_in is not None, "call set_graph() before forward()"
        return self.forward_adj(x, self.adj_in, self.adj_out)

    def _decode_heads_linear(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        policy = _clamp_logits(_cat_shared_linear_pass(self.policy_head, h))
        opp = _clamp_logits(_cat_shared_linear_pass(self.opp_head, h))
        soft = _clamp_logits(_cat_shared_linear_pass(self.soft_head, h))
        attn = torch.softmax(self.value_attn(h).squeeze(-1), dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)
        hv = F.relu(self.value_fc1(pooled))
        return {
            "policy": policy,
            "opp": opp,
            "soft": soft,
            **_pool_heads(self, h, hv),
        }


class _CNNResBlock(nn.Module):
    """3×3 residual block; circular pad on toroidal boards, else zero pad."""

    def __init__(self, channels: int, toroidal: bool = False):
        super().__init__()
        self.toroidal = toroidal
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gpool = _SpatialPoolBias(channels)

    def _pad(self, x):
        if self.toroidal:
            return F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(self._pad(x))))
        h = self.bn2(self.conv2(self._pad(h)))
        return self.gpool(F.relu(x + h))


class Cnn2dPolicyValueNet(nn.Module, _GpuNetIO):
    """2D-convolutional policy-value net for grid-shaped boards.

    GPU 2DCNN: a 1x1 stem, then `n_blocks` 2D residual blocks (no global
    self-attention — that layer is GNN-only),     a 1x1 policy head (pass is the same conv on the spatial mean;
    Graph-Go; k-in-a-row masks index n), and an
    attention-pooled value head. Same interface as `GnnPolicyValueNet`.

    Unlike the GNN (any graph), the 2DCNN only applies to graphs with
    ``.grid`` metadata: it reshapes n vertices into an R×C image. The 1x1
    policy head and attention-pooled value head keep it transferable across
    board sizes; `set_graph` recovers (R, C) and toroidality from `graph.grid`.
    """

    def __init__(self, n_features: Optional[int] = None, hidden_dim: int = 512,
                 n_blocks: int = 20, graph: Optional[DiGraph] = None,
                 device: str = "cpu", lr: float = 1e-3,
                 weight_decay: float = 1e-4, seed: Optional[int] = 0,
                 attn_layer: int = 8, n_heads: int = 4,
                 num_players: int = 2, zero_value_heads: bool = True):
        super().__init__()
        self.num_players = int(num_players)
        self.n_features = (feature_dim(self.num_players)
                           if n_features is None else int(n_features))
        require_feature_dim(self.n_features, self.num_players)
        self.hidden_dim = hidden_dim   # conv channel count
        self.n_blocks = n_blocks
        self.net_type = "2dcnn"
        self.device = device
        # GNN-only hyperparams on the object; 2DCNN forward does not use them.
        self.attn_layer = min(max(attn_layer, 0), n_blocks - 1)
        self.n_heads = n_heads

        self.stem = nn.Conv2d(n_features + 1, hidden_dim, 1)
        self.stem_norm = nn.BatchNorm2d(hidden_dim)
        self.blocks = nn.ModuleList([_CNNResBlock(hidden_dim)
                                     for _ in range(n_blocks)])
        self.policy_conv = nn.Conv2d(hidden_dim, 1, 1)   # grid + shared pass
        self.pass_head = nn.Linear(hidden_dim, 1)        # in state_dict; not in forward
        self.value_attn = nn.Linear(hidden_dim, 1)       # per-vertex attn logits
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.own_head = nn.Linear(hidden_dim, 1)         # per-vertex ownership
        self.opp_conv = nn.Conv2d(hidden_dim, 1, 1)
        self.opp_pass = nn.Linear(hidden_dim, 1)         # in state_dict; not in forward
        self.soft_conv = nn.Conv2d(hidden_dim, 1, 1)
        self.soft_pass = nn.Linear(hidden_dim, 1)        # in state_dict; not in forward
        self.fut_head = nn.Linear(hidden_dim, 1)
        self.belief_head = nn.Linear(256, SCORE_BINS)
        self.stdev_head = nn.Linear(256, 1)
        _zero_squash_heads(self, value_heads=zero_value_heads)

        # grid shape (data, not weights)
        self._m = None
        self._n_grid = None
        self._toroidal = False
        self._aug_graph = None

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        if seed is not None:
            torch.manual_seed(seed)
        if graph is not None:
            self.set_graph(graph)

    def set_graph(self, graph: DiGraph) -> "Cnn2dPolicyValueNet":
        """Recover the m x n grid shape and toroidality from `graph.grid`.

        Only grid graphs (those built by `_square_grid`) carry `.grid`; non-grid
        graphs raise here — the 2DCNN cannot represent them.
        """
        grid = getattr(graph, "grid", None)
        if grid is None:
            raise ValueError(
                "Cnn2dPolicyValueNet requires rectangular .grid metadata "
                "(any grid key, including 2 / G*; default training omits oversized 2). "
                "Graph has no grid metadata.")
        m, n_grid, toroidal = grid
        self._m = int(m)
        self._n_grid = int(n_grid)
        self._toroidal = bool(toroidal)
        for blk in self.blocks:
            blk.toroidal = self._toroidal
        self._aug_graph = graph
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, n, F) -> dict of heads. Search uses policy, value, stdev."""
        if self._m is None:
            raise RuntimeError("call set_graph() on a grid graph before forward()")
        x = encode_features_torch(x, self.num_players)
        m, ng = self._m, self._n_grid
        # reshape fails if n != m*ng. Do not `assert n == m*ng`: during
        # jit.trace that converts a Tensor to a Python bool (TracerWarning flood).
        h = x.transpose(1, 2).reshape(x.size(0), x.size(-1), m, ng)
        h = self.stem(h)
        h = F.relu(self.stem_norm(h))
        for blk in self.blocks:
            h = blk(h)
        h_flat = h.flatten(2).transpose(1, 2)
        attn = torch.softmax(self.value_attn(h_flat).squeeze(-1), dim=1)
        pooled = (h_flat * attn.unsqueeze(-1)).sum(dim=1)
        hv = F.relu(self.value_fc1(pooled))
        policy = _clamp_logits(_cat_shared_conv_pass(self.policy_conv, h))
        opp = _clamp_logits(_cat_shared_conv_pass(self.opp_conv, h))
        soft = _clamp_logits(_cat_shared_conv_pass(self.soft_conv, h))
        return {
            "policy": policy,
            "opp": opp,
            "soft": soft,
            **_pool_heads(self, h_flat, hv),
        }


GPU_NET_LABELS = {"gnn": "GNN", "2dcnn": "2DCNN"}


def gpu_net_type(net_type: str) -> str:
    t = str(net_type).strip().lower()
    if t not in GPU_NET_LABELS:
        raise ValueError(f"unknown GPU net_type {net_type!r}; expected 'gnn' or '2dcnn'")
    return t


def gpu_net_label(net_type: str) -> str:
    return GPU_NET_LABELS[gpu_net_type(net_type)]


def _player_count(k) -> int:
    k = int(k)
    if k not in (2, 3, 4):
        raise ValueError(f"num_players must be 2, 3, or 4; got {k}")
    return k


def tagged_num_players(src) -> int:
    """Player-count tag on a net or checkpoint dict (required)."""
    if isinstance(src, dict):
        if "num_players" not in src:
            raise ValueError("checkpoint missing num_players")
        return _player_count(src["num_players"])
    return _player_count(src.num_players)


def gpu_policy_prefixes(net_type: str):
    """Parameter-name prefixes of the policy readout (pass shares this Linear)."""
    if gpu_net_type(net_type) == "2dcnn":
        return ("policy_conv",)
    return ("policy_head",)


def set_gpu_policy_frozen(net, frozen: bool) -> int:
    """Freeze or unfreeze the policy readout and rebuild Adam on the rest.

    Gradients still flow through the frozen head into the trunk (the readout
    weights themselves do not move). Returns how many tensors were frozen.
    """
    prefixes = gpu_policy_prefixes(net.net_type)
    n_frozen = 0
    for name, p in net.named_parameters():
        is_pol = any(name == pref or name.startswith(pref + ".")
                     for pref in prefixes)
        if is_pol:
            p.requires_grad = not frozen
            if frozen:
                n_frozen += 1
        else:
            p.requires_grad = True
    lr = 1e-4
    wd = 1e-4
    if getattr(net, "optimizer", None) is not None and net.optimizer.param_groups:
        lr = float(net.optimizer.param_groups[0]["lr"])
        wd = float(net.optimizer.param_groups[0].get("weight_decay", 1e-4))
    trainable = [p for p in net.parameters() if p.requires_grad]
    net.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    return n_frozen


def make_net(net_type: str, n_features: Optional[int], hidden_dim: int, n_blocks: int,
             graph: Optional[DiGraph] = None, device: str = "cpu",
             lr: float = 1e-3, attn_layer: int = 8, n_heads: int = 4,
             num_players: int = 2, zero_value_heads: bool = True,
             value_weight: float = 1.0, own_weight: float = 1.0):
    """Build a GPU net by type ('gnn' | '2dcnn'). F = feature_dim(num_players).

    ``zero_value_heads=False`` leaves value/own heads at their random init —
    required for supervised distillation, where the zero init collapses them.
    ``value_weight`` / ``own_weight`` scale the SGD terms (reported losses stay
    unweighted). Go self-play bats pass the same 30 / 5 as distill stage 1.
    """
    net_type = gpu_net_type(net_type)
    np_ = _player_count(num_players)
    if n_features is None:
        n_features = feature_dim(np_)
    n_features = require_feature_dim(n_features, np_)
    if net_type == "2dcnn":
        net = Cnn2dPolicyValueNet(n_features=n_features, hidden_dim=hidden_dim,
                                  n_blocks=n_blocks, graph=graph,
                                  device=device, lr=lr,
                                  attn_layer=attn_layer, n_heads=n_heads,
                                  num_players=np_, zero_value_heads=zero_value_heads)
    else:
        net = GnnPolicyValueNet(n_features=n_features, hidden_dim=hidden_dim,
                                n_blocks=n_blocks, graph=graph, device=device,
                                lr=lr, attn_layer=attn_layer, n_heads=n_heads,
                                num_players=np_, zero_value_heads=zero_value_heads)
    net.value_weight = float(value_weight)
    net.own_weight = float(own_weight)
    return net


def save_net(net, path: str):
    if not gpu_net_finite(net):
        raise ValueError("refusing to save non-finite GPU weights")
    net_type = gpu_net_type(net.net_type)
    ckpt = {"net_type": net_type,
            "label": gpu_net_label(net_type),
            "num_players": int(net.num_players),
            "model_state_dict": net.state_dict(),
            "n_features": int(net.n_features),
            "hidden_dim": int(net.hidden_dim),
            "n_blocks": int(net.n_blocks),
            "attn_layer": int(net.attn_layer),
            "n_heads": int(net.n_heads)}
    torch.save(ckpt, path)


def peek_gpu_tags(path: str):
    """Return (architecture label, num_players) without constructing the net."""
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "net_type" not in ckpt or "num_players" not in ckpt:
        raise ValueError(f"{path} is not a GKT GPU checkpoint")
    lab = gpu_net_label(ckpt["net_type"])
    if "label" in ckpt and str(ckpt["label"]).strip().upper() != lab:
        raise ValueError(f"{path} label {ckpt['label']!r} != {lab}")
    return lab, tagged_num_players(ckpt)


def peek_gpu_label(path: str) -> str:
    """Read GNN / 2DCNN from a .pt without constructing the net."""
    lab, k = peek_gpu_tags(path)
    return f"{lab} · {k}P"


def load_net(path: str, device: str = "cpu",
             graph: Optional[DiGraph] = None):
    """Load a net saved by `save_net`."""
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "net_type" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a GKT GPU checkpoint")
    if "num_players" not in ckpt:
        raise ValueError(f"{path} missing num_players")
    net_type = gpu_net_type(ckpt["net_type"])
    kwargs = dict(n_features=int(ckpt["n_features"]),
                  hidden_dim=int(ckpt["hidden_dim"]),
                  n_blocks=int(ckpt["n_blocks"]),
                  attn_layer=int(ckpt["attn_layer"]),
                  n_heads=int(ckpt["n_heads"]),
                  num_players=tagged_num_players(ckpt),
                  graph=graph, device=device)
    net = make_net(net_type, **kwargs)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    if not gpu_net_finite(net):
        raise ValueError(f"{path} contains non-finite weights")
    net.eval()
    return net


class _GnnScriptWrap(nn.Module):
    def __init__(self, net: GnnPolicyValueNet):
        super().__init__()
        self.net = net

    def forward(self, x, mask, adj_in, adj_out):
        out = self.net.forward_adj(x, adj_in, adj_out)
        logits = out["policy"].masked_fill(~mask, MASK_VALUE)
        return torch.softmax(logits, dim=-1), out["value"].squeeze(-1), out["stdev"].reshape(-1)


class _Cnn2dScriptWrap(nn.Module):
    def __init__(self, net: Cnn2dPolicyValueNet):
        super().__init__()
        self.net = net

    def forward(self, x, mask, adj_in, adj_out):
        out = self.net.forward(x)
        logits = out["policy"].masked_fill(~mask, MASK_VALUE)
        return torch.softmax(logits, dim=-1), out["value"].squeeze(-1), out["stdev"].reshape(-1)


class JitInfer:
    """TorchScript policy / value / stdev forward for MCTS (`predict_batch` only)."""

    def __init__(self, net, graph: Optional[DiGraph] = None,
                 device: Optional[str] = None):
        self._src = net
        self.net_type = net.net_type
        self.num_players = int(net.num_players)
        self.n_features = int(net.n_features)
        self.device = getattr(net, "device", device or "cpu")
        self.mod = None
        self.adj_in = None
        self.adj_out = None
        if graph is not None:
            self.set_graph(graph)
        elif getattr(net, "adj_in", None) is not None or getattr(net, "_m", None):
            self._retrace()

    def set_graph(self, graph: DiGraph):
        self._src.set_graph(graph)
        self._retrace()
        return self

    def eval(self):
        return self

    def _retrace(self):
        self.mod, self.adj_in, self.adj_out = _trace_policy_value(self._src)

    @torch.no_grad()
    def predict(self, X, legal_mask=None):
        pol, val, _ = self.predict_batch(
            np.asarray(X, dtype=np.float32)[None, ...],
            None if legal_mask is None else np.asarray(legal_mask)[None, ...])
        return pol[0], float(val[0])

    @torch.no_grad()
    def predict_batch(self, X_batch, masks=None):
        x = torch.from_numpy(np.asarray(X_batch, dtype=np.float32)).to(self.device)
        B, n, _ = x.shape
        if masks is None:
            m = torch.ones(B, n + 1, dtype=torch.bool, device=self.device)
        else:
            m = torch.from_numpy(np.asarray(masks) > 0).to(self.device)
        pol, val, stdev = self.mod(x, m, self.adj_in, self.adj_out)
        return (pol.detach().float().cpu().numpy().astype(np.float32),
                val.detach().float().reshape(-1).cpu().numpy().astype(np.float32),
                stdev.detach().float().reshape(-1).cpu().numpy().astype(np.float32))


def _trace_example_and_check(n: int, F: int, device):
    """Non-zero features + mixed legal mask so masked_fill is actually recorded."""
    g = torch.Generator()
    g.manual_seed(1)
    x = torch.randn(2, n, F, generator=g).to(device)
    mask = torch.ones(2, n + 1, dtype=torch.bool, device=device)
    mask[0, -1] = False
    mask[1, 0] = False
    g2 = torch.Generator()
    g2.manual_seed(2)
    x_chk = torch.randn(4, n, F, generator=g2).to(device)
    m_chk = torch.ones(4, n + 1, dtype=torch.bool, device=device)
    m_chk[0, -1] = False
    m_chk[1, 0] = False
    if n > 1:
        m_chk[2, 1] = False
        m_chk[3, n // 2] = False
    return (x, mask), (x_chk, m_chk)


def _assert_trace_matches_eager(wrap, traced, x, mask, ain, aout,
                                rtol: float = 1e-3, atol: float = 1e-4):
    wrap.eval()
    with torch.no_grad():
        e_pol, e_val, e_std = wrap(x, mask, ain, aout)
        t_pol, t_val, t_std = traced(x, mask, ain, aout)
    if (not torch.allclose(e_pol, t_pol, rtol=rtol, atol=atol)
            or not torch.allclose(e_val, t_val, rtol=rtol, atol=atol)
            or not torch.allclose(e_std, t_std, rtol=rtol, atol=atol)):
        dp = (e_pol - t_pol).abs().max().item()
        dv = (e_val - t_val).abs().max().item()
        ds = (e_std - t_std).abs().max().item()
        raise ValueError(
            f"TorchScript output mismatch vs eager (max |d policy|={dp:.3e}, "
            f"|d value|={dv:.3e}, |d stdev|={ds:.3e})")


def _trace_policy_value(net):
    net.eval()
    wrap = _GnnScriptWrap(net) if net.net_type == "gnn" else _Cnn2dScriptWrap(net)
    wrap.eval()
    device = net.device
    F = int(net.n_features)
    if net.net_type == "gnn":
        if net.adj_in is None:
            raise ValueError("call set_graph() before tracing a GNN")
        n = int(net.adj_in.shape[0])
        ain = net.adj_in
        aout = net.adj_out
    else:
        if net._m is None:
            raise ValueError("call set_graph() before tracing a 2DCNN")
        n = int(net._m) * int(net._n_grid)
        ain = torch.zeros(n, n, device=device)
        aout = ain
    example, check = _trace_example_and_check(n, F, device)
    x, mask = example
    x_chk, m_chk = check
    args = (x, mask, ain, aout)
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            try:
                traced = torch.jit.trace(wrap, args, check_trace=True)
            except Exception:
                traced = torch.jit.trace(wrap, args, check_trace=False)
        _assert_trace_matches_eager(wrap, traced, x_chk, m_chk, ain, aout)
        # Freeze only for conv nets: for GNN the adjacency is a runtime input
        # (not an inlinable constant), and jit.freeze forces the dense adj
        # matmul onto a slow path (measured ~4x slower than eager / traced).
        if net.net_type != "gnn":
            try:
                frozen = torch.jit.freeze(traced)
                _assert_trace_matches_eager(wrap, frozen, x_chk, m_chk, ain, aout)
                traced = frozen
            except Exception:
                pass
    return traced, ain, aout


def make_script_infer(net, graph: Optional[DiGraph] = None,
                      device: Optional[str] = None) -> JitInfer:
    """Trace GNN / 2DCNN to TorchScript for MCTS leaf evaluation.

    Raises if the traced (and frozen, if freeze succeeds) module disagrees
    with eager on a random check batch; callers may fall back to `net`.
    """
    return JitInfer(net, graph=graph, device=device)


def maybe_script_infer(net, graph: Optional[DiGraph] = None,
                       device: Optional[str] = None):
    """`make_script_infer`, or `net` if tracing / check fails."""
    try:
        return make_script_infer(net, graph, device)
    except Exception:
        return net


def export_script(net, path: str) -> str:
    """Save traced forward(x, mask, adj_in, adj_out) → policy, value, stdev."""
    traced, _, _ = _trace_policy_value(net)
    traced.save(path)
    return path


# ---------------------------------------------------------------------------
# Self-play worker (top-level so it can be pickled for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _selfplay_worker(graph: DiGraph, weights: Dict[str, np.ndarray],
                     n_features: int, hidden_dim: int, n_blocks: int,
                     n_simulations: int, temperature: float,
                     n_games: int, max_moves: int,
                     seed: Optional[int], device: str = "cpu",
                     batch_size: int = 32,
                     attn_layer: int = 8, n_heads: int = 4,
                     q_lambda: float = 0.5,
                     net_type: str = "gnn",
                     num_players: int = 2,
                     progress_file: Optional[str] = None,
                     main_progress_file: Optional[str] = None,
                     worker_id: int = 0,
                     rules: str = "go",
                     win_length: int = 5) -> List[Tuple]:
    """Run `n_games` self-play games on a copy of the net. Returns samples.

    Each sample is the 12-tuple from `GktSelfPlay.play_one_game`.

    `device` selects where the network forward runs. MCTS select/expand/
    backprop run in C++ (`gkt_native`); leaf evaluation is `predict_batch`
    via `maybe_script_infer` (TorchScript, else eager). On CUDA, the dense
    (n x n) adjacency matmul is much faster on GPU than CPU.
    """
    if seed is not None:
        random.seed(seed + os.getpid())
        np.random.seed(seed + os.getpid())
    pid = os.getpid()
    n = len(graph.vertices)
    tag = f"w{int(worker_id)} pid={pid}"
    progress_append(progress_file,
                    f"{tag} worker_start n={n} games={n_games} "
                    f"sim={n_simulations} batch={batch_size} "
                    f"device={device} net={net_type}")
    net = make_net(net_type, n_features=n_features, hidden_dim=hidden_dim,
                   n_blocks=n_blocks, graph=graph,
                   attn_layer=attn_layer, n_heads=n_heads,
                   device=device, lr=0.0, num_players=num_players)
    net.set_weights(weights)
    net.eval()
    progress_append(progress_file, f"{tag} net_ready tracing...")
    infer = maybe_script_infer(net, graph, device)
    kind = type(infer).__name__
    progress_append(progress_file, f"{tag} infer={kind} starting games")
    driver = GktSelfPlay(graph, infer,
                            n_simulations=n_simulations,
                            temperature=temperature,
                            max_moves=max_moves, batch_size=batch_size,
                            q_lambda=q_lambda,
                            rules=rules, win_length=win_length)
    samples: List[Tuple] = []
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
# Parallel GKT trainer
# ---------------------------------------------------------------------------

class GktTrainer:
    """N C++ self-play workers + GPU batched SGD on the shared net."""

    def __init__(self, graph: DiGraph,
                 n_features: Optional[int] = None, hidden_dim: int = 512,
                 n_blocks: int = 20,
                 device: str = "cuda", lr: float = 1e-3,
                 n_workers: int = 1, games_per_worker: int = 16,
                 n_simulations: int = 800, temperature: float = 1.0,
                 batch_size: int = 128,
                 selfplay_device: str = "cpu", selfplay_batch: int = 32,
                 attn_layer: int = 8, n_heads: int = 4,
                 q_lambda: float = 0.5,
                 steps_per_cycle: int = 4, buffer_capacity: int = 200000,
                 max_moves: Optional[int] = None, log_fn=print,
                 net_type: str = "gnn",
                 num_players: int = 2,
                 progress_file: Optional[str] = None,
                 rules: str = "go",
                 win_length: int = 5,
                 value_weight: float = 1.0,
                 own_weight: float = 1.0,
                 freeze_policy: bool = False):
        self.graph = graph
        self.num_players = int(num_players)
        self.n_features = (feature_dim(self.num_players)
                           if n_features is None else int(n_features))
        require_feature_dim(self.n_features, self.num_players)
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.attn_layer = attn_layer
        self.n_heads = n_heads
        self.q_lambda = q_lambda
        self.n_workers = n_workers
        self.games_per_worker = games_per_worker
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.batch_size = batch_size
        self.selfplay_device = selfplay_device
        self.selfplay_batch = selfplay_batch
        self.steps_per_cycle = steps_per_cycle
        self.buffer_capacity = buffer_capacity
        self.log = log_fn

        self.net_type = net_type
        self.progress_file = progress_file
        self.rules = str(rules)
        self.win_length = int(win_length)
        if is_k_in_row_rules(self.rules):
            self.max_moves = len(graph.vertices)
        else:
            self.max_moves = max_moves or (len(graph.vertices) * 2 + 40)
        self.net = make_net(net_type, n_features=n_features, hidden_dim=hidden_dim,
                            n_blocks=n_blocks, graph=graph,
                            attn_layer=attn_layer, n_heads=n_heads,
                            device=device, lr=lr, num_players=self.num_players,
                            value_weight=value_weight, own_weight=own_weight)
        self.device = device
        self.freeze_policy = bool(freeze_policy)
        if self.freeze_policy:
            n_fr = set_gpu_policy_frozen(self.net, True)
            self.log(f"policy head frozen ({n_fr} tensors); "
                     f"train trunk + value/own/aux")

    def train(self, n_cycles: int = 10, seed: int = 0,
              replay: Optional[List[Tuple]] = None) -> Dict:
        """Run `n_cycles` of {parallel self-play -> GPU training -> sync}.

        `replay` is unused samples already on disk for this graph. New games
        are appended; a successful SGD batch is removed from the buffer.
        """
        losses: List[float] = []
        policy_losses: List[float] = []
        value_losses: List[float] = []
        own_losses: List[float] = []
        buffer: List[Tuple] = list(replay) if replay else []
        n_replay = len(buffer)
        this_round: List[Tuple] = []

        self.log(f"GPU parallel training: graph={len(self.graph.vertices)} vertices, "
                 f"workers={self.n_workers} x {self.games_per_worker} games, "
                 f"sim={self.n_simulations}, device={self.device}, "
                 f"batch={self.batch_size} steps={self.steps_per_cycle} "
                 f"value_weight={getattr(self.net, 'value_weight', 1.0):g} "
                 f"own_weight={getattr(self.net, 'own_weight', 1.0):g}"
                 f"{' policy=FROZEN' if self.freeze_policy else ''}")

        weights = self.net.get_weights()
        # Use 'spawn' on all platforms via mp_context for consistency; the
        # default on Windows is already spawn.
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx)

        new_samples = 0
        try:
            for cycle in range(n_cycles):
                t0 = time.time()

                # 1. parallel self-play with the current weights
                futures = [executor.submit(
                    _selfplay_worker, self.graph, weights,
                    self.n_features, self.hidden_dim, self.n_blocks,
                    self.n_simulations, self.temperature,
                    self.games_per_worker, self.max_moves,
                    seed + cycle * 1000 + i, self.selfplay_device,
                    self.selfplay_batch, self.attn_layer,
                    self.n_heads, self.q_lambda, self.net_type,
                    self.num_players,
                    worker_progress_path(self.progress_file, i),
                    self.progress_file,
                    i, self.rules, self.win_length) for i in range(self.n_workers)]

                new_samples = 0
                for fut in futures:
                    samples = fut.result()
                    this_round.extend(samples)
                    buffer.extend(samples)
                    new_samples += len(samples)
                if len(buffer) > self.buffer_capacity:
                    buffer = buffer[-self.buffer_capacity:]

                # 2. GPU batched training
                cycle_losses = []
                cycle_policy_losses = []
                cycle_value_losses = []
                cycle_own_losses = []
                cycle_aux_losses = []
                if len(buffer) >= self.batch_size:
                    for _ in range(self.steps_per_cycle):
                        if len(buffer) < self.batch_size:
                            break
                        idx = random.sample(range(len(buffer)), self.batch_size)
                        batch = [buffer[i] for i in idx]
                        X = np.stack([s[0] for s in batch])
                        mask = np.stack([s[1] for s in batch])
                        pol = np.stack([s[2] for s in batch])
                        z = np.array([s[4] for s in batch], dtype=np.float32)
                        own = np.stack([s[5] for s in batch])
                        ploss, vloss, oloss, aloss, loss = self.net.train_on_batch(
                            X, mask, pol, z, own, stack_aux(batch))
                        if all(math.isfinite(x) for x in (ploss, vloss, oloss, aloss, loss)):
                            drop = set(idx)
                            buffer = [s for i, s in enumerate(buffer) if i not in drop]
                            cycle_losses.append(loss)
                            cycle_policy_losses.append(ploss)
                            cycle_value_losses.append(vloss)
                            cycle_own_losses.append(oloss)
                            cycle_aux_losses.append(aloss)
                            losses.append(loss)
                            policy_losses.append(ploss)
                            value_losses.append(vloss)
                            own_losses.append(oloss)
                        else:
                            self.log("skip SGD step: non-finite loss (batch kept)")
                            if not gpu_net_finite(self.net):
                                self.log("abort cycle SGD: non-finite weights")
                                break

                # 3. sync weights for the next cycle
                weights = self.net.get_weights()

                avg_policy = (sum(cycle_policy_losses) / len(cycle_policy_losses)
                              if cycle_policy_losses else float("nan"))
                avg_value = (sum(cycle_value_losses) / len(cycle_value_losses)
                              if cycle_value_losses else float("nan"))
                avg_own = (sum(cycle_own_losses) / len(cycle_own_losses)
                           if cycle_own_losses else float("nan"))
                avg_aux = (sum(cycle_aux_losses) / len(cycle_aux_losses)
                           if cycle_aux_losses else float("nan"))
                self.log(f"[cycle {cycle+1}/{n_cycles}] "
                         f"+{new_samples} samples "
                         f"(buffer={len(buffer)} replay={n_replay}), "
                         f"ploss={avg_policy:.4f} vloss={avg_value:.4f} "
                         f"oloss={avg_own:.4f} aloss={avg_aux:.4f}, "
                         f"{time.time()-t0:.1f}s")
        finally:
            executor.shutdown(wait=True)

        return {"losses": losses, "policy_losses": policy_losses,
                "value_losses": value_losses, "own_losses": own_losses,
                "buffer_size": len(buffer), "new_samples": new_samples,
                "round_samples": this_round, "unused": buffer}
