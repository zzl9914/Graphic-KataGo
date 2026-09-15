"""Training-time vertex permutations.

Permute occupancy, policy, and ownership together. Both search (Arena / UI)
and training self-play relabel via ``SearchAugNet``: training self-play draws a
fresh random S_n perm on *every* ``predict_batch`` (i.e. every leaf evaluation
the net reads), so the net can never latch onto a stable vertex numbering
("index superstition"). See ``ref/algorithm.md`` §8.5.

Relabeling is dispatched by net kind (index-superstition defense):

- **GNN / 1DCNN / MLP** (no board geometry): random S_n full relabeling.
  **GNN** also permutes ``adj`` so the edge relation is unchanged; 1DCNN / MLP
  just permute features (their convolution / layer order is a "fake geometry",
  so an arbitrary relabeling is the ablation itself).
- **2DCNN** (real board geometry): board symmetry only — D4 (square) / Klein
  four-group (rectangle) / torus shift, via ``random_graph_perm``. An arbitrary
  S_n would pull non-adjacent vertices into a conv neighborhood and break
  the grid prior. Pass slot is never permuted.

``random_graph_perm`` / D4 helpers are automorphism maps for the **grid2d**
geometry only (not the SGD/search default, which is random S_n):

- 2D rectangle: dihedral (D4 if square, Klein four-group if not).
- 2D torus: that dihedral plus a random lattice shift (circular).
"""
from __future__ import annotations

import numpy as np


def _invert_dest(dest: np.ndarray) -> np.ndarray:
    """dest[src] = destination index → gather perm[i] = source of cell i."""
    perm = np.empty_like(dest)
    perm[dest] = np.arange(dest.shape[0], dtype=np.int64)
    return perm


def grid_sym_perm(m: int, n: int, reflect: bool, rot: int) -> np.ndarray:
    """perm[i] = source index for new cell i (gather: Y = X[perm]).

    `rot` is a number of 90° clockwise turns. Only 0 or 2 are valid when m≠n.
    """
    r = np.arange(m, dtype=np.int64)[:, None]
    c = np.arange(n, dtype=np.int64)[None, :]
    rr = np.broadcast_to(r, (m, n)).copy()
    cc = np.broadcast_to(c, (m, n)).copy()
    if reflect:
        cc = n - 1 - cc
    rot = int(rot) % 4
    if rot == 1:
        rr, cc = cc, m - 1 - rr
    elif rot == 2:
        rr, cc = m - 1 - rr, n - 1 - cc
    elif rot == 3:
        rr, cc = n - 1 - cc, rr
    nn = n if rot % 2 == 0 else m
    dest = (rr * nn + cc).reshape(-1)
    return _invert_dest(dest)


def random_grid_perm(m: int, n: int, rng=None, toroidal: bool = False) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    reflect = bool(rng.integers(0, 2))
    if m == n:
        rot = int(rng.integers(0, 4))
    else:
        rot = int(rng.choice((0, 2)))
    perm = grid_sym_perm(m, n, reflect, rot)
    if toroidal:
        dr = int(rng.integers(0, m))
        dc = int(rng.integers(0, n))
        r = np.arange(m * n, dtype=np.int64) // n
        c = np.arange(m * n, dtype=np.int64) % n
        src = ((r - dr) % m) * n + ((c - dc) % n)
        perm = perm[src]
    return perm


def random_graph_perm(graph, rng=None):
    """Return a grid2d board-symmetry gather perm, or None for any other graph.

    Only ``grid2d`` (2D rectangle / torus) has a known automorphism map that
    the 2DCNN's spatial prior respects. Every other graph (random, line,
    sphere, cubic, triangular, hollow, …) returns None so callers fall back to
    random S_n.
    """
    sym = getattr(graph, "sym", None)
    if not sym:
        return None
    rng = np.random.default_rng() if rng is None else rng
    if sym.get("type") == "grid2d":
        return random_grid_perm(sym["m"], sym["n"], rng,
                                toroidal=bool(sym.get("toroidal")))
    return None


def apply_vertex_perms(X, mask, policy, perms, ownership=None,
                       extra_policy=None, extra_vertex=None):
    """Gather each row by ``perms[i]``. Pass slot (index n) is not permuted."""
    extra_policy = list(extra_policy or [])
    extra_vertex = list(extra_vertex or [])
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    policy = np.asarray(policy, dtype=np.float32)
    ownership = (None if ownership is None
                 else np.asarray(ownership, dtype=np.float32))
    perms = np.asarray(perms, dtype=np.int64)
    B, n_vert, _ = X.shape
    X_out = np.empty_like(X)
    mask_out = np.empty_like(mask)
    pol_out = np.empty_like(policy)
    own_out = None if ownership is None else np.empty_like(ownership)
    ep_out = [np.empty_like(a) for a in extra_policy]
    ev_out = [np.empty_like(a) for a in extra_vertex]
    pass_idx = n_vert
    for i in range(B):
        perm = perms[i]
        X_out[i] = X[i, perm]
        mask_out[i, :n_vert] = mask[i, perm]
        mask_out[i, pass_idx] = mask[i, pass_idx]
        pol_out[i, :n_vert] = policy[i, perm]
        pol_out[i, pass_idx] = policy[i, pass_idx]
        if own_out is not None:
            own_out[i] = ownership[i, perm]
        for k, a in enumerate(extra_policy):
            ep_out[k][i, :n_vert] = a[i, perm]
            ep_out[k][i, pass_idx] = a[i, pass_idx]
        for k, a in enumerate(extra_vertex):
            ev_out[k][i] = a[i, perm]
    return X_out, mask_out, pol_out, own_out, ep_out, ev_out


def augment_vertex_batch(X, mask, policy, rng=None, ownership=None,
                         extra_policy=None, extra_vertex=None):
    """Random S_n relabeling per row. Pass slot is not permuted.

    For a GNN, permute ``adj`` with the returned gather perms so the
    edge relation is the same graph. CNN spatial nets should not use this.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float32)
    B, n_vert, _ = X.shape
    perms = np.stack([rng.permutation(n_vert) for _ in range(B)]).astype(np.int64)
    Xo, Mo, Po, Oo, ep, ev = apply_vertex_perms(
        X, mask, policy, perms, ownership=ownership,
        extra_policy=extra_policy, extra_vertex=extra_vertex)
    if Oo is None:
        Oo = np.asarray(ownership, dtype=np.float32) if ownership is not None else None
    return Xo, Mo, Po, Oo, ep, ev, perms


def augment_graph_batch(X, mask, policy, graph, rng=None, ownership=None,
                        extra_policy=None, extra_vertex=None):
    """Random automorphism per row (CNN). Pass slot is not permuted.

    extra_policy: list of (B, n+1) arrays (opp / soft targets).
    extra_vertex: list of (B, n) arrays (future occupancy, …).
    """
    extra_policy = list(extra_policy or [])
    extra_vertex = list(extra_vertex or [])
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    policy = np.asarray(policy, dtype=np.float32)
    ownership = np.asarray(ownership, dtype=np.float32)
    if not getattr(graph, "sym", None):
        return X, mask, policy, ownership, extra_policy, extra_vertex
    rng = np.random.default_rng() if rng is None else rng
    B, n_vert, _ = X.shape
    perms = []
    identity = np.arange(n_vert, dtype=np.int64)
    for _ in range(B):
        perm = random_graph_perm(graph, rng)
        perms.append(identity if perm is None else perm)
    return apply_vertex_perms(
        X, mask, policy, np.stack(perms), ownership=ownership,
        extra_policy=extra_policy, extra_vertex=extra_vertex)


def infer_net_kind(net) -> str:
    """``gnn`` / ``2dcnn`` / ``mlp`` / ``1dcnn``, or empty if unknown."""
    nt = getattr(net, "net_type", None)
    if nt:
        return str(nt)
    try:
        from gkt_cpu import cpu_net_type
        return cpu_net_type(net)
    except (ValueError, TypeError, ImportError):
        return ""


def search_gather_perm(net, graph, n, rng=None) -> np.ndarray:
    """Gather perm for one MCTS search, chosen by net kind.

    GNN / 1DCNN / MLP (no board geometry): random S_n — any vertex relabeling
    is legitimate, so this is the index-superstition defense.

    2DCNN (board geometry): a board symmetry only — grid2d D4 (square) / Klein
    four (rectangle) / torus shift via ``random_graph_perm``. An arbitrary S_n
    would pull non-adjacent vertices into a conv neighborhood and break the
    2DCNN's grid prior. Falls back to S_n on any non-grid2d graph.
    """
    rng = np.random.default_rng() if rng is None else rng
    if infer_net_kind(net) == "2dcnn" and graph is not None:
        perm = random_graph_perm(graph, rng)
        if perm is not None:
            return perm.astype(np.int64)
    return rng.permutation(int(n)).astype(np.int64)


class SearchAugNet:
    """MCTS leaf wrapper: relabel vertices, map policy back.

    GNN also permutes ``adj`` (JIT ``mod`` or eager ``forward_adj``). CNN / MLP
    only permute features.

    ``fresh_each_batch=False`` (default): one gather perm per search — call
    ``begin_search()`` before each ``native.search`` (Arena / UI eval).

    ``fresh_each_batch=True``: a new random perm on *every* ``predict_batch``,
    so each leaf evaluation the net reads sees a fresh vertex numbering.
    Training self-play uses this to prevent index superstition.
    """

    def __init__(self, inner, graph, fresh_each_batch: bool = False):
        self._inner = inner
        self._graph = graph
        self._kind = infer_net_kind(inner)
        self._perm = None
        self._fresh = bool(fresh_each_batch)
        self.net_type = getattr(inner, "net_type", self._kind or None)
        self.num_players = getattr(inner, "num_players", 2)

    def begin_search(self):
        self._perm = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def predict(self, X, legal_mask=None):
        m = None if legal_mask is None else np.asarray(legal_mask)[None, ...]
        pol, val, *rest = self.predict_batch(
            np.asarray(X, dtype=np.float32)[None, ...], m)
        return (pol[0], float(np.asarray(val).reshape(-1)[0]))

    def predict_batch(self, X_batch, masks=None):
        X = np.asarray(X_batch, dtype=np.float32)
        B, n, _ = X.shape
        if self._fresh:
            perm = search_gather_perm(self._inner, self._graph, n)
        else:
            if self._perm is None or int(self._perm.shape[0]) != n:
                self._perm = search_gather_perm(self._inner, self._graph, n)
            perm = self._perm
        Xp = np.ascontiguousarray(X[:, perm, :])
        if masks is None:
            Mp = None
        else:
            M = np.asarray(masks)
            Mp = np.empty_like(M)
            Mp[:, :n] = M[:, perm]
            Mp[:, n] = M[:, n]
        if self._kind == "gnn" and getattr(self._inner, "adj_in", None) is not None:
            pol, val, stdev = self._gnn_forward(Xp, Mp, perm)
            ret3 = True
        else:
            ret = self._inner.predict_batch(Xp, Mp)
            pol, val = ret[0], ret[1]
            stdev = ret[2] if len(ret) > 2 else None
            ret3 = stdev is not None
        pol = np.asarray(pol, dtype=np.float32)
        pol_out = np.empty_like(pol)
        pol_out[:, perm] = pol[:, :n]
        pol_out[:, n] = pol[:, n]
        val = np.asarray(val, dtype=np.float32).reshape(-1)
        if ret3:
            st = np.asarray(stdev, dtype=np.float32).reshape(-1)
            return pol_out, val, st
        return pol_out, val

    def _gnn_forward(self, Xp, Mp, perm):
        import torch
        inner = self._inner
        device = getattr(inner, "device", "cpu")
        x = torch.from_numpy(Xp).float().to(device)
        B, n, _ = x.shape
        if Mp is None:
            m = torch.ones(B, n + 1, dtype=torch.bool, device=device)
        else:
            m = torch.from_numpy(np.asarray(Mp) > 0).to(device)
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=device)
        ain = inner.adj_in.index_select(0, perm_t).index_select(1, perm_t)
        aout = inner.adj_out.index_select(0, perm_t).index_select(1, perm_t)
        with torch.no_grad():
            mod = getattr(inner, "mod", None)
            if mod is not None:
                pol, val, stdev = mod(x, m, ain, aout)
            else:
                from gkt_gpu import MASK_VALUE
                out = inner.forward_adj(x, ain, aout)
                logits = out["policy"].masked_fill(~m, MASK_VALUE)
                pol = torch.softmax(logits, dim=-1)
                val = out["value"].squeeze(-1)
                stdev = out["stdev"].reshape(-1)
        return (pol.detach().float().cpu().numpy().astype(np.float32),
                val.detach().float().reshape(-1).cpu().numpy().astype(np.float32),
                stdev.detach().float().reshape(-1).cpu().numpy().astype(np.float32))

