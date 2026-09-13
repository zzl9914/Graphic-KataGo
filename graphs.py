"""Board graphs: directed G=(V,E) and built-in presets.

Undirected boards are stored as bidirectional arcs. Weights and features do
not depend on |V|; adjacency is data for the GNN.

Graph-Go rules: ``ref/rules.md``. Gomoku / Anti-Gomoku (grid win, graph features):
``ref/gomoku.md``. Layout of this module: ``ref/implementation.md``.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any, Optional
import random


# ---------------------------------------------------------------------------
# Core graph data structure
# ---------------------------------------------------------------------------

class DiGraph:
    """A directed graph with labeled vertices.

    Attributes:
        vertices: list of vertex labels (order is the "index" of each vertex).
        out_adj: dict label -> set of labels reachable via an outgoing edge.
        in_adj:  dict label -> set of labels that have an outgoing edge into it.
    """

    def __init__(self, vertices: Iterable, edges: Iterable[Tuple],
                 grid: Optional[Tuple] = None):
        self.vertices: List = list(vertices)
        # Optional grid metadata: (rows, cols, toroidal) for graphs that are
        # actually m x n rectangular grids in row-major vertex order. Used by
        # the 2DCNN (Cnn2dPolicyValueNet) to reshape the (n, F) vertex features
        # into an image and to pick circular vs zero padding. None for non-grid
        # graphs (random, line, cubic, sphere, triangular, ...).
        self.grid: Optional[Tuple] = grid
        # Optional grid2d automorphism hint (`grid_sym.py` D4 / torus maps),
        # derived from `grid`. Only the 2DCNN uses it (SGD + Arena/UI search);
        # GNN / 1DCNN / MLP relabel with random S_n instead, so non-grid graphs
        # carry no hint.
        self.sym: Optional[Dict] = (
            {"type": "grid2d", "m": int(grid[0]), "n": int(grid[1]),
             "toroidal": bool(grid[2])} if grid is not None else None)
        idx = {v: i for i, v in enumerate(self.vertices)}
        # sanity: unique labels
        if len(idx) != len(self.vertices):
            raise ValueError("Vertex labels must be unique.")
        # cached label -> index map for O(1) lookups (performance-critical)
        self._idx: Dict = idx
        self.out_adj: Dict = {v: set() for v in self.vertices}
        self.in_adj: Dict = {v: set() for v in self.vertices}
        for (u, v) in edges:
            if u not in idx or v not in idx:
                raise ValueError(f"Edge ({u},{v}) references unknown vertex.")
            self.out_adj[u].add(v)
            self.in_adj[v].add(u)

    def index_of(self, v) -> int:
        """O(1) lookup of a vertex label's index."""
        return self._idx[v]

    # convenience
    @property
    def n(self) -> int:
        return len(self.vertices)

    def out_neighbors(self, v) -> set:
        return self.out_adj[v]

    def in_neighbors(self, v) -> set:
        return self.in_adj[v]

    def neighbors(self, v) -> set:
        """Union of in & out neighbors (== neighbors for undirected graphs)."""
        return self.out_adj[v] | self.in_adj[v]

    def is_undirected(self) -> bool:
        """True iff every edge has its reverse."""
        for u in self.vertices:
            for v in self.out_adj[u]:
                if u not in self.out_adj[v]:
                    return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertices": list(self.vertices),
            "out_adj": {str(k): sorted(list(s)) for k, s in self.out_adj.items()},
            "in_adj":  {str(k): sorted(list(s)) for k, s in self.in_adj.items()},
            "undirected": self.is_undirected(),
            "n": self.n,
            "m": sum(len(s) for s in self.out_adj.values()),
        }


def _square_grid(m: int, n: int, toroidal: bool = False, name: str = None,
                 connectivity: int = 4) -> DiGraph:
    """m x n rectangular grid as an undirected (bi-directional) digraph.

    Vertex labels are A_{i*n+j}, i in [0,m), j in [0,n).
    Edges: 4-neighbor (horizontal + vertical). If ``connectivity=8``, also
    add the two diagonals. If toroidal, wrap edges connect first/last row
    and column (and diagonals wrap independently).
    """
    V = [f"A{i*n+j}" for i in range(m) for j in range(n)]
    E = set()
    dirs4 = ((0, 1), (1, 0))
    dirs8 = ((0, 1), (1, 0), (1, 1), (1, -1))
    dirs = dirs8 if int(connectivity) == 8 else dirs4
    for i in range(m):
        for j in range(n):
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if toroidal:
                    if m > 1:
                        ni %= m
                    if n > 1:
                        nj %= n
                    if (ni, nj) == (i, j):
                        continue
                    E.add((i * n + j, ni * n + nj))
                else:
                    if 0 <= ni < m and 0 <= nj < n:
                        E.add((i * n + j, ni * n + nj))
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    edges = list(set(edges))
    suffix = "_8n" if int(connectivity) == 8 else ""
    label = name or f"{m}x{n}_grid{'_torus' if toroidal else ''}{suffix}"
    return DiGraph(V, edges, grid=(m, n, toroidal)), label


def _sphere_grid(m: int, n: int, name: str = None) -> DiGraph:
    """m x n grid with left-right wrap (cylinder) plus two poles P0, P1.

    P0 connects to every vertex in row 0 (one pole for the "top"),
    P1 connects to every vertex in row m-1 (the other pole for the "bottom").
    This matches the user's graph #4 spec: 19x19 + P0,P1.
    """
    V = [f"A{i*n+j}" for i in range(m) for j in range(n)] + ["P0", "P1"]
    E = set()
    for i in range(m):
        for j in range(n):
            cur = i*n+j
            if j+1 < n:
                E.add((cur, cur+1))
            else:
                E.add((cur, i*n))  # left-right wrap (cylinder)
            if i+1 < m:
                E.add((cur, cur+n))
    # poles: P0 to row 0, P1 to row m-1
    for j in range(n):
        E.add((0*n+j, len(V)-2))      # A_{j} - P0  (row 0)
        E.add(((m-1)*n+j, len(V)-1))  # A_{(m-1)*n+j} - P1 (last row)
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    edges = list(set(edges))
    label = name or f"{m}x{n}_sphere"
    return DiGraph(V, edges), label


def _triangular_diamond(m: int, n: int, toroidal: bool = False, name: str = None) -> DiGraph:
    """m x n triangular-grid rhombus (triangular lattice) as a bi-directional digraph.

    Matches user's graph #5 (m=n=19). Vertices A_{i*n+j}, i in [0,m), j in [0,n).
    Three edge directions per vertex:
      - horizontal (i,j)-(i,j+1)
      - vertical   (i,j)-(i+1,j)
      - diagonal   (i,j)-(i+1,j+1)
    If toroidal, each direction wraps to the opposite side; in particular the
    diagonal direction also wraps horizontally (connecting the last column to
    the next row's first column), so the left-right wrap carries a diagonal edge.
    """
    V = [f"A{i*n+j}" for i in range(m) for j in range(n)]
    E = set()
    for i in range(m):
        for j in range(n):
            cur = i*n+j
            # horizontal
            if j+1 < n:
                E.add((cur, cur+1))
            elif toroidal and n > 1:
                E.add((cur, i*n))  # wrap to first column
            # vertical
            if i+1 < m:
                E.add((cur, cur+n))
            elif toroidal and m > 1:
                E.add((cur, j))  # wrap to first row
            # diagonal (i+1, j+1): wrap each overflowing axis independently
            if i+1 < m and j+1 < n:
                E.add((cur, cur+n+1))
            elif toroidal:
                ni, nj = i+1, j+1
                if ni < m and nj == n and n > 1:
                    # horizontal wrap -> (i+1, 0)
                    E.add((cur, ni*n))
                if nj < n and ni == m and m > 1:
                    # vertical wrap -> (0, j+1)
                    E.add((cur, nj))
                if ni == m and nj == n and m > 1 and n > 1:
                    # both wrap -> (0, 0)
                    E.add((cur, 0))
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    edges = list(set(edges))
    label = name or f"{m}x{n}_tri_diamond{'_torus' if toroidal else ''}"
    return DiGraph(V, edges), label


def _cubic_grid(nx: int, ny: int, nz: int, toroidal: bool = False, name: str = None) -> DiGraph:
    """nx x ny x nz cubic lattice, undirected. Matches user's graph #6 (19^3).

    Vertex A_{i*ny*nz + j*nz + k}, i in [0,nx), j in [0,ny), k in [0,nz).
    Edges along the x (k), y (j), and z (i) axes between adjacent cells.
    If toroidal, each axis wraps to the opposite side.
    """
    V = [f"A{i*ny*nz + j*nz + k}" for i in range(nx) for j in range(ny) for k in range(nz)]
    E = set()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cur = i*ny*nz + j*nz + k
                # x-axis (k direction)
                if k+1 < nz:
                    E.add((cur, cur+1))
                elif toroidal and nz > 1:
                    E.add((cur, i*ny*nz + j*nz))  # wrap to k=0
                # y-axis (j direction)
                if j+1 < ny:
                    E.add((cur, cur+nz))
                elif toroidal and ny > 1:
                    E.add((cur, i*ny*nz + k))  # wrap to j=0
                # z-axis (i direction)
                if i+1 < nx:
                    E.add((cur, cur+ny*nz))
                elif toroidal and nx > 1:
                    E.add((cur, j*nz + k))  # wrap to i=0
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    edges = list(set(edges))
    label = name or f"{nx}x{ny}x{nz}_cubic{'_torus' if toroidal else ''}"
    return DiGraph(V, edges), label


def _line_graph(n: int, name: str = None) -> DiGraph:
    """n-vertex path graph, undirected. Matches user's graph #7."""
    V = [f"A{i}" for i in range(n)]
    E = set()
    for i in range(n-1):
        E.add((i, i+1))
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    label = name or f"{n}_line"
    return DiGraph(V, edges), label


def _hollow_cubic_grid(outer: int, inner: int, name: str = None) -> DiGraph:
    """outer^3 cubic lattice with the central inner^3 block removed.

    Matches user's graph #4.5 (outer=9, inner=7): the 7×7×7 core is hollowed
    out, leaving a cubic shell of thickness (outer-inner)/2 per side.

    Vertices are those (i,j,k) with i,j,k in [0,outer) that are NOT all in
    the interior range [(outer-inner)//2, (outer-inner)//2 + inner).
    """
    s = outer
    lo = (outer - inner) // 2
    hi = lo + inner  # exclusive
    def is_shell(i, j, k):
        return not (lo <= i < hi and lo <= j < hi and lo <= k < hi)
    V = []
    idx_map = {}
    for i in range(s):
        for j in range(s):
            for k in range(s):
                if is_shell(i, j, k):
                    idx_map[(i, j, k)] = len(V)
                    V.append(f"A{i*s*s+j*s+k}")
    E = set()
    for (i, j, k), ci in idx_map.items():
        for di, dj, dk in [(1,0,0),(0,1,0),(0,0,1)]:
            ni, nj, nk = i+di, j+dj, k+dk
            if (ni, nj, nk) in idx_map:
                E.add((ci, idx_map[(ni, nj, nk)]))
    edges = [(V[a], V[b]) for (a, b) in E] + [(V[b], V[a]) for (a, b) in E]
    label = name or f"{outer}^3_hollow_{inner}^3"
    return DiGraph(V, edges), label


def _random_bounded_degree_graph(n: int, seed: int, name: str = None) -> DiGraph:
    """Random directed graph on n labeled vertices with every in-degree and
    out-degree at most 6. Deterministic given (n, seed).

    Construction:
      1. A random directed cycle through a random permutation guarantees the
         graph is strongly connected (no isolated vertices) and gives each
         vertex in-degree = out-degree = 1.
      2. Random extra arcs are then added (respecting the degree caps) to
         reach a moderate density (~4 average out-degree), giving the "random"
         look while keeping every degree bounded at 6.
    """
    rng = random.Random(seed)
    V = [f"V{i}" for i in range(n)]
    perm = list(range(n))
    rng.shuffle(perm)
    arcs = set()
    out_deg = [0] * n
    in_deg = [0] * n
    for i in range(n):
        u, v = perm[i], perm[(i + 1) % n]
        arcs.add((u, v))
        out_deg[u] += 1
        in_deg[v] += 1
    # extra random arcs: ~3n more, average out-degree ~4 (capped at 6)
    target_extra = 3 * n
    added = 0
    attempts = 0
    max_attempts = 60 * n
    while added < target_extra and attempts < max_attempts:
        u = rng.randrange(n)
        v = rng.randrange(n)
        attempts += 1
        if u == v or (u, v) in arcs:
            continue
        if out_deg[u] >= 6 or in_deg[v] >= 6:
            continue
        arcs.add((u, v))
        out_deg[u] += 1
        in_deg[v] += 1
        added += 1
    edges = [(V[a], V[b]) for (a, b) in arcs]
    label = name or f"random_{n}_deg6"
    return DiGraph(V, edges), label


# Gomoku rotation (engine keys). UI aliases: 0→G19, 0.5→G7.
GOMOKU_TRAIN_KEYS = ("0", "0.5", "1", "3", "G9", "G15", "G7d", "G9d")
# Extra Gomoku-only boards; Graph-Go default training skips these (0 / 0.5 stay).
GOMOKU_KEYS = frozenset({"G9", "G15", "G7d", "G9d"})

_NET_DIR = {"gnn": "gnn", "mlp": "mlp", "1dcnn": "cnn1d", "2dcnn": "cnn2d"}

RULES_CHOICES = ("go", "gomoku", "antigomoku")


def normalize_rules(rules: str) -> str:
    t = "".join(c for c in str(rules).lower() if c not in "-_")
    if t == "antigomoku":
        return "antigomoku"
    if t == "gomoku":
        return "gomoku"
    if t in ("go", "graphgo"):
        return "go"
    return t


def is_k_in_row_rules(rules: str) -> bool:
    return normalize_rules(rules) in ("gomoku", "antigomoku")


def net_mod_dir(net: str) -> str:
    """Folder suffix matching start_train.bat (cnn1d / cnn2d, not 1dcnn / 2dcnn)."""
    k = str(net).lower()
    return _NET_DIR.get(k, k)


def default_train_outdir(rules: str, net: str) -> str:
    """Relative to ``scr/``: ``cur_mod_*``, ``cur_mod_gomoku_*``, or ``cur_mod_antigomoku_*``."""
    kind = net_mod_dir(net)
    r = normalize_rules(rules)
    if r == "antigomoku":
        return f"../cur_mod_antigomoku_{kind}"
    if r == "gomoku":
        return f"../cur_mod_gomoku_{kind}"
    return f"../cur_mod_{kind}"


# ---------------------------------------------------------------------------
# Built-in preset registry
# ---------------------------------------------------------------------------

def builtin_graphs() -> Dict[str, Dict[str, Any]]:
    """Return the built-in example graphs from the user's spec.

    Keys '0', '0.5', '1'..'7', '4.5', '5.5', Gomoku G9/G15/G7d/G9d, and 'R1'..'R5'.
    '0' is 19×19 (Gomoku UI: G19), '0.5' is 7×7 (Gomoku UI: G7),
    '2' is 61×61 (Gomoku UI: G61).
    """
    out = {}

    # 0. standard 19x19 Go board (the baseline)
    g, label = _square_grid(19, 19, name="19x19_standard")
    out["0"] = {"graph": g, "label": label, "spec": "19x19 standard Go grid",
                "notes": "Baseline 19x19 board."}

    # 0.5. 7x7 small board
    g, label = _square_grid(7, 7, name="7x7_grid")
    out["0.5"] = {"graph": g, "label": label, "spec": "7x7 square grid",
                  "notes": "Small board for fast training checks. Gomoku UI shows this as G7."}

    # Extra Gomoku sizes / 8-neighbor ablations. 7×7 4-neighbor is key 0.5 (UI: G7).
    g, label = _square_grid(9, 9, name="gomoku_9x9_4n", connectivity=4)
    out["G9"] = {"graph": g, "label": label, "spec": "9x9 Gomoku (4-neighbor GNN)",
                 "notes": "Standard-ish small Gomoku; diagonals are not graph edges."}
    g, label = _square_grid(15, 15, name="gomoku_15x15_4n", connectivity=4)
    out["G15"] = {"graph": g, "label": label, "spec": "15x15 Gomoku (4-neighbor GNN)",
                 "notes": "Freestyle Gomoku board; GNN still only sees 4-neighbors."}
    g, label = _square_grid(7, 7, name="gomoku_7x7_8n", connectivity=8)
    out["G7d"] = {"graph": g, "label": label, "spec": "7x7 Gomoku (8-neighbor GNN)",
                 "notes": "Diagonals are graph edges; a snake of 5 is also connected."}
    g, label = _square_grid(9, 9, name="gomoku_9x9_8n", connectivity=8)
    out["G9d"] = {"graph": g, "label": label, "spec": "9x9 Gomoku (8-neighbor GNN)",
                 "notes": "8-neighbor ablation: connectivity ≠ collinearity."}

    # 1. 27x13 rectangular grid
    g, label = _square_grid(27, 13, name="27x13_grid")
    out["1"] = {"graph": g, "label": label, "spec": "27x13 rectangular grid",
                "notes": "Aspect ratio ~2.08; longer sides and corners."}

    # 2. 61x61 large square grid
    g, label = _square_grid(61, 61, name="61x61_grid")
    out["2"] = {"graph": g, "label": label, "spec": "61x61 square grid",
                "notes": "Large board (3721 vertices); excluded from default training."}

    # 3. 19x19 torus
    g, label = _square_grid(19, 19, toroidal=True, name="19x19_torus")
    out["3"] = {"graph": g, "label": label, "spec": "19x19 toroidal square grid",
                "notes": "No boundary; no corners or edges."}

    # 4. 19x19 sphere + 2 poles
    g, label = _sphere_grid(19, 19, name="19x19_sphere")
    out["4"] = {"graph": g, "label": label, "spec": "19x19 sphere lat/lon + 2 poles",
                "notes": "Only two singularities (poles); elsewhere locally planar."}

    # 4.5. 9^3 cubic shell with central 7^3 hollowed out
    g, label = _hollow_cubic_grid(9, 7, name="9cubed_hollow7")
    out["4.5"] = {"graph": g, "label": label, "spec": "9^3 cube minus central 7^3 shell",
                  "notes": "One-cell-thick cubic shell with inner and outer faces."}

    # 5. 19x19 triangular diamond (rhombus, bounded)
    g, label = _triangular_diamond(19, 19, name="19x19_tri_diamond")
    out["5"] = {"graph": g, "label": label, "spec": "19x19 triangular diamond grid",
                "notes": "Interior degree up to 6; denser adjacency than square grid."}

    # 5.5. 19x19 triangular torus (wrap on all three lattice directions)
    g, label = _triangular_diamond(19, 19, toroidal=True, name="19x19_tri_torus")
    out["5.5"] = {"graph": g, "label": label,
                  "spec": "19x19 toroidal triangular diamond",
                  "notes": "No boundary; lattice shift is a graph automorphism."}

    # 6. 19^3 cubic
    g, label = _cubic_grid(19, 19, 19, name="19_cubed")
    out["6"] = {"graph": g, "label": label, "spec": "19^3 cubic grid",
                "notes": "3D board, 6859 vertices; excluded from default training, playable in the UI."}

    # 7. 19-vertex line
    g, label = _line_graph(19, name="19_line")
    out["7"] = {"graph": g, "label": label, "spec": "19-vertex path",
                "notes": "1D board; smallest engine/AI sanity check."}

    # R1..R5. random bounded-degree directed graphs (n < 400, in/out degree <= 6)
    g, label = _random_bounded_degree_graph(120, 101, name="random_R1_120")
    out["R1"] = {"graph": g, "label": label, "spec": "120-vertex random digraph, deg<=6",
                 "notes": "Strongly connected via a random directed cycle; in/out degree <= 6."}

    g, label = _random_bounded_degree_graph(256, 202, name="random_R2_256")
    out["R2"] = {"graph": g, "label": label, "spec": "256-vertex random digraph, deg<=6",
                 "notes": "Strongly connected via a random directed cycle; in/out degree <= 6."}

    g, label = _random_bounded_degree_graph(300, 303, name="random_R3_300")
    out["R3"] = {"graph": g, "label": label, "spec": "300-vertex random digraph, deg<=6",
                 "notes": "Strongly connected via a random directed cycle; in/out degree <= 6."}

    g, label = _random_bounded_degree_graph(363, 404, name="random_R4_363")
    out["R4"] = {"graph": g, "label": label, "spec": "363-vertex random digraph, deg<=6",
                 "notes": "Strongly connected via a random directed cycle; in/out degree <= 6."}

    g, label = _random_bounded_degree_graph(396, 505, name="random_R5_396")
    out["R5"] = {"graph": g, "label": label, "spec": "396-vertex random digraph, deg<=6",
                 "notes": "Strongly connected via a random directed cycle; in/out degree <= 6."}

    return out


def get_builtin(key: str) -> DiGraph:
    """Fetch a built-in graph by key ('0','0.5','1'..'7','G9','R1'..)."""
    G = builtin_graphs()
    if key not in G:
        raise KeyError(f"Unknown built-in graph key: {key!r}. Available: {list(G)}")
    return G[key]["graph"]


# ---------------------------------------------------------------------------
# Custom graph parsing
# ---------------------------------------------------------------------------

def parse_graph_spec(spec: Dict[str, Any]) -> DiGraph:
    """Build a DiGraph from a plain dict spec.

    Expected shape:
        {
          "vertices": ["A0","A1",...],        # list of labels
          "edges": [["A0","A1"], ...],        # list of [u,v] directed edges
          "undirected": True                  # if True, auto-add reverse edges
        }
    """
    V = list(spec["vertices"])
    directed_edges = [(u, v) for u, v in spec["edges"]]
    if spec.get("undirected", False):
        directed_edges += [(v, u) for u, v in directed_edges]
    return DiGraph(V, directed_edges)


# ---------------------------------------------------------------------------
# CLI helper: dump a graph's adjacency to stdout for inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python graphs.py <key|json-spec>  (e.g. 0, 0.5, G9, G7d, R5)")
        sys.exit(1)
    arg = sys.argv[1]
    if arg in builtin_graphs():
        g = get_builtin(arg)
    else:
        g = parse_graph_spec(json.loads(arg))
    print(json.dumps(g.to_dict(), indent=2, ensure_ascii=False))
