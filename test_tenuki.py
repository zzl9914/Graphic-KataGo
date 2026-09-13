r"""Tenuki / position-judgment construction tests for a distilled base net.

Human-analysis tool: build a handful of tiny, hand-checkable board positions
and read off WHERE the net wants to play (policy), WHAT it thinks of the whole
board (value), and WHERE it thinks territory lies (ownership). This turns the
abstract "the model can't tenuki / can't judge the position" into concrete
numbers you can watch improve as distillation / finetune progresses.

Positions (all 19x19, key "0"; vertex i = y*19 + x, y=0 = top row):

    s0  empty board, Black to move        -> should play a corner (3-3 / 4-4)
    s1  Black 3-3 at TL, 3 empty corners  -> should tenuki to a fresh corner
    s2  B 3-3 TL + W 3-3 BR               -> should tenuki to a fresh corner
    s3  White (2,2) in atari; capture=1 stone vs empty corner
                                          -> should tenuki, not grab the stone

Run:  D:\Python\pythoncore-3.14-64\python.exe scr\test_tenuki.py [--model ..\base\gnn\new.pt]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graphs import get_builtin  # noqa: E402
import gkt_cpp  # noqa: E402
from gkt_gpu import load_net  # noqa: E402

N = 19


def idx(x: int, y: int) -> int:
    return y * N + x


def coord(i: int):
    return i % N, i // N


# Corner "boxes" (3-3 .. 4-4 neighbourhood) used to classify a move.
CORNER_BOXES = {
    "TL": [(x, y) for y in range(1, 5) for x in range(1, 5)],
    "TR": [(x, y) for y in range(1, 5) for x in range(14, 18)],
    "BL": [(x, y) for y in range(14, 18) for x in range(1, 5)],
    "BR": [(x, y) for y in range(14, 18) for x in range(14, 18)],
}
BOX_OF = {}
for name, pts in CORNER_BOXES.items():
    for (x, y) in pts:
        BOX_OF[idx(x, y)] = name


def box_label(i: int) -> str:
    return BOX_OF.get(i, "其他")


def stones_of(box: str) -> set:
    return {idx(x, y) for (x, y) in CORNER_BOXES[box]}


SCENARIOS = [
    {
        "name": "s0 空盘（黑先）",
        "to_move": 1,
        "stones": [],
        "expect": "占角：top1 应落在某个空角（3-3/4-4）",
    },
    {
        "name": "s1 黑占左上三三，三空角待占",
        "to_move": 1,
        "stones": [(1, 2, 2)],
        "expect": "脱先：top1 应落在 TR/BL/BR 空角，而非贴着左上三三",
    },
    {
        "name": "s2 黑白各占对角（三三），两空角",
        "to_move": 1,
        "stones": [(1, 2, 2), (2, 16, 16)],
        "expect": "脱先：top1 应落在 TR/BL 空角，而非纠缠已占角",
    },
    {
        "name": "s3 白(2,2)被打吃，可提一子 vs 空角",
        "to_move": 1,
        "stones": [(1, 1, 2), (1, 3, 2), (1, 2, 1), (2, 2, 2)],
        "expect": "脱先：提一子(2,3)远小于空角，top1 应占空角",
    },
]


def build_board(stones):
    b = np.zeros(N * N, dtype=np.int8)
    for color, x, y in stones:
        b[idx(x, y)] = color
    return b


def render(board, to_move, marks):
    sym = {0: ".", 1: "B", 2: "W"}
    out = ["     " + "".join(str(x % 10) for x in range(N))]
    for y in range(N):
        row = [f"{y:2d}  "]
        for x in range(N):
            i = idx(x, y)
            row.append(marks.get(i, sym[int(board[i])]))
        out.append("".join(row))
    return "\n".join(out)


def run_scenario(net, ng, native, sc):
    board = build_board(sc["stones"])
    to_move = sc["to_move"]
    game = native.Game(ng, 2, [], to_move, board, "go", 5)
    me = int(game.position.to_move)
    X = np.asarray(native.extract_features(game.position, me), dtype=np.float32)

    legal = list(game.legal_moves())
    mask = torch.zeros(N * N + 1, dtype=torch.bool, device=net.device)
    for a in legal:
        if 0 <= int(a) <= N * N:
            mask[int(a)] = True

    x = torch.from_numpy(X).float().unsqueeze(0).to(net.device)
    net.eval()
    with torch.no_grad():
        out = net.forward(x)
    logits = out["policy"][0].clone()
    logits = logits.masked_fill(~mask, -1e9)
    policy = torch.softmax(logits, dim=-1).cpu().numpy()
    value = float(out["value"].item())
    own = out["own"][0].cpu().numpy()  # mover-relative, tanh [-1,1]

    order = np.argsort(policy)[::-1]
    top = [(int(a), float(policy[a])) for a in order[:10] if a < N * N]

    # Tenuki metric: probability mass in the corners the net is NOT in.
    occupied = set()
    for _, x, y in sc["stones"]:
        b = box_label(idx(x, y))
        if b != "其他":
            occupied.add(b)
    free_corners = [b for b in CORNER_BOXES if b not in occupied]
    free_mass = sum(policy[i] for b in free_corners for i in stones_of(b))
    top_box = box_label(top[0][0]) if top else "?"

    print("=" * 64)
    print(sc["name"])
    print("  期望：", sc["expect"])
    print(f"  value(行棋方视角) = {value:+.4f}")
    print(f"  ownership 均值 = {float(own.mean()):+.4f}  标准差 = {float(own.std()):+.4f}"
          "  （std≈0 说明 own 头塌缩=不会评估局势）")
    for b in ("TL", "TR", "BL", "BR"):
        mass = sum(policy[i] for i in stones_of(b))
        print(f"  角 {b}: policy质量 {mass:.3f}")
    print(f"  空角 policy 总质量 = {free_mass:.3f}")
    print(f"  top1 = ({coord(top[0][0])[0]},{coord(top[0][0])[1]}) "
          f"[{box_label(top[0][0])}] p={top[0][1]:.3f}")
    verdict = "脱先✓" if top_box in free_corners else (
        "继续局部缠斗✗" if top_box != "其他" else "非角（需人工判断）")
    print(f"  判定：{verdict}")
    marks = {}
    for k, (a, _p) in enumerate(top[:3]):
        marks[a] = str(k + 1)
    print(render(board, to_move, marks))
    print("  标记：1/2/3 = top1/top2/top3，B=黑 W=白 .=空")
    print()
    return {"name": sc["name"], "top_box": top_box, "free_mass": free_mass,
            "value": value, "own_std": float(own.std()), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="Tenuki construction tests")
    ap.add_argument("--model", default="../base/gnn/new.pt")
    ap.add_argument("--key", default="0")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    native = gkt_cpp.require_native()
    graph = get_builtin(args.key)
    ng = gkt_cpp.py_graph_to_native(graph)
    net = load_net(os.path.abspath(args.model), device=device, graph=graph)
    print(f"模型：{args.model}  arch={net.net_type}  device={device}  "
          f"H={net.hidden_dim} blocks={net.n_blocks}")
    print()

    rows = [run_scenario(net, ng, native, sc) for sc in SCENARIOS]
    print("=" * 64)
    print("汇总")
    for r in rows:
        print(f"  {r['name']:<36} top1={r['top_box']:<4} 空角质量={r['free_mass']:.3f} "
              f"value={r['value']:+.4f} own_std={r['own_std']:+.4f}  {r['verdict']}")


if __name__ == "__main__":
    main()
