# Graphic-KataGo · 五子棋

本文件定义本仓库的 **Gomoku（无禁手 / freestyle \(k\) 连珠）** 与 **Anti-Gomoku（反五子棋）**：对象、合法着、终局，以及它们和图围棋、网络实验的关系。  
图围棋规则只写在 [`rules.md`](rules.md)，不因本节改写。实现见 [`implementation.md`](implementation.md) §12。

所有术语按本节定义；实现不得另发明胜负谓词。

---

## 为何存在（实验，不是第二种围棋）

两条主网对着两类规则，各验一件完备性：

| 规则 | 胜负写在哪 | 要检验的网 | 负对照 |
|---|---|---|---|
| 图围棋 [`rules.md`](rules.md) | 图论：气、块、围歼，只依赖 \((V,E)\) | **GNN**：邻接与规则同构，换图仍该能学 | MLP / 1DCNN（几乎不看真边） |
| 五子棋 / 反五子棋（本节） | 网格几何：坐标四向 \(k\) 连珠，**不读边** | **2DCNN**：3×3 核与平面八邻域同构 | GNN（默认 4-邻接上看不见斜线） |

图围棋问的是：把盘换成乱图/有向边之后，「占子—气—围歼」还剩多少——这是 **GNN 是否完备**。  
五子棋问的是：把胜负绑在网格直线上之后，卷积还能不能看见那条线——这是 **2DCNN 是否完备**。  
反五子棋共用同一套几何谓词，只把「成线者胜」改成「成线者负」，用来看 2DCNN 学的是共线结构，还是「连五=赢」的标签捷径。

因此本节不是职业连珠（无禁手、无交换）。也不是「GNN 在任意任务上都学不会」：层数到 \(O(\sqrt n)\)、再把边界当坐标系，消息传递原则上可以重建格子。能**严格主张**的是：连珠所用的共线 **不是** \((V,E)\) 上的图性质；默认 4-邻接上，一条对角五连在一跳内与 \(k\) 个孤立子无法区分。若同一套自对弈下 2DCNN 能学而 GNN 不能，缺口就是缺网格归纳偏置，不是宽度不够。

---

## 0. 基本对象

- **坐标网格**：整数矩形 \(\{0,\ldots,R-1\}\times\{0,\ldots,C-1\}\)，\(R,C\ge 2\)。点记 \((r,c)\)。
- **顶点集** \(V\)：与格子一一对应，行优先编号
  \[
    v(r,c)=rC+c,\qquad |V|=n=RC.
  \]
  反函数：\(r=\lfloor v/C\rfloor\)，\(c=v\bmod C\)。
- **图** \(G=(V,E)\)：仍是有向图，供 GNN / 力导向 / 与图围棋共用的 `Position` 使用。\(E\) **不进入** §4 的胜负定义。默认内置盘上 \(E\) 是 4-邻接（横竖双向）；消融盘把对角也连上（8-邻接）。无向网格实现为每条无向边对应一对有向边。
- **网格元数据** \(\mathrm{grid}=(R,C,\tau)\)：\(\tau\) 为是否环面（§5）。规则引擎用它把 \(v\) 映回 \((r,c)\)。
- **玩家集** \(P=\{1,2\}\)（黑先）。不推广到 \(k\ge 3\)。
- **占领** \(\mathrm{occ}:V\to P\cup\{\emptyset\}\)。
- **局面** \(s=(\mathrm{occ},\,p_{\mathrm{move}})\)。无虚手条、无超劫历史对合法着的约束。
- **连珠长度** \(k\ge 2\)（默认 \(k=5\)）。

---

## 1. 棋盘与行棋

- 对局在空盘或给定的合法占领上开始；轮到 \(p_{\mathrm{move}}\)。
- 一方着法是选一个 \(\mathrm{occ}(v)=\emptyset\) 的顶点，令 \(\mathrm{occ}(v):=p\)，然后换手。
- **没有虚手。** 动作空间是空点集合，不含「过」。
- **没有提子、自杀、劫。** 落子只改一个顶点的占领，不根据 \(E\) 改其它顶点。
- 棋盘填满后若仍无人按 §4 取胜，则为和棋（§4.2）。

---

## 2. 禁手

仅此一条：目标顶点已被占领，或对局已结束，或不是该方回合。  
不禁止「自己无气」（本规则无气）。不禁止局面重复。

---

## 3. 几何方向

在坐标平面上取四个方向（及其反向，见 §4 的双向计数）：

\[
  D=\{(0,1),\,(1,0),\,(1,1),\,(1,-1)\}.
\]

即横、竖、主对角、副对角。这是 \(\mathbb{Z}^2\) 上的直线，不是「图上长度为 \(k-1\) 的路」。

---

## 4. 终局与胜利

### 4.1 过点射线与连珠长度

固定格子 \((r_0,c_0)\)、方向 \((d_r,d_c)\in D\)、颜色 \(p\)。  
从该点沿 \(+\) 与 \(-\) 各走一步步长 \(t=1,2,\ldots\)，直到出界或遇到非 \(p\) 的占领。两端同色格点数（含起点）之和为该方向上过该点的 **连珠长度** \(\ell\)。

平面（\(\tau=\mathrm{false}\)）上，格 \((r,c)\) 合法当且仅当 \(0\le r<R\) 且 \(0\le c<C\)。

环面（\(\tau=\mathrm{true}\)）上，坐标按模运算：
\[
  r\mapsto r\bmod R,\qquad c\mapsto c\bmod C.
\]
若射线在走了周期 \(t\) 步后回到起点，则整圈同色：\(\ell=t\)，**禁止再把反向再数一遍**（否则长度为 \(2t\)）。

一方在 \(v\) 落子后取胜，当且仅当存在 \((d_r,d_c)\in D\)，使过 \(v\) 的该方向连珠长度 \(\ge k\)。

只需检查刚落下的点：更早的子若已成线，对局应已结束。

### 4.2 结束条件

- **胜**：§4.1 成立，胜者是刚落子的一方（或摆盘时已成线的那一方）。
- **和**：\(V\) 上已无空点，且无人满足 §4.1。
- 否则对局进行中。

不计领地、不贴目、不焦点扩张。终局分数是 \(\{0,1/2,1\}\)：胜者 1、负者 0、和棋各 \(1/2\)。

### 4.3 反五子棋（Anti-Gomoku）

合法着、棋盘、\(k\)、无虚手、无提子均与上文相同。唯一差别：§4.1 成立时，**刚成线的一方负**，对方胜。满盘无人成线仍为和棋。引擎字符串 `rules=antigomoku`（也接受 `anti-gomoku` / `anti_gomoku`）。轮训图与五子棋相同（`GOMOKU_TRAIN_KEYS`）。

---

## 5. 环面

仅当 \(\mathrm{grid}\) 标明环面时，连珠可绕过左右/上下边。这仍是坐标上的周期，不是「沿着 \(E\) 绕一圈」。  
若 \(E\) 本身不是环面 4-邻接，GNN 看见的图与规则看见的周期可以不一致——那是实验可以故意制造的不对称，不是规则漏洞。

---

## 6. 规则引擎如何在顶点图与邻接上运行

引擎与图围棋共用 `Position`：`occupancy[v]`、`to_move`、共享的 `Graph`。分叉从谓词开始。

### 6.1 图对象（两边都有）

`Graph` 存 CSR：入边 / 出边 / 无向化边，以及稠密 0/1 矩阵 `adj_in`、`adj_out`（\(n\times n\)）。GNN 的一层是

\[
  h \leftarrow h + f\bigl(h,\; A_{\mathrm{in}}h,\; A_{\mathrm{out}}h,\; \log(1+d_{\mathrm{in}}),\; \log(1+d_{\mathrm{out}})\bigr).
\]

顶点特征是相对视角的占子独热（通道 0 = 走棋方）。**五子棋不把「是否共线」写进这些矩阵。**

### 6.2 合法着（不读边）

空点列表。不调用提子核心、不查超劫、不把动作 \(n\)（图围棋的虚手）算进去。`apply_pass` 对五子棋是恒等；`Game::play(n)` 与 `pass_move()` 均非法。

### 6.3 胜负（只读坐标，不读 CSR）

`grid` 必须满足 \(R\cdot C=n\)。对刚落下的 \(v\)，用 \(r,c\) 与 §3 的四个 \((d_r,d_c)\) 扫描占领数组。邻接矩阵、入邻居、块、气均不参与。

因此可以出现：**对角五连在 4-邻接图上同色度数为 0**，而规则已经判胜。这是实验要的缝，不是漏实现。

### 6.4 搜索与训练标签

MCTS / 自对弈与图围棋同一套调度。终局价值是 \(\{+1,0,-1\}\)（相对走棋方），不是领地分。手数上限固定为 \(n\)（填满即止），不把中盘截断标成和棋。无超劫，局面不深拷历史集合。

网络结构不改：GNN 仍吃该图的 \(A\)；2DCNN 把长度为 \(n\) 的顶点特征按 \((R,C)\) reshape 成平面。故 **同一套权、两种归纳偏置，对着同一套几何规则。**

### 6.5 内置盘（图边 ≠ 连珠方向）

| key | 格子 | \(E\) | 用途 |
|---|---|---|---|
| 0.5 / 0 / 2 / G9 / G15 | \(7\times7\) / \(19\times19\) / \(61\times61\) / \(9\times9\) / \(15\times15\) | 4-邻接 | 主实验：斜线不是边。网页把 `0.5` 显示成 G7、`0` 显示成 G19、`2` 显示成 G61 |
| 1 / 3 | \(27\times13\) / \(19\times19\) 环面 | 该图原有的边 | 默认轮训里的异形 / 环面盘 |
| G7d / G9d | \(7\times7\) / \(9\times9\) | 8-邻接 | 消融：连通仍不等于共线（蛇形五子也连通） |

键 `7` 是 19 点 **路**，不是 7×7。图围棋默认训练排除 `G*`。五子棋网页不下非矩形网格图。

---

## 7. 与图围棋对照

| 项 | 图围棋 [`rules.md`](rules.md) | 本节 |
|---|---|---|
| 盘 | 任意有向图 | 必须有矩形坐标嵌入 |
| 胜负谓词 | 气 / 块 / 领地（依赖 \(E\)） | \(k\) 共线（依赖 \((r,c)\)，不依赖 \(E\)） |
| 虚手 / 劫 / 提子 | 有 | 无 |
| GNN 输入 | \(\mathrm{occ}\) + \(A\) | 同；但 \(A\) 与胜负不对齐 |
| 2DCNN | 网格对照；图围棋下的形状对照 | **与规则同构的完备性检验** |
| 训练价值 | 领地领先 | 胜/负/和 |

---

## 8. 训练入口

自对弈学下棋（与图围棋同一套 trainer，`--rules gomoku` 或 `antigomoku`）。手数上限固定为 \(n\)，不用图围棋的 `--min-moves` / `--max-move-factor` / `--curriculum-rounds`。

默认轮训图（内部键；UI 别名）：`0`（G19）、`0.5`（G7）、`1`、`3`、`G9`、`G15`、`G7d`、`G9d`。不含 `2`（G61）。每轮打乱图顺序。SGD 与 Arena/UI 搜索按网络分派顶点重编号——GNN / 1DCNN / MLP 随机 \(S_n\)（GNN 同时置换邻接），2DCNN 走 grid2d 棋盘对称 D4/Klein/torus。MCTS 打乱合法着顺序。

四个入口与目录（图围棋包装 `train_gnn.bat` 等、五子棋包装 `train_gomoku_gnn.bat` 等、反五子棋包装 `train_antigomoku_gnn.bat` 等都调用同一个 `start_train.bat`；五子棋传入 `--rules gomoku`，反五子棋传入 `--rules antigomoku`。未指定 `--outdir` 时 trainer 默认写这些目录）。产物格式与图围棋 `cur_mod_*` 相同：`gkt_train.log`（CPU 为 `gkt_train_cpu.log`）、`summary.json`、`progress.txt`、`progress.wN.txt`、`progress/summary_{key}.json`、`replay/{key}/unused.npz`、`new.pt` / `new.npz`、`roundN.*`。

```text
train_gomoku_gnn.bat / train_gomoku_mlp.bat / train_gomoku_cnn1d.bat / train_gomoku_cnn2d.bat
start_train.bat gnn|mlp|cnn1d|cnn2d --rules gomoku
cur_mod_gomoku_gnn  cur_mod_gomoku_mlp  cur_mod_gomoku_cnn1d  cur_mod_gomoku_cnn2d

train_antigomoku_gnn.bat / train_antigomoku_mlp.bat / train_antigomoku_cnn1d.bat / train_antigomoku_cnn2d.bat
start_train.bat gnn|mlp|cnn1d|cnn2d --rules antigomoku
cur_mod_antigomoku_gnn  cur_mod_antigomoku_mlp  cur_mod_antigomoku_cnn1d  cur_mod_antigomoku_cnn2d
```

```bash
python cpp/build.py
cd scr
python gkt_train_gpu.py --rules gomoku --net gnn --sim 800 --workers 1 --gpw 16 --infinite
python gkt_train_gpu.py --rules antigomoku --net gnn --sim 800 --workers 1 --gpw 16 --infinite
python gkt_train_gpu.py --rules gomoku --net 2dcnn --sim 800 --workers 1 --gpw 16
python gkt_train_cpu.py --rules gomoku --net mlp --sim 800 --workers 1 --gpw 16
python gkt_train_cpu.py --rules gomoku --net 1dcnn --sim 800 --workers 1 --gpw 16
```

Web UI：规则选「五子棋」或「反五子棋」时列出同一套矩形网格；`0` 显示为 G19，`0.5` 显示为 G7，`2` 显示为 G61。配色：图围棋偏红，五子棋偏绿，反五子棋偏蓝紫。
