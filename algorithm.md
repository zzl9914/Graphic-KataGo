# Graphic-KataGo · 算法

本文件讲 **模型架构、特征、MCTS、多头、搜索侧并行、UI 与内置图**。  
蒸馏、分阶段自对弈、**分布式启动与安全**写在 [`training_method.md`](training_method.md)（§8）。本文件不写操作步骤。  
图围棋规则只写在 [`rules.md`](rules.md)；五子棋 / 反五子棋写在 [`gomoku.md`](gomoku.md)。图围棋检验 **GNN 完备性**；五子棋 / 反五子棋检验 **2DCNN 完备性**。

**架构**：图 → C++ 规则引擎 / MCTS / 自对弈 → CPU 或 GPU 网络 → Python 跨图训练循环 → Web 对弈。  
**主体名 GKT Go**：搜索/自对弈热路径在 `cpp/`（`gkt_native`），Python `gkt.py`（`GktSelfPlay`）只做调度；GPU 并行训练器 `GktTrainer`。文件与类名里的 `gkt` 即此缩写。  
**规范树**：文档 `ref/`，Python `scr/`，C++ 热路径 `cpp/`。

---

## 1. 代码与命名约定

全树统一命名。

| 类别 | 约定 | 例子 |
|---|---|---|
| 模块文件 | `snake_case.py` | `gkt.py`, `gkt_gpu.py`, `graphs.py` |
| 类 | `PascalCase` | `DiGraph`, `GktSelfPlay`, `GktTrainer` |
| 函数 / 方法 / 变量 | `snake_case` | `play_one_game`, `n_simulations` |
| 模块内私有 | 单下划线前缀 | `_selfplay_worker` |
| 常量 | `UPPER_SNAKE` | `BLACK`, `MASK_VALUE`, `SCORE_BINS` |
| CLI | kebab-case | `--selfplay-batch`, `--q-lambda` |
| 文档互引 | ``ref/*.md``、``scr/*.py`` | 例如 ``ref/rules.md`` |

**换行符**

一律按 **LF** 写（编辑器 / 工具默认）。Windows `cmd.exe` 跑 `*.bat` 需要 **CRLF**。不要在正文里手改 `\r`；写完 `.bat` 后调用仓库根目录的 `eol.py`：

```
python eol.py              # 默认：仓库内全部 *.bat  LF→CRLF
python eol.py a.bat b.bat  # 只转列出的文件（或目录下的 *.bat）
python eol.py --to lf      # 相反：CRLF→LF（同一批文件）
```

只动换行，不动编码（UTF-8 与系统 ANSI 混用时 `echo` 会乱码；已经在跑的 cultivate2 启动器不要再动编码）。源码 / 文档保持 LF，不必过 `eol.py`。

**网络鸭子接口**（四种网络共用）：

- `predict(X, legal_mask) -> (policy, value)`
- `predict_batch(X_batch, masks) -> (policy, value)`；GPU 另回 `stdev`（第三项）。缺省时 C++ 按 \(\mathrm{softplus}(0)=\ln 2\) 填
- 训练：CPU 用 `backward(...)` 逐样本；GPU 用 `train_on_batch(...)` 批量
- 可选：`set_graph` / `get_weights` / `set_weights` / `state_dict`

| 设备 | 文件 | 网络 |
|---|---|---|
| CPU（NumPy） | `gkt_cpu.py` | **MLP**、**1DCNN** |
| GPU（PyTorch） | `gkt_gpu.py` | **GNN**、**2DCNN**（仅网格） |

CLI：`--net mlp|1dcnn`（`gkt_train_cpu.py`），`--net gnn|2dcnn`（`gkt_train_gpu.py`）。

MCTS 用 policy + **value_rto**（lead/\(n\)）+ stdev（加权备份、方差缩放 cPUCT）。监督用的 **value_abs**（子数差）不进搜索。其余辅助头也不进搜索。随机图上的讲解 / drone 规划见 §10.1。

**搜索次数**：构造参数名 `n_simulations`，实例字段 `n_sim`，命令行 `--sim`。自对弈逐步搜索在 \([\mathrm{sim}/4,\,4\cdot\mathrm{sim}]\) 对数均匀抽样。Arena / Elo / UI 用固定次数（UI 模型席 = 该席 `sim`，不少于 1）。训练自对弈根 Dirichlet \(f=0.25\)（C++ `MCTSConfig`）；Arena / Elo / UI 为 \(f=0\)（`gkt.py` 的 `DIRICHLET_FRAC`，纯访问 argmax）。

训练自对弈与 Arena **不使用贴目**（`Game` 默认补偿为 0）。贴目只出现在规则对局与 Web UI。[`rules.md`](rules.md) 里二人 7.5（以及三/四人递减序列）是 **UI 展示默认**，不是 C++ `Game()` 构造默认。

---

## 2. 技术栈

| 层 | 选型 | 用途 |
|---|---|---|
| 热路径 | C++17（`cpp/`，`gkt_native` via pybind11） | 规则、合法着、MCTS、自对弈样本 |
| 训练调度 | Python 3 | 跨图循环、Adam、checkpoint、HTTP |
| 数组 | NumPy | 特征进出 Python、CPU 网络 |
| 深度学习 | PyTorch | GNN / 2DCNN、Adam、GPU 训练；MCTS 叶子用 `torch.jit.trace`（`JitInfer` / `maybe_script_infer`）；`export_script` 可另存给 LibTorch |
| 推理（可选） | C++ LibTorch | 仅当 `gkt_native` 以 `GKT_WITH_TORCH` 编译时走 `ScriptedNet`；默认仍是 Python 回调 `predict_batch` |
| 并行 | `ProcessPoolExecutor` + `spawn` | 自对弈 worker；Windows 与 Linux 一致 |
| 图数据 | Python `DiGraph` 构图；C++ CSR + 稠密 0/1 邻接 | GNN 求和消息传递；邻接是数据不是权重 |
| Web | `http.server` + 单文件 `index.html` | 无 Flask；对弈走 C++ `Game` / `search` |

**刻意不用的**：稀疏图卷积库、分布式参数服务器、异步 GPU 评估器、WebSocket、数据库。单卡笔记本上这些没有收益（§9）。

训练与 UI **必须**加载 `gkt_native`（失败即报错，不回退慢路径）。Windows 入口 `start_train.bat` / `start_ui.bat` / `start_dist_main.bat` / `start_dist_cont.bat` 钉死 **Python 3.14**。不要用 `py -3`（可能开到 free-threading 的 `python3.14t.exe`），也不要用 3.13：扩展是 `gkt_native.cp314-win_amd64.pyd`。编译见 [`training_method.md`](training_method.md) §1。分布式启动步骤只写 [`training_method.md`](training_method.md) §8。

---

## 3. 模块与数据流

```
内置图 / JSON 规格     graphs.py
        ↓
   C++ Graph / Game / 特征 / MCTS / 自对弈     cpp/ (gkt_native)
        ↓
 GktSelfPlay 调度           gkt.py
        ↓                 ↓
 gkt_cpu.py           gkt_gpu.py
 (MLP / 1DCNN)        (GNN / 2DCNN + GktTrainer + TorchScript)
        ↓                 ↓
 gkt_train_cpu.py     gkt_train_gpu.py     跨图同一套权重
        ↓
 gkt_elo.py（可选，只展示）
 web_ui/server.py + index.html     对弈 / 观战 / 回放
```

| 文件 | 职责 | 不负责 |
|---|---|---|
| `graphs.py` | 有向图、内置预设、`grid`/`sym` 元数据 | 规则、网络 |
| `cpp/` | 合法着、提子、SSK、计分、MCTS、自对弈 | 权重训练 |
| `gkt.py` | 训练辅助、自对弈调度、Arena 评估 | 权重张量 |
| `gkt_cpp.py` | 加载 `gkt_native`、Python 图 → C++ | |
| `grid_sym.py` | SGD + 训练自对弈 + Arena/UI 搜索：顶点置换按网络分派——GNN/1DCNN/MLP 走 \(S_n\)，2DCNN 走 grid2d 棋盘对称 D4/Klein/torus（GNN 另换邻接） | 规则 |
| `gkt_cpu.py` | CPU：MLP、1DCNN | GPU 网络 |
| `gkt_gpu.py` | GPU：GNN、2DCNN + `GktTrainer` + TorchScript | 跨图调度 |
| `gkt_train_*.py` | 跨图轮换、课程、Arena、checkpoint | 网络定义、Elo |
| `distill.py` | KataGo 强标签监督预训练（policy/value/own），产出蒸馏基模型 | 自对弈、辅助头 |
| `distill_katago.py` | KataGo analysis JSON → 蒸馏 JSONL（唯一知道 KataGo 坐标的地方） | gkt 顶点顺序 |
| `gkt_elo.py` | 全算法共享 Elo 榜（仅展示） | 训练、门控 |
| `gkt_dist.py` | 异步 selfplay/shuffle/train/gate/contribute（启动与安全见 [`training_method.md`](training_method.md) §8） | 参数服务器、本机 `gkt_train_*` 调度 |
| `web_ui/` | HTTP + Canvas | 训练 |

分阶段入口与旗标见 [`training_method.md`](training_method.md)。

---

## 4. 规则引擎要点

- `Position` 冻结；落子返回新对象。实现在 C++（`cpp/src/engine.cpp`）。默认 `rules=go`（[`rules.md`](rules.md)）；`rules=gomoku` / `antigomoku` 见 [`gomoku.md`](gomoku.md)。
- 图围棋：气 / 块在 **CSR**（入邻接 + 无向并）上计算。超劫用 Zobrist 增量 XOR。虚手只换回合盐、不查重复禁手。种子 `0x5EEDC0DE` 与 Python `random.Random` 对齐。二人单遍合法着（`legal_moves_fast` 与 `try_move` 同一套：提完他方后己方仍须有气）；多人逐点 try-move。`legal_moves_for` 含虚手（下标 \(n\)）；`Game.play(n)` 即虚手。`try_move` 校验 `player == to_move`。终局后 `play(n)` 与 `pass_move()` 均非法；底层 `apply_pass` 仍恒等。未知 `rules` 字符串拒收。Python `predict_batch` 失败或 policy 长度不是 \(B\times(n+1)\) 时抛错。畸形 `occupancy`（长度 ≠ \(n\)）读取时抛错。
- 五子棋 / 反五子棋：合法着仅为空点；不读 CSR 判胜，只读 `.grid` 坐标四向连珠。必须 \(R,C\ge 2\) 且 \(R\cdot C=n\)。`play(n)` 与 `pass_move()` 均非法。反五子棋成线方负。
- 终局列表为空。图围棋 `finalize()` 返回无贴目领地分；五子棋 / 反五子棋为 \(\{0,1/2,1\}\)。Web UI 对图围棋再加选手贴目。训练自对弈、Arena、Elo **不加贴目**。[`rules.md`](rules.md) §5.4 的 7.5 是 UI 默认，与 `Game()` 全 0 构造无关。

---

## 5. 特征

每顶点 \(F=\texttt{num_players}+6\)：**视角相对**占领独热（通道 0 = 当前走棋方），再拼 1 气 / 2 气 / 3+ 气、\(\log(1+\text{块大小})\)、上一手 one-hot、刚被提的空点。编码前再派生一维空点 \(1-\sum\mathrm{onehot}\)。气按规则是块的空入邻居。权不依赖 \(n\)。\(F\) 必须等于 `feature_dim(num_players)`（2P 即 8）。邻接只进 GNN。

---

## 6. MCTS

- 懒展开：子节点先不物化 `Position`。
- 批量叶子 + **虚拟损失**：一批内后续选择偏离已走分支。
- 根 Dirichlet：训练自对弈 \(\alpha=0.3\)、\(f=0.25\)（C++ 默认）；Arena / Elo / UI \(f=0\)（`DIRICHLET_FRAC`）。自对弈走子按温度采样（围棋主线 \(\tau=0.1\)）；Arena / UI / Elo 仍 `argmax` 访问，sim 固定。
- **顶点重编号（防「编号迷信」）**：训练自对弈的**每次**叶子评估（`predict_batch`）都抽新随机置换——GNN / 1DCNN / MLP 走 \(S_n\)（GNN 同步置换 `adj`），2DCNN 走 grid2d 棋盘对称（D4 / Klein / torus shift，非 grid 图 fallback \(S_n\)），策略 gather 回原编号。Arena / UI / Elo 每次 `search` 抽一次置换、搜索内叶子共用（`SearchAugNet` 非 fresh 模式）。两者都经 `native.search`。分派细节见 §8.5。
- 合法着每节点 `shuffle`，打破「平局总走下标 0」——这是搜索层公平，独立于模型层重编号。
- 叶子对每个座位各做一次视角特征，得到 \(k\) 维 **rto** lead；备份时路径上每节点取**该节点执子方**的分量，**不是**二人翻号。搜索价值公式见 [`training_method.md`](training_method.md) 的 `score_lead`（二人即 \((\text{我}-\text{对方})/n\)）。
- **FPU**：未访问子的 Q 为 \(\mathrm{parent}Q - 0.2\sqrt{\sum \pi_{\mathrm{nn}}(\text{已访问子})}\)，不是 0。
- **stdev**：叶子权重 \(w=1/(1+\sigma^2)\)；节点 Q 为 \(\sum wv/\sum w\)（不是 \(\sum wv/N\)）；cPUCT 乘 \(\sqrt{\mathrm{Var}(Q)+\varepsilon}/\sqrt{\varepsilon}\)。
- 训练目标：访问分布先丢掉「次数低且先验也低」的噪声着，再归一；样本权重来自 surprise。

`randomized_sim_count(nominal)`：\(\exp\mathcal{U}(\log(\mathrm{sim}/4),\log(4\cdot\mathrm{sim}))\) 再四舍五入。自对弈每步调用；Arena / Elo / UI 不随机。自对弈手数上限用同一变换、区间为 \([\mathrm{cap}/2,\,2\cdot\mathrm{cap}]\)，每局抽一次。

---

## 7. 多头

主干宽度与搜索算法固定。训练同时拟合这些头；**进 MCTS 的只有 policy / value_rto / stdev**。

| 头 | 目标 | 损失（权重） |
|---|---|---|
| policy | 剪枝后的访问分布，长度 \(n+1\)，下标 \(n\) 为虚手维（五子棋该维 mask 为 0） | 按 surprise 加权的 CE |
| value_rto（搜索） | mix：\(\lambda\cdot\mathrm{lead}+(1-\lambda)\cdot Q\)，lead 是 \(z/n\) | MSE × `--value-rto-weight`（默认 5；tanh 前向；反向雅可比 \(\max(1-\tanh^2, 0.2)\)）。JIT / MCTS / Arena / Elo 读这个头（`out["value"]`） |
| value_abs（监督） | 图围棋：终局子数差 \(z\)（二人即 \(\mathrm{我}-\mathrm{对方}\)，**不**除 \(n\)，**不** mix Q）。五子棋 / 反五子棋：与 rto 同一 mix（已在 \([-1,1]\)） | MSE × `--value-weight`（默认 0.08）。逐点 Linear 再对顶点求和，**无 tanh**。不进搜索 |
| value_cons（软耦合） | 图围棋 \((\mathrm{abs}-n\cdot\mathrm{rto})^2\)；五子棋 / 反五子棋 \((\mathrm{abs}-\mathrm{rto})^2\) | MSE × `--value-cons-weight`（默认 0.01）。不是硬约束 |
| ownership | 终局每点归属 | MSE × `--own-weight`（默认 5；tanh 前向，反向同 rto） |
| opp policy | 下一手（下一家）的访问分布 | CE × 0.25，末手权重 0 |
| soft policy | \(\pi^{1/T}\)，\(T=4\)，合法着上再归一 | CE × 0.15 |
| score belief | lead 在 \([-1,1]\) 上 21 bin 的软直方图 | pdf CE × 0.30 + cdf MSE × 0.30 |
| stdev | \(\lvert\mathrm{lead}-Q\rvert\) | MSE × 0.15（softplus 前向，反向 \(\max(\sigma, 0.2)\)） |
| futurepos | 约 4 个决策后的占子 | MSE × 0.25（tanh 前向，反向同 rto） |

自对弈 **rto** 一律是 mix：\(\lambda\cdot\mathrm{lead}+(1-\lambda)\cdot Q\)（`--q-lambda` 是 **MC \(z/n\) 的权重**，默认 0.5）。**abs** 在图围棋上是纯 MC 子数差。两个头用很低的 \((\mathrm{abs}-n\cdot\mathrm{rto})^2\)（`--value-cons-weight` 默认 0.01；五子棋 / 反五子棋是 \((\mathrm{abs}-\mathrm{rto})^2\)）做软耦合，abs 仍可以跟 mix 不完全一致。belief 始终用终局 `lead`（\(z/n\)）。rto / ownership / futurepos / stdev 的末层权为零；abs 头（`value_abs_head` / `W_vabs`）同样零初始化（蒸馏除外）。日志 vloss 是 **abs** 输出空间的未加权 MSE；cons 进 aloss。权重见 [`training_method.md`](training_method.md) §7。

`load_net` / CPU `_load_arrays_strict` 都是严格加载：必须有 `value_abs_head`（GPU）或 `W_vabs` / `b_vabs`（CPU），以及 `net_type`（GPU `.pt`）或 `_net_type`（CPU `.npz`），人数 `num_players` / `_num_players`，且 \(F=\texttt{feature_dim}(k)\)。缺键或 \(F\) 不符则拒载。

---

## 8. 网络

四种网络同一套 `predict` / `predict_batch` 接口；MCTS 不区分实现。角色不同：**GNN** 是图围棋主网（检验图规则下 GNN 完备性）；**2DCNN** 是网格主网（图围棋里当形状对照，五子棋 / 反五子棋里检验 2DCNN 完备性）；**MLP** 与 **1DCNN** 是消融，用来证明 GNN 不是靠宽度瞎拟合。

### 8.1 GPU · GNN（图拓扑主路径）

线性编码（占领 + 空点 + 气/块/上一手/提子）→ LayerNorm → `n_blocks` 层入/出求和聚合 + \(\log(1+d)\) MLP 残差（残差流 LayerNorm）+ global pooling bias（mean / max / \(\mathrm{mean}\sqrt{n}\) 且 \(\sqrt{n}\) 上限 8，投影零初始化）→ 在 `attn_layer`（默认 8）前插入一层全局多头自注意力（同样残差流 LN）→ 逐点 policy（虚手与落子共用同一 `Linear`，作用在 \(\mathrm{mean}(h)\) 上）+ attention-pool **value_rto** + 逐点 Linear 求和 **value_abs** + 逐点 ownership，以及 opp / soft / future / belief / stdev 辅助头（opp / soft 同样共用顶点头）。`state_dict` 含 `pass_head` / `opp_pass` / `soft_pass`，前向不读。policy / opp / soft / belief logit 封顶 \(\pm 20\)，stdev 上限 8。非有限 loss 或梯度则跳过该次 `step`。  
**权重只依赖 \(F\) 与 \(H\)**，不依赖 \(n\)。`set_graph` 注入 0/1 邻接。

入/出分开：规则里气是入邻居。全局注意力打破「感受野 = 层数」，表达势。6GB 卡上注意力 \(n\times n\) 把叶子 batch 卡在约 64。

消息传递用稠密 `adj @ h` 求和：\(n\le 400\) 时 cuBLAS GEMM 快于 scatter；稀疏化留给图 2 / 6 量级。

### 8.2 GPU · 2DCNN（网格拓扑对照）

只在带 `.grid` 的矩形上跑（图围棋默认训练：0 / 0.5 / 1 / 3；五子棋 / 反五子棋默认另加 G9 / G15 / G7d / G9d；都不含 oversized 的 2）。图 2（61×61）不进默认训练，UI 可下，作网格泛化检查。1×1 stem + BatchNorm + 3×3 残差（默认 \(H=512\)、20 块，每块末尾 spatial gpool，\(\sqrt{n}\) 上限 8），环面 circular pad。policy / opp / soft 的虚手与格点共用同一 1×1 conv（空间均值）。policy / opp / soft / belief logit 同样封顶。**不加** GNN 那层全局多头自注意力（value_rto 仍用逐点 softmax 池化；value_abs 仍是逐点 Linear 求和）。`--net 2dcnn`。`--attn-layer` 只作用于 GNN；2DCNN 对象上保存 `attn_layer` / `n_heads`，前向不用。`state_dict` 同样含 `pass_head` / `opp_pass` / `soft_pass`，前向不读。

图围棋下和 GNN 比的是：同一套规则与辅助头下，网格平移对称还剩多少。五子棋下它才是与规则同构的完备性检验（见 [`gomoku.md`](gomoku.md)）。2DCNN 不能吃无 `.grid` 的图。

### 8.3 CPU · MLP（消融）

两层全连接（默认 \(H=512\)，输入为占领+空点+图特征）+ 与 GPU 相同的 attention-pool value_rto / 逐点求和 value_abs / 逐点 ownership。**不加** trunk gpool（消融）。顶点互不见边。用来看「没有拓扑时还能不能装成像那么回事」。`--net mlp`。

### 8.4 CPU · 1DCNN（消融）

按顶点**下标**做一维卷积（默认 \(H=512\)、20 层、核宽 3；编码同样是占领+空点+图特征），邻接是编号顺序，不是图的边。**不加** trunk gpool。value 头与 MLP / GPU 对齐。用来量「假几何」值多少；GNN 若只是略强于 1DCNN，就还没证明学到了真拓扑。`--net 1dcnn`。权重约 60MB，小图（线图 / 7×7）CPU 可跑；19×19 上 `predict_batch` 仍是逐样本循环，MCTS 会很慢。

### 8.5 顶点置换与对称打乱（编号迷信防御）

无论训练自对弈还是 Arena / UI，模型每次读样本都可能面对随机编号，让网络无法把顶点编号当稳定信号。

- **训练自对弈**：`GktSelfPlay` 用 `SearchAugNet(net, graph, fresh_each_batch=True)` 包装，**每次叶子评估**（`predict_batch`）都抽新随机置换，策略 gather 回原编号。
- **Arena / UI / Elo**：`SearchAugNet` 非 fresh 模式，每次 `search` 抽一次置换、搜索内叶子共用。
- **按网络分派**：GNN / 1DCNN / MLP（无棋盘几何）走随机 \(S_n\) 全置换——GNN 另用同一 gather 同步置换 `adj_in` / `adj_out`（拓扑不变），1DCNN / MLP 只换点特征（其卷积核序 / 层序是「假几何」，任意重编号正是消融本身）；2DCNN（有棋盘几何）走 grid2d 棋盘自同构 D4（方阵）/ Klein 四元群（非方阵）/ 环面平移，非 grid 图 fallback \(S_n\)（任意 \(S_n\) 会把不相邻顶点拉进卷积邻域、破坏网格先验）。
- **SGD 增强同样分派**：`_gpu_train_on_batch` 里 GNN / 1DCNN / MLP 走 `augment_vertex_batch`（\(S_n\)，GNN 另换邻接），2DCNN 走 `augment_graph_batch`（D4，无需换邻接）。蒸馏训练集用同一套，验证集不打乱，见 [`training_method.md`](training_method.md) §3。
- **正交的搜索层公平**：合法着每节点 `shuffle`（`mcts.cpp`）独立于上面的模型层重编号，二者不互相替代。

### 8.6 随机性：故意不可复现 vs 图序必须可复现

**训练应当与某一次运行的随机种子无关。** 换 PID、换 minibatch 抽样、换顶点置换，网络仍应学会同一套图无关权。对照实验比的是「同一套超参能否都学会」，不是「两条 loss 曲线逐点重合」。因此下面这些不可复现是设计，不是断点：

- 自对弈 worker：`random.seed(seed + os.getpid())`（CPU / GPU 相同）。每次启动 PID 不同，对局序列不同。
- GPU `GktTrainer.train(..., seed=0)` 只是 worker 公式里的常数，不是实验种子。
- SGD minibatch：局部未播种的 `random.Random().sample`（与图序 RNG 隔离）。
- 顶点重编号 / 棋盘对称：`grid_sym` 每次 `np.random.default_rng()` 新熵。
- C++：根 Dirichlet、合法着 `shuffle`、对数均匀 sim 与手数、随机让子开局。

**例外：每轮训练图的顺序必须可复现。** 这是为了**跨图访问均匀**，不是为了复现某次实验。固定顺序会让靠前的 key 被多训、共享权下更新次序会变成假信号。实现是 `gkt.shuffle_graph_keys`：对 `str(key)` 列表做 `random.Random(int(rnd)).shuffle`，**私有 RNG、公式固定**，与 SGD / worker 全局 `random` 无关。`--resume` 用同一公式从 `summary.json` 末行切到下一张图（`resume_graph_cursor`）。JSON 里的 key 一律当字符串比（避免 `0` / `"0"` 对不上而整轮静默跳过）。图集合对不上时打日志并进下一 round，不假装还能接着同一排列。

Zobrist 表用固定种子 `0x5EEDC0DE` 是**规则身份**（Python / C++ 劫哈希一致），不是训练随机。内置 R 图的种子是图规格：换种子就是另一张棋盘。

---

## 9. 并行化（搜索侧）

单卡（典型 RTX 4050 Laptop 6GB）上，self-play 为多进程：规则/MCTS 在 C++，每进程自己 `predict_batch`。围棋主线为 `1 worker × gpw 32`（每 cycle 32 局），名义 `--sim 256`。CPU 与 GPU 一样：**每个 worker 提交一次**、`n_games=gpw`（1DCNN 约 60MB 权只 pickle 一次/worker，而不是每局一次）。叶子评估 `--selfplay-batch`：`gkt_train_gpu.py` 与 `gkt_dist.py` 默认 64；`gkt_train_cpu.py` 默认 32。`start_train.bat` 只补 `--net`、`--infinite` 和已有 checkpoint 的 `--resume`，不改写 sim / gpw / selfplay-batch。

不要把多局串进单进程做集中评估：规则/选择已在 C++，叶子仍是每进程自己 `predict_batch`。现在卡在 GPU 前向（注意力 \(n\times n\)）和单卡吞吐。

不要用「提交叶子后继续 select」的异步 inflight：每次下潜先减 `virtual_loss`，回传再加回；多批挂在同一节点会把 Q 压歪。KataGo 用原子操作保证每节点虚拟损失只计一次；Python 回调做不到等价语义。

因此：瓶颈在 GPU 叶子与注意力 \(n^2\)；必须在 **GPU device** 上 profile；提速路径是多卡各跑 worker，或改线性/稀疏注意力。

`GktTrainer`：C++ 生成样本；GNN/2DCNN 叶子默认 `maybe_script_infer`（`jit.trace`，优先 `check_trace=True`，再用随机 batch 对照 eager；`freeze` 通过对照才采用）。失败回退 eager。主进程 `train_on_batch` → `get_weights` 广播。构建：`python cpp/build.py`；Windows / VS 18 用 `python cpp/_build_with_sdk.py`（会先载入 `vcvars64` 并注入 `ScopeCppSDK`），再 `python cpp/_deploy_pyd.py` 把 `gkt_native.*.pyd` 拷到 `scr/`。`scr/` 里的旧 pyd 若被训练进程占用，需先停再拷。

多机异步闭环的**启动、与本机训练共用权、HTTP 安全**只写在 [`training_method.md`](training_method.md) §8。搜索仍是每进程自己的 `predict_batch`，没有参数服务器。

---

## 10. Web UI

`python web_ui/server.py`：标准库 HTTP，单线程，状态在 `STATE`，绑 `127.0.0.1`。POST 必须带 `Content-Length`（缺 411）；JSON 体上限 1 MiB、模型上传 64 MiB（超 413）；请求 `timeout=30`。  
选手 2/3/4 人，每人真人或模型。下拉只列出与本局人数一致的 checkpoint（标签 kP）；对不上拒挂。无匹配权则该席随机（均匀合法着；图围棋含虚手，五子棋 / 反五子棋不含），sim 禁用。模型席 MCTS 恰好 `sim` 次，根 Dirichlet \(f=0\)（与 Arena / Elo 相同，不是自对弈的 \(0.25\)），并做随机顶点重编号（GNN 同步换邻接），走子仍为访问 `argmax`。若存在 `models/elo.json`，下拉可显示 Elo（展示用）。2DCNN 仅网格图。终局画布展示 **finalize 填空之前** 的占领；图围棋分数栏在 `finalize()` 之上另加对局贴目（三人默认约 7.5/4.5/0，四人约 7.5/4.5/2.5/0，末位 0）。规则下拉：图围棋 / 五子棋 / 反五子棋（后两者两人、无虚手条）。界面配色图围棋偏红、五子棋偏绿、反五子棋偏蓝紫。

布局：矩形网格；图 3 / 5.5 可拖动 wrap（按屏幕铺格点、占用周期取模、未平移主区外虚化，同一套路）；图 5 三角菱形；图 6 为 19³ 可拖旋转立方体（空交叉点不画圆点；点外侧后只亮三平面；蓝圈为选点；Pass 下 Confirm / Cancel；棋盘下可改 x/y/z，W/S·Q/E·A/D 微调）；图 7 路图画成 M；图 4 / 4.5 罗宾逊投影（可拖旋转）；无 LAYOUT 的随机图用力导向。图 1 / 5 / 5.5 / 7 默认横放。图 2（61×61）可下：默认训练不含此图，给 2DCNN / GNN 做训练后网格泛化检查。回放时 AI 着可叠该手 MCTS 访问占比与落子前 ownership（人着无网分析）。

### 10.1 随机图上的讲解、对话与 drone 指挥（规划）

**组装门槛：** 讲解、对话与 drone 在**第一个有显著棋力的模型出世前不考虑组装**。在那之前本节只作规划：不接对话 API、命令解析或区域掩码搜索。回放叠网分析（访问占比 / ownership）已有，不属于讲解产品。

力导向可以把 R 图「尽量撑开」，对人仍是一团乱麻：没有棋盘坐标、没有「角地边」口语，点选顶点不可靠。产品方向不是让人在头发团上点格子，而是：

1. **局势网**（已有）：GNN 等给出 policy / value 与辅助头，作为盘面的结构化感知。
2. **语言模型**（要接）：人用自然语言问形势、下决心；模型只根据引擎事实 + 辅助头说话，不凭布局图编造。
3. **drone**（战术执行体）：人给意图（杀哪块、做活、收气、侵消），由网 + MCTS 在约束下精细落子。人掌握的是**图上的作战意图**，不是逐点坐标。

搜索用 policy + value + stdev。讲解与对话走完整 `forward`（GPU）或 CPU 对应输出。多人按**被讲解的座位**抽视角特征。训练不加贴目；展示胜负时可在话术层加上 `komi_schedule`。

| 头 | 训练目标 | 给语言模型 / 人的接地 |
|---|---|---|
| `own` | 终局归属 \([-1,1]\) | 哪块更像我的地、哪里在抢 |
| `future` | 约 4 手后占子 | 短线会往哪长 |
| `opp` | 下一家 MCTS 访问分布 | 对方下一手可能 |
| `soft` | 高温软化本方策略 | 候选着集合 |
| `belief` | 终局 lead 的 21 档 | 胶着 / 略好 / 大胜 |
| `stdev` | \(\lvert\mathrm{lead}-Q\rvert\) | 形势不明则不准说满 |
| policy / value | 着法与领先 | 主变化；drone 的默认战术先验 |

**接地规则（必须）：** 语言模型不得发明顶点编号。编号、块、气、合法着只来自 C++ 引擎；归属/未来/候选只来自辅助头。回复里若提到「那块龙」，必须能指回引擎的块 id 或顶点集合。力导向图只是辅视，不是推理依据。

**drone：** 一次命令是结构化意图（目标块或顶点集、目标：杀 / 活 / 逃 / 收气 / 围空 / 停手），执行器用现有合法着过滤 + 可选区域掩码的 MCTS（或根 policy 在掩码上采样）。人可以改口、追问「为什么下这里」，再用辅助头解释，而不是改去点头发团。

**未做：** 无对话 API、无命令解析、无区域约束搜索；future / opp / belief 不进画布。回放已能叠 AI 着的访问占比与 ownership（另一次 `forward`，不进 MCTS）。`predict_batch` 给搜索回 policy + value + stdev。规划顺序仍是「完整 forward → 张量摘要 → 带工具的对话」，再接 drone 掩码。

---

## 11. 技巧亮点（实现层，不是规则）

1. **图无关参数化**：同一套权在 7×7、19×19、随机有向图上轮换，邻接不当成可学习参数。
2. **视角相对独热 + 图可算特征**：通道 0 永远是「我」。checkpoint 的 \(F=\texttt{feature_dim}(k)\)，2P/3P/4P 分训。
3. **C++ 规则与搜索**：合法着、提子、超劫、MCTS、自对弈均在 `cpp/`。
4. **Zobrist SSK**：历史集合存整数，落子 O(1) 更新。
5. **不可变 Position + 懒展开 MCTS**：少拷棋盘、少构造未访问子树。
6. **虚拟损失批量叶子**：PUCT + 虚拟损失；inflight 深度保持 1 个 batch 语义。
7. **入/出求和消息 + 残差流 LN + 一层全局注意力**：有向气与全图势；logit / stdev 封顶。
8. **价值 = \((k s-S)/((k-1)n)\)、辅助 = 归属**：训练不依赖贴目；Web 对局用 `komi_schedule` 判胜。
9. **对数均匀随机 sim 与手数**：名义 `--sim`、课程 `cap` 为几何中心；弱搜与长短局都见到。五子棋 / 反五子棋不随机化手数上限（固定 \(n\)，避免中盘截断被标成和）。
10. **课程停表 + 手数抖动 + 领地标签**（课程默认关闭）：图围棋名义 cap 随 round 涨（默认关闭则全程全长），每局再对数均匀抽 \([\mathrm{cap}/2,2\mathrm{cap}]\)；停表后 `finalize` 仍给中局监督。
11. **训练/搜索置换与规则正交**：随机顶点重编号与合法着 shuffle 不进入劫判断；不把对称局面写成超劫。
12. **图级可复现 shuffle + `summary.json` resume**（§8.6 的例外）：只保证「这一轮每张图各到一次、中断后续下一张」；不保证对局与 SGD 可复现。每图另有 `progress/summary_{key}.json` 累积各 round。
13. **鸭子类型网络**：MCTS 不知道背后是 GNN、2DCNN、MLP 还是 1DCNN。
14. **掩码用有限 \(-10^9\)**：避免 `0 * log_softmax(-inf)` 变 NaN。
15. **虚拟损失语义**：inflight 深度为一个 batch；Python 异步评估器无法保证每节点虚拟损失只计一次。
16. **连续分数差做 Arena / Elo 信号**：对局不记 0/1 胜负，而记归一化子数差 `score_lead`（与 **value_rto** 同一尺度）。先手优势 δ 是加性项，交替先后手后精确抵消（`E[lead]=棋力差/n`），强弱比较不依赖贴目；0/1 胜负在 δ 巨大的图（线图/乱图）上会因「谁先谁赢」截断棋力差。
17. **大快照面板 Arena**：对手不止 best，而是 best + 上一轮小快照 + 最近 N 个大快照——短程抓「单轮崩」、长程抓「缓慢漂移」；默认 `--arena-lead-threshold 0.0`（不退步即接受）。

---

## 12. 内置图（实现元数据）

| key | 几何 | 训练 | UI |
|---|---|---|---|
| 0 / 0.5 / 1 | 矩形网格 | 是 | 网格（1 默认横放）；五子棋 / 反五子棋 UI 将 0 显示为 G19、0.5 显示为 G7 |
| 2 | 61×61 | 默认排除（过大；2DCNN 训练也不用） | 网格；五子棋 / 反五子棋 UI 显示为 G61。训练后可在此盘评估网格泛化 |
| 3 | 环面 | 是 | 可拖动 wrap 格点（与 5.5 同套路） |
| 4 | 经纬+两极 | 是 | 罗宾逊（可拖旋转） |
| 4.5 | 空心立方壳 | 是 | 罗宾逊（可拖旋转） |
| 5 | 三角菱形 | 是 | 菱形（默认横放） |
| 5.5 | 三角环面 | 是 | 可拖动 wrap 格点（默认横放） |
| 6 | 19³ | 默认排除 | 可拖旋转立方体；三平面切片；Confirm / Cancel；棋盘下 x/y/z |
| 7 | 路 | 是 | M 形折线（默认横放） |
| R1–R5 | 随机有向（见下） | 是 | 力导向 |
| G9 / G15 | 五子棋 4-邻接 9×9 / 15×15 | 否（围棋训练排除） | 网格 |
| G7d / G9d | 五子棋 8-邻接 7×7 / 9×9 | 否 | 网格 |

2DCNN 仅带 `.grid` 的矩形（围棋**训练**：0 / 0.5 / 1 / 3，不含 oversized 的 2；五子棋 / 反五子棋另加 G*）。图 2、图 6 可在 UI 对弈（默认训练排除）。GNN / MLP / 1DCNN 不限网格。

R1–R5 由 `graphs.py` 的 `_random_bounded_degree_graph` 生成（给定 \(n\) 与种子可复现），对应 [`rules.md`](rules.md) §0：平均入度过高则围棋策略退化。先沿随机置换加一条有向哈密顿环（强连通，每点入/出度先为 1），再随机加边，目标约 **\(3n\) 条额外弧、平均入度（因而平均出度）约为 4**；每点入度、出度均 **不超过 6**，禁止自环与重边。

---

## 13. 五子棋 / 反五子棋引擎

规则与实验主张：[`gomoku.md`](gomoku.md)。C++ `Game(..., rules="gomoku"|"antigomoku", win_length=5)`，必须带 `.grid` 且 \(R,C\ge 2\)、\(R\cdot C=n\)。合法着是空点；`play(n)` 非法，`apply_pass` 恒等。胜负扫描坐标四向连珠，不读 CSR。反五子棋把成线方判负（`winner` 为对方）。MCTS / 自对弈同一套，价值 \(\{+1,0,-1\}\)，手数上限 \(n\)。轮训图 `GOMOKU_TRAIN_KEYS`。图围棋默认图列表排除 `G*`。训练入口见 [`training_method.md`](training_method.md) §10。
