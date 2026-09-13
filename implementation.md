# Graphic-KataGo · 实现

本项目为 **Graphic-KataGo**，缩写 **GKT Go**。本文件描述 **`scr/`** 中的实现：架构、技术栈、模块、训练目标、并行化边界与技巧亮点。  
图围棋规则只写在 [`rules.md`](rules.md)；五子棋与反五子棋写在 [`gomoku.md`](gomoku.md)。图围棋检验 **GNN 完备性**；五子棋 / 反五子棋检验 **2DCNN 完备性**。

**架构**：图 → C++ 规则引擎 / MCTS / 自对弈 → CPU 或 GPU 网络 → Python 跨图训练循环 → Web 对弈。  
**主体名 GKT Go**：搜索/自对弈热路径在 `cpp/`（`gkt_native`），Python `gkt.py`（`GktSelfPlay`）只做调度；GPU 并行训练器 `GktTrainer`。文件与类名里的 `gkt` 即此缩写。  
**规范树**：文档 `ref/`，Python `scr/`，C++ 热路径 `cpp/`。

## 研究策略（主线：蒸馏，理论线：从 0 自对弈）

本项目要回答的命题是 **图无关性（graph-independence）与权重可迁移（weight transferability）**：同一套网络权重能否不依赖棋盘拓扑、在任意有向图上维持棋力。围绕这个命题有两条路线，**优先级已重新划定**：

1. **蒸馏（实际主线，优先）**：用官方 KataGo 的强标签（policy 访问分布 / score lead / ownership）监督预训练图无关网络，得到**已经有棋力**的 **base 模型（「基础培养」）**（`base/` 启动器 + `scr/distill.py` + `scr/distill_katago.py`）。四种架构都配了蒸馏入口——`gnn`（图无关主线）、`2dcnn`（传统卷积对照）、`mlp` / `1dcnn`（无图结构 / 仅水平局部性的消融对照），分别产出 `base/{gnn,cnn2d,mlp,cnn1d}/`。它绕开了「value 头在弱信号下塌缩、自对弈 bootstrap 死锁」这一小算力卡死的瓶颈，让 value 头从第一步就收到有区分度的真实信号。**后续所有实验（跨图迁移、消融、迁移边界探测）都以这个蒸馏出的 base 模型为基础**。
2. **从 0 自对弈（理论线，算力充足时的理想）**：AlphaZero / KataGo 式「随机初始化 → 自对弈 → 强化」的闭环（`gkt_train_gpu.py` / `gkt_train_cpu.py`）。这是命题的**理论完备形态**——完全无外部监督、纯靠图无关自对弈收敛。但实测单卡（RTX 4050 6 GB）距离 AlphaGo Zero「~20 万局脱离随机」的门槛差 **70–370 倍数据量**，单机不可行，故**降级为「算力充足时」的理论路线保留**，仅留 `M0`（7×7 脱离随机的 sanity gate，见 `cur_mod_gnn/experiment/M0.bat`）作为最低限度验证，不再作为当前主推进度。

两条线共享同一套 12 元组样本、同一 F=8 特征编码（`extract_features`）与同一 checkpoint 格式（GPU 网 `new.pt`、CPU 网 `new.npz`），蒸馏产物可被自对弈训练器 `--resume` 原样加载续训。**文档其余部分描述的规则、MCTS、多头目标、跨图循环均为两条线共用的基础设施。**

启动前先编译 native：`python cpp/build.py`（C++17 编译器 + `pip install pybind11`；Windows 走 setuptools/MSVC，CMake 可选）。**pip 一律走清华源**（`start_ui.bat` 已写死；也可设为默认）：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --upgrade pip
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install pybind11 numpy
```

CUDA 版 `torch` 不在普通 PyPI 里，主源仍用清华，轮子从 PyTorch CUDA 索引补：

```bash
python -m pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu130
```

```bash
python cpp/build.py
cd scr
python distill.py --net gnn --data ../distill_data/m2_19x19.jsonl --outdir ../base/gnn   # 主线：蒸馏出 base 模型
python gkt_train_gpu.py --sim 256 --workers 1 --gpw 32 --steps 16 --lr 1e-4 --value-weight 3000 --own-weight 25 --temperature 0.1 --device cuda   # 蒸馏后续训 / 理论线自对弈
python web_ui/server.py --port 8765 --device cpu
```

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
| CLI | kebab-case | `--selfplay-batch`, `--value-target` |
| 文档互引 | ``ref/*.md``、``scr/*.py`` | 例如 ``ref/rules.md`` |

**网络鸭子接口**（四种网络共用）：

- `predict(X, legal_mask) -> (policy, value)`
- `predict_batch(X_batch, masks) -> (policy, value)`；GPU 另回 `stdev`（第三项）。缺省时 C++ 按 \(\mathrm{softplus}(0)=\ln 2\) 填
- 训练：CPU 用 `backward(...)` 逐样本；GPU 用 `train_on_batch(...)` 批量
- 可选：`set_graph` / `get_weights` / `set_weights` / `state_dict`

对应关系：

| 设备 | 文件 | 网络 |
|---|---|---|
| CPU（NumPy） | `gkt_cpu.py` | **MLP**、**1DCNN** |
| GPU（PyTorch） | `gkt_gpu.py` | **GNN**、**2DCNN**（仅网格） |

CLI：`--net mlp|1dcnn`（`gkt_train_cpu.py`），`--net gnn|2dcnn`（`gkt_train_gpu.py`）。主体名 **GKT Go**（`gkt.py` / `GktSelfPlay` / `GktTrainer`）。

**自对弈样本**一律 12 元组：

`(features, legal_mask, policy, me, score_lead, ownership, q, opp_policy, opp_weight, future, lead, weight)`

- `score_lead`：不含贴目的 \((k\cdot s_{\mathrm{me}}-S)/((k-1)n)\)（二人即 \((\text{我}-\text{对方})/n\)）。`--value-target` 为 `q`/`mix` 时这一槽是 Q 或混合
- `ownership[i]\in[-1,1]`：终局该点从 `me` 视角的归属
- `q`：该步根节点 MCTS Q（与 lead 同量纲），给 stdev 头
- `opp_policy`：下一手（下一家）的访问分布；最后一步 `opp_weight=0`
- `future`：约 4 手后的占子图（C++ `FUTURE_PLIES`；+1 我 / −1 对方 / 0 空）
- `lead`：终局同一公式，给 score-belief，不随 `mix` 搅浑
- `weight`：policy surprise \(\mathrm{KL}(\pi_{\mathrm{MCTS}}\|\pi_{\mathrm{nn}})\) 映射到 \([0.5,8]\)，SGD 按样本加权
- GPU 损失：`(policy, value, own, aux, total)`，`total` 含加权辅助项
- CPU `backward` 返回四元组，`total` 含辅助损失

MCTS 用 policy + value(lead) + stdev（加权备份、方差缩放 cPUCT）。其余辅助头不进搜索。随机图上的**对话讲解与 drone 指挥**要接语言模型，并以辅助头和引擎为接地，见 §8.1。

**搜索次数**：构造参数名 `n_simulations`，实例字段 `n_sim`，命令行 `--sim`。自对弈逐步搜索在 \([\mathrm{sim}/4,\,4\cdot\mathrm{sim}]\) 对数均匀抽样。Arena / Elo / UI 用固定次数（UI 模型席 = 该席 `sim`，不少于 1）。Arena 与 UI 的根 Dirichlet 与自对弈相同（`DIRICHLET_FRAC=0.25`）；Elo 仍为 0。

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

**刻意不用的**：稀疏图卷积库、分布式参数服务器、异步 GPU 评估器、WebSocket、数据库。单卡笔记本上这些没有收益（§7）。

训练与 UI **必须**加载 `gkt_native`（失败即报错，不回退慢路径）。Windows 入口 `start_train.bat` / `start_ui.bat` 钉死 **Python 3.14**（`D:\Python\pythoncore-3.14-64\python.exe`，缺则 `python`）。不要用 `py -3`（可能开到 free-threading 的 `python3.14t.exe`），也不要用 3.13：扩展是 `gkt_native.cp314-win_amd64.pyd`，ABI 对不上会 ImportError。`cpp/build.py` 缺 pybind11 时用清华源安装。

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
| `distill.py` | **主线**：KataGo 强标签监督预训练（policy/value/own 三头），产出 base 模型（`--net gnn\|2dcnn\|mlp\|1dcnn`） | 自对弈、辅助头 |
| `distill_katago.py` | KataGo analysis JSON → 蒸馏 JSONL（唯一知道 KataGo 坐标的地方） | gkt 顶点顺序 |
| `gkt_elo.py` | 全算法共享 Elo 榜（仅展示） | 训练、门控 |
| `gkt_dist.py` | KataGo 式 selfplay/shuffle/train/gate/contribute | 单机 `gkt_train_*` |
| `web_ui/` | HTTP + Canvas | 训练 |

---

## 4. 规则引擎要点

- `Position` 冻结；落子返回新对象。实现在 C++（`cpp/src/engine.cpp`）。默认 `rules=go`（[`rules.md`](rules.md)）；`rules=gomoku` / `antigomoku` 见 [`gomoku.md`](gomoku.md)。
- 图围棋：气 / 块在 **CSR**（入邻接 + 无向并）上计算。超劫用 Zobrist 增量 XOR。虚手只换回合盐、不查重复禁手。种子 `0x5EEDC0DE` 与 Python `random.Random` 对齐。二人单遍合法着（`legal_moves_fast` 与 `try_move` 同一套 §4.1：提完他方后己方仍须有气）；多人逐点 try-move。`legal_moves_for` 含虚手（下标 \(n\)）；`Game.play(n)` 即虚手。`try_move` 校验 `player == to_move`（三种规则相同）。终局后 `play(n)` 与 `pass_move()` 均非法（同一条路径）；底层 `apply_pass` 仍恒等。未知 `rules` 字符串拒收，不默当成图围棋。Python `predict_batch` 失败或 policy 长度不是 \(B\times(n+1)\) 时抛错。畸形 `occupancy`（长度 ≠ \(n\)）读取时抛错。
- 五子棋 / 反五子棋：合法着仅为空点；不读 CSR 判胜，只读 `.grid` 坐标四向连珠。必须 \(R,C\ge 2\) 且 \(R\cdot C=n\)。`play(n)` 与 `pass_move()` 均非法。反五子棋成线方负。
- 终局列表为空。图围棋 `finalize()` 返回无贴目领地分；五子棋 / 反五子棋为 \(\{0,1/2,1\}\)。Web UI 对图围棋再加选手贴目。训练自对弈、Arena、Elo **不加贴目**。[`rules.md`](rules.md) §5.4 的 7.5 是 UI 默认，与 `Game()` 全 0 构造无关。

---

## 5. 搜索与训练目标

### 5.1 特征

每顶点 \(F=\texttt{num_players}+6\)：**视角相对**占领独热（通道 0 = 当前走棋方），再拼 1 气 / 2 气 / 3+ 气、\(\log(1+\text{块大小})\)、上一手 one-hot、刚被提的空点。编码前再派生一维空点 \(1-\sum\mathrm{onehot}\)。气按规则是块的空入邻居。权不依赖 \(n\)。\(F\) 必须等于 `feature_dim(num_players)`（2P 即 8）。邻接只进 GNN。

### 5.2 MCTS

- 懒展开：子节点先不物化 `Position`。
- 批量叶子 + **虚拟损失**：一批内后续选择偏离已走分支。
- 根 Dirichlet（\(\alpha=0.3\)，\(f=0.25\)）：训练自对弈、Arena、UI。自对弈走子按温度 1 采样；Arena / UI 仍 `argmax` 访问，sim 固定。Elo：`argmax` 且 `dirichlet_frac=0`。
- **顶点重编号（防「编号迷信」）**：训练自对弈的**每次**叶子评估（`predict_batch`）都抽新随机置换——GNN / 1DCNN / MLP 走 \(S_n\)（GNN 同步置换 `adj`），2DCNN 走 grid2d 棋盘对称（D4 / Klein / torus shift，非 grid 图 fallback \(S_n\)），策略 gather 回原编号，让网络永远无法把顶点编号当稳定信号。Arena / UI / Elo 每次 `search` 抽一次置换、搜索内叶子共用（`SearchAugNet` 非 fresh 模式）。两者都经 `native.search`。
- 合法着每节点 `shuffle`，打破「平局总走下标 0」——这是搜索层公平，独立于上面的模型层重编号，二者正交。
- 叶子对每个座位各做一次视角特征，得到 \(k\) 维 lead；备份时路径上每节点取**该节点执子方**的分量，**不是**二人翻号。价值公式见样本 `score_lead`（二人即 \((\text{我}-\text{对方})/n\)）。
- **FPU**：未访问子的 Q 为 \(\mathrm{parent}Q - 0.2\sqrt{\sum \pi_{\mathrm{nn}}(\text{已访问子})}\)，不是 0。
- **stdev**：叶子权重 \(w=1/(1+\sigma^2)\)；节点 Q 为 \(\sum wv/\sum w\)（不是 \(\sum wv/N\)）；cPUCT 乘 \(\sqrt{\mathrm{Var}(Q)+\varepsilon}/\sqrt{\varepsilon}\)。
- 训练目标：访问分布先丢掉「次数低且先验也低」的噪声着，再归一；样本权重来自 surprise。

### 5.3 KataGo 式多头监督

主干宽度与搜索算法固定；训练同时拟合 KataGo 式辅助头：

| 头 | 目标 | 损失（权重） |
|---|---|---|
| policy | 剪枝后的访问分布，长度 \(n+1\)，下标 \(n\) 为虚手维（五子棋该维 mask 为 0） | 按 surprise 加权的 CE |
| value | `score_lead`，可选 `mc` / `q` / `mix` | MSE（tanh 前向；反向雅可比 \(\max(1-\tanh^2, 0.2)\)） |
| ownership | 终局每点归属 | 同上 |
| opp policy | 下一手（下一家）的访问分布 | CE × 0.25，末手权重 0 |
| soft policy | \(\pi^{1/T}\)，\(T=4\)，合法着上再归一 | CE × 0.15 |
| score belief | lead 在 \([-1,1]\) 上 21 bin 的软直方图 | pdf CE × 0.30 + cdf MSE × 0.30 |
| stdev | \(\lvert\mathrm{lead}-Q\rvert\) | MSE × 0.15（softplus 前向，反向 \(\max(\sigma, 0.2)\)） |
| futurepos | 约 4 个决策后的占子 | MSE × 0.25（tanh 前向，反向同 value） |

`--value-target`：`mc` 终局 lead；`q` 根节点 Q（同一量纲）；`mix` 为 \(\lambda\cdot\mathrm{lead}+(1-\lambda)\cdot Q\)。belief 始终用终局 `lead`。value / ownership / futurepos / stdev 的末层权为零。反向雅可比 \(\max(1-\tanh^2, 0.2)\)（stdev 为 \(\max(\sigma, 0.2)\)）：饱和处仍有梯度，又不会把 squash 当成恒等。日志 vloss / oloss 仍是输出空间的 MSE。

加载权重必须键齐全：GPU `load_state_dict(..., strict=True)`，CPU 按 `state_dict` 精确匹配。checkpoint 必须带 `net_type`（GPU `.pt`）或 `_net_type`（CPU `.npz`），以及人数 `num_players` / `_num_players`；\(F=\texttt{feature_dim}(k)\)。缺键或 \(F\) 不符则拒载。

### 5.4 随机让子开局

仅**自对弈训练**（`make_training_game`）。Arena / Web / 评估从空盘开始。

每局先抽 \(p\sim\mathcal{U}(0,\,p_{\max})\)，每个顶点独立以概率 \(p\) 放子（颜色在 \(1..k\) 上均匀），否则为空。图围棋 \(p_{\max}=0.25\)。五子棋 \(p_{\max}=0.12\)：连珠很容易在随机开局上提前成五，更稀的占子才能留下可学的中盘。然后用规则引擎对静态盘提子：从末座到首座依次提无气块（二人即先白后黑）。子数最少的一方先行（二人：黑多于白则白先）。Zobrist 按整盘占领重算，历史为空。

### 5.5 随机搜索强度

`randomized_sim_count(nominal)`：\(\exp\mathcal{U}(\log(\mathrm{sim}/4),\log(4\cdot\mathrm{sim}))\) 再四舍五入。自对弈每步调用；Arena / Elo / UI 不随机。意图：策略在弱搜下不崩。自对弈手数上限用同一变换、区间为 \([\mathrm{cap}/2,\,2\cdot\mathrm{cap}]\)，每局抽一次。

### 5.6 跨图训练

- 排除超大图 `2`、`6`（顶点数过大）。二者都不进默认训练，但 UI 都可下：图 2 作网格泛化检查；图 6 为可拖旋转的 19³ 切片棋盘。
- 每轮 `random.Random(rnd).shuffle` 图顺序（GPU / CPU 相同，可复现，供图级 resume）。`--resume` 读 `summary.json` 末行，**cycle 序号接续**，并从下一张图接着训。**若末行恰好是该轮最后一张图（该 round 训练已完成）且 Arena 开启**，则不再静默跳过该轮的 Arena 门控，而是先补跑这一次门控（图列表置空、跳过 round-end 产物，直接落入 Arena），再进下一轮；`best_weights` 此时从 `best.*` 恢复（无 `best.*` 则回退当前权重）。
- 课程（**默认关闭**，`--curriculum-rounds 0` = 蒸馏强起点 regime 全程全长；从 0 时显式传 `--min-moves 40 --curriculum-rounds 20000` 可重启）：`curriculum_max_moves`（`gkt.py`）：名义手数上限从 `--min-moves` 线性涨到 \(n\cdot\mathrm{factor}+\)min-moves（`--min-moves 40 --max-move-factor 2.0`，即 40 → \(2n+40\)）。自对弈每局再按与 sim 相同的对数均匀抽 \(\exp\mathcal{U}(\log(\mathrm{cap}/2),\log(2\cdot\mathrm{cap}))\)（Arena / Elo / UI 用名义上限、不随机）。停表后 `finalize` 提供中局领地监督。图围棋自对弈在前 \(\texttt{forbid_pass_after}=\min(\mathrm{cap},\,\max(\lfloor n/8\rfloor,16))\) 手若仍有合法落子则 **MCTS 去掉虚手**（规则层虚手仍合法；UI / Arena / Elo 不禁）。
- 每图结束（CUDA）：`gc.collect()` + `empty_cache()`，减轻切图后分配变慢。
- 自对弈：同一套权打满 \(k\) 个座位（不是 \(k\) 个独立对手）。
- Arena（**默认开**，`--no-arena` 关）：新 vs 面板跨图门槛，**据此接受或回滚**；固定 `--arena-sim`。面板 = best-so-far + 上一轮小快照 `round{rnd-1}` + 最近 `--big-snapshot-rounds` 个大快照 `big*`（长程防缓慢漂移）。新网轮流占一座、对手占其余；每局记**连续归一化分数差** `score_lead`（value 头同一目标，非 0/1 胜负），交替先后手使先手优势 δ 作为加性项抵消（`E[lead]=棋力差/n`），再与 `--arena-lead-threshold`（默认 0.0 = 不退步即接受）比。训练仍无贴目。**带心跳**：开始时打一行 `arena: round N gate starting — P opponents x G graphs x Q games (...)`，每打完一个「图×对手」打一行 `arena: [i/total] graph K vs L: lead +x.xxxx (g games, s秒)`，结束打 `arena: round N played ... games in ...s`——沉默期从「一整段」变成「可数进度条」。
- `--players {2,3,4}`：自对弈人数与 \(F=\texttt{feature_dim}(k)\) 一致；2P/3P/4P 分网。缺人数键或 \(F\neq\texttt{feature_dim}(k)\) 拒载。
- `--lr` 默认 \(1\mathrm{e}{-4}\)。非有限 loss / 梯度跳过该次 SGD；拒绝把非有限权写入 `new.pt`。
- **每图 unused 队列**：`outdir/replay/{key}/unused.npz`。SGD 成功用过的样本从 buffer 删除、不写回。未用样本留在该文件里，**下次再训同一张图时仍会加载**。条数上限 `buffer_capacity`（默认 20 万，超出留最新尾巴）。图与图不混。`--replay-rounds 1` 不读盘上队列。`--resume` 读已有 `unused.npz`。从 round `--buffer-drop-from-round`（代码默认 11；围棋主线 starter 传 5 = 保留 5 代 replay）起，每训完一张图按 `--buffer-drop` 丢掉队列最旧的样本（默认 auto = 本轮未使用量 `new - consumed`，即「比例固定 1.0」的换血语义）。
- Elo（`gkt_elo.py`）：四种架构 + 均匀随机共用一张榜，**只供展示**。默认只评 2P 文件。对局协议与 Arena 相同：交替先后手 + 连续 `score_lead` 信号（线性映射到 \([0,1]\) 作 Elo 得分），故 Elo 差 = 平均每点优势、跨图可比、不依赖贴目。基准分 1500、尺度 400、K=24。不替代 Arena。

```bash
cd scr
python gkt_elo.py --models-dir ../models --sim 100 --games 2 --device cpu
```

### 5.7 蒸馏预训练（主线入口）

`scr/distill.py` 把 KataGo 强标签监督拟合到图无关网络，产出 **base 模型（「基础培养」）** 的 checkpoint（GPU 网 `new.pt`、CPU 网 `new.npz`，`--resume` 可续）。这是当前产出**第一个有棋力模型**的入口，后续跨图迁移 / 消融实验都从它出发。`--net` 支持四种架构：`gnn`（图无关主线）、`2dcnn`（传统卷积对照）、`mlp`、`1dcnn`（消融对照）。

- **三阶段蒸馏**：每个阶段 `--epochs` 轮（默认 10+10+10）。① **联合** policy + value + own（**不冻结**任何参数，Adam `lr=1e-4`），loss = `pl + 30*vl + 5*ol`（`--value-weight 30 --own-weight 5`，报告的 pl / vl / ol 仍是未加权原值）；② 只训 own 头（**冻结 trunk**，Adam lr ×25，unweighted MSE）；③ 只训 value 头（冻结 trunk，Adam lr ×100）。不要用 3000/25 去乘 MSE（那会在 `clip_grad_norm=1` 下把政策梯度挤没）。GPU 用 Adam（β=0.9/0.999）；CPU 蒸馏同样用 NumPy Adam。围棋自对弈 starter 的 value/own 权重与学习率不变。
- **特征逐位一致**：`build_sample` 用 native 引擎 `extract_features` 算 F=8 特征，与自对弈完全同源；value/own 的归一化公式与自对弈 `score_lead` / `ownership` 语义完全对齐（仍是 `/n`，跨图可比）。
- **训练增强、验证集固定**：训练集每个 batch 按网络分派与自对弈 SGD 相同的置换——GNN / MLP / 1DCNN 随机 \(S_n\)（GNN 同步换邻接 \(A'[i,j]=A[\pi(i),\pi(j)]\)，与 `SearchAugNet` 一致），2DCNN 走 grid2d D4 / Klein / 环面平移。**课程打乱**：默认 `--aug-from-epoch 4`，全局 epoch 1–3 用原编号，从第 4 轮起才打乱（`1` = 第一轮就打乱，`0` = 全程不打乱）。**验证集仍是 JSONL 顺序前 `val-frac`（默认 5%）、不做增强、不重采样**，专门用来暴露过拟合。value 是标量，不随置换改变。心跳行带 `aug=on/off`。
- **逐步心跳**：`outdir/progress.txt`，每个 epoch 开始时清空；逐 train/val step 打 `pl/vl/ol`，行首带 `stage=policy|own|value e{k}/10 g{G}/30`（`fsync` 约 1s 节流）。控制台仍打 epoch 汇总。
- **数据流**：`distill_katago.py`（KataGo analysis JSON → gkt 顶点顺序的 JSONL）→ `distill.py`（JSONL → 12 元组 → 三阶段 Adam → checkpoint `round1`…`round30`）。启动器 `base/distill_{gnn,mlp,cnn1d,cnn2d}.bat`（默认读 `distill_data/m2_19x19.jsonl`）。
- **内存约束**：19×19（n=361）的稠密邻接 \((B,n,n)\) 在 6 GB 卡上 batch 上限约 **16**（32 OOM）；7×7 可更大。`--hidden 512 --n-blocks 20` 与自对弈默认一致，保证产物结构可被主训练器无缝 `--resume` 续训。CPU 1DCNN 蒸馏改 mini-batch（默认 `--batch-size 64`），`_im2col` 已向量化。
- **与自对弈的关系**：蒸馏打破 value 塌缩死锁后，仍可接自对弈训练器（`--resume base/<arch>/new.pt`）继续微调，作为迁移实验的起点。

---

## 6. 网络

四种网络同一套 `predict` / `predict_batch` 接口；MCTS 不区分实现。角色不同：**GNN** 是图围棋主网（检验图规则下 GNN 完备性）；**2DCNN** 是网格主网（图围棋里当形状对照，五子棋 / 反五子棋里检验 2DCNN 完备性）；**MLP** 与 **1DCNN** 是消融，用来证明 GNN 不是靠宽度瞎拟合。

### 6.1 GPU · GNN（图拓扑主路径）

线性编码（占领 + 空点 + 气/块/上一手/提子）→ LayerNorm → `n_blocks` 层入/出求和聚合 + \(\log(1+d)\) MLP 残差（残差流 LayerNorm）+ global pooling bias（mean / max / \(\mathrm{mean}\sqrt{n}\) 且 \(\sqrt{n}\) 上限 8，投影零初始化）→ 在 `attn_layer`（默认 8）前插入一层全局多头自注意力（同样残差流 LN）→ 逐点 policy（虚手与落子共用同一 `Linear`，作用在 \(\mathrm{mean}(h)\) 上）+ attention-pool value + 逐点 ownership，以及 opp / soft / future / belief / stdev 辅助头（opp / soft 同样共用顶点头）。`state_dict` 含 `pass_head` / `opp_pass` / `soft_pass`，前向不读。policy / opp / soft / belief logit 封顶 \(\pm 20\)，stdev 上限 8。非有限 loss 或梯度则跳过该次 `step`。  
**权重只依赖 \(F\) 与 \(H\)**，不依赖 \(n\)。`set_graph` 注入 0/1 邻接。

入/出分开：规则里气是入邻居。全局注意力打破「感受野 = 层数」，表达势。6GB 卡上注意力 \(n\times n\) 把叶子 batch 卡在约 64。

消息传递用稠密 `adj @ h` 求和：\(n\le 400\) 时 cuBLAS GEMM 快于 scatter；稀疏化留给图 2 / 6 量级。

### 6.2 GPU · 2DCNN（网格拓扑对照）

只在带 `.grid` 的矩形上跑（图围棋默认训练：0 / 0.5 / 1 / 3；五子棋 / 反五子棋默认另加 G9 / G15 / G7d / G9d；都不含 oversized 的 2）。图 2（61×61）不进默认训练，UI 可下，作网格泛化检查。1×1 stem + BatchNorm + 3×3 残差（默认 \(H=512\)、20 块，每块末尾 spatial gpool，\(\sqrt{n}\) 上限 8），环面 circular pad。policy / opp / soft 的虚手与格点共用同一 1×1 conv（空间均值）。policy / opp / soft / belief logit 同样封顶。**不加** GNN 那层全局多头自注意力（value 头仍用逐点 softmax 池化）。`--net 2dcnn`。`--attn-layer` 只作用于 GNN；2DCNN 对象上保存 `attn_layer` / `n_heads`，前向不用。`state_dict` 同样含 `pass_head` / `opp_pass` / `soft_pass`，前向不读。

图围棋下和 GNN 比的是：同一套规则与辅助头下，网格平移对称还剩多少。五子棋下它才是与规则同构的完备性检验（见 [`gomoku.md`](gomoku.md)）。2DCNN 不能吃无 `.grid` 的图。

### 6.3 CPU · MLP（消融）

两层全连接（默认 \(H=512\)，输入为占领+空点+图特征）+ 与 GPU 相同的 attention-pool value / 逐点 ownership。**不加** trunk gpool（消融）。顶点互不见边。用来看「没有拓扑时还能不能装成像那么回事」。`--net mlp`。

### 6.4 CPU · 1DCNN（消融）

按顶点**下标**做一维卷积（默认 \(H=512\)、20 层、核宽 3；编码同样是占领+空点+图特征），邻接是编号顺序，不是图的边。**不加** trunk gpool。用来量「假几何」值多少；GNN 若只是略强于 1DCNN，就还没证明学到了真拓扑。`--net 1dcnn`。权重约 60MB，小图（线图 / 7×7）CPU 可跑；19×19 上 `predict_batch` 仍是逐样本循环，MCTS 会很慢。

### 6.5 顶点置换与对称打乱（编号迷信防御）

顶点重编号从「与 MCTS 搜索绑定」改为「与模型绑定」：无论训练自对弈还是 Arena / UI，模型每次读样本都可能面对随机编号，让网络无法把顶点编号当稳定信号（编号迷信）。

- **训练自对弈**：`GktSelfPlay` 用 `SearchAugNet(net, graph, fresh_each_batch=True)` 包装，**每次叶子评估**（`predict_batch`）都抽新随机置换，策略 gather 回原编号。
- **Arena / UI / Elo**：`SearchAugNet` 非 fresh 模式，每次 `search` 抽一次置换、搜索内叶子共用。
- **按网络分派**：GNN / 1DCNN / MLP（无棋盘几何）走随机 \(S_n\) 全置换——GNN 另用同一 gather 同步置换 `adj_in` / `adj_out`（拓扑不变），1DCNN / MLP 只换点特征（其卷积核序 / 层序是「假几何」，任意重编号正是消融本身）；2DCNN（有棋盘几何）走 grid2d 棋盘自同构 D4（方阵）/ Klein 四元群（非方阵）/ 环面平移，非 grid 图 fallback \(S_n\)（任意 \(S_n\) 会把不相邻顶点拉进卷积邻域、破坏网格先验）。
- **SGD 增强同样分派**：`_gpu_train_on_batch` 里 GNN / 1DCNN / MLP 走 `augment_vertex_batch`（\(S_n\)，GNN 另换邻接），2DCNN 走 `augment_graph_batch`（D4，无需换邻接）。
- **正交的搜索层公平**：合法着每节点 `shuffle`（`mcts.cpp`）独立于上面的模型层重编号，二者不互相替代。

---

## 7. 并行化

单卡（典型 RTX 4050 Laptop 6GB）上，self-play 为多进程：规则/MCTS 在 C++，每进程自己 `predict_batch`。默认与 `train_*.bat` 为 `1 worker × gpw 32`（每 cycle 32 局），名义 `--sim 256`。CPU 与 GPU 一样：**每个 worker 提交一次**、`n_games=gpw`（1DCNN 约 60MB 权只 pickle 一次/worker，而不是每局一次）。叶子评估 `--selfplay-batch`：`gkt_train_gpu.py` 与 `gkt_dist.py` 默认 64；`gkt_train_cpu.py` 默认 32（`GktTrainer` 构造默认也是 32，GPU 跨图入口传入 64）。`start_train.bat` 不改写该旗标。

不要把多局串进单进程做集中评估：规则/选择已在 C++，叶子仍是每进程自己 `predict_batch`。现在卡在 GPU 前向（注意力 \(n\times n\)）和单卡吞吐。

不要用「提交叶子后继续 select」的异步 inflight：每次下潜先减 `virtual_loss`，回传再加回；多批挂在同一节点会把 Q 压歪。KataGo 用原子操作保证每节点虚拟损失只计一次；Python 回调做不到等价语义。

因此：

1. 瓶颈在 GPU 叶子与注意力 \(n^2\)。
2. 必须在 **GPU device** 上 profile。
3. 提速路径：多卡各跑 worker，或改线性/稀疏注意力。

`GktTrainer`：C++ 生成样本；GNN/2DCNN 叶子默认 `maybe_script_infer`（`jit.trace`，优先 `check_trace=True`，再用随机 batch 对照 eager；`freeze` 通过对照才采用）。失败回退 eager。主进程 `train_on_batch` → `get_weights` 广播。构建：`python cpp/build.py`（**若报 `io.h`/`rc.exe` 找不到**，用 `python cpp/_build_with_sdk.py` 注入 VS 18 的 `ScopeCppSDK` include/lib/PATH，再 `python cpp/_deploy_pyd.py` 复制 `gkt_native.*.pyd` 到 `scr/`；`scr/` 的旧 pyd 若被在跑的训练进程占用，需先停进程再复制）。

### 7.1 分布式（KataGo 式异步闭环）

官方 KataGo 是异步进程靠**共享目录**衔接：selfplay → shuffle → train → 可选 gatekeeper；远端再 `katago contribute`。本仓库对应 `scr/gkt_dist.py`。

**单机、只想接着训 `cur_mod_*`：** 继续用 `gkt_train_gpu.py` / `gkt_train_cpu.py` 或根目录 `start_train.bat gnn|mlp|cnn1d|cnn2d`（图围棋包装 `train_gnn.bat` 等带课程旗标；五子棋包装 `train_gomoku_gnn.bat` 等传 `--rules gomoku`，默认目录 `cur_mod_gomoku_*`；反五子棋包装 `train_antigomoku_gnn.bat` 等传 `--rules antigomoku`，默认目录 `cur_mod_antigomoku_*`）。不要和 `gkt_dist` 抢同一套权；分布式是另一条闭环，basedir 默认 `../dist_run`。

**多机：** 只有一台当「服务器」写权，其余当自对弈客户。客户**不必**挂共享盘，走 HTTP（`serve` + `contribute`）。`selfplay` / `shuffle` / `train` / `gate` 必须看见**同一个** `--basedir`（本机磁盘，或 NAS/SMB 映射到同一路径）。

#### 角色怎么分

| 机器 | 跑什么 | 说明 |
|---|---|---|
| 服务器（建议最强 GPU） | `init` 一次；然后长期 `shuffle`、`train`、`serve`；可选 `gate` | 唯一写 `models/` 的地方。`serve` 把已接受网发给客户、收样本进 `selfplay/` |
| 服务器上也可再开 | `selfplay --device cuda` | 本机也产样本；和客户一样写进 basedir 的 `selfplay/` |
| 其他电脑 | 只跑 `contribute --url http://服务器LAN地址:8877` | 拉最新 `.pt`/`.npz`，下一局，POST 样本。`--device cuda` 或 `cpu` |

同一 basedir 只对应**一种**网、**一种**人数、**一种规则**（`init --from` 种子里的 `net_type` / `_net_type` 与 `num_players`；`--net` 必须与种子一致，否则 `init` 拒绝；`--rules go|gomoku|antigomoku` 写入 `run.json`）。2P GNN 和 3P、GNN 和 2DCNN、或图围棋和五子棋/反五子棋，不要混在一个 `dist_run` 里。

五子棋分布式：

```bash
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gomoku_gnn/new.pt --net gnn --rules gomoku --token 自己设一串密文
```

默认图列表为 `GOMOKU_TRAIN_KEYS`；手数上限固定为 \(n\)（不用图围棋课程）。种子必须是 2P。`--win-length` 写入 `run.json`，selfplay / gate / contribute 都会带上。

目录（`--basedir`，默认 `../dist_run`）：

| 目录 | 角色 |
|---|---|
| `run.json` | 图列表、sim、窗口、是否门控、token |
| `models/` | 已接受网络（selfplay / contribute 只读这里） |
| `selfplay/` | 自对弈 12 元组 `.npz`（含 surprise `weight`；一文件一图）；shuffle 只保留每图最近 `window_files` 个 |
| `shuffled/` | 按图滑窗打乱后的训练包；每图只留 `shuffled_keep`（默认 2）份，`train` 每个保存周期换最新包 |
| `modelstobetested/` | 训练刚写出、待门控（仅 `init --gate`） |
| `rejectedmodels/` | 门控未过 |
| `client_cache/` | 服务器本机缓存；客户机用 `contribute --cache` |

#### 服务器上先 init

各机工作目录都是该机上的 `scr/`，解释器不要用 `py -3`（与 UI 相同）。种子用已经训过的 `new.pt` / `new.npz`（例如 `../cur_mod_gnn/new.pt`），`--net` 必须与种子一致：

```bash
cd scr
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gnn/new.pt --net gnn --token 自己设一串密文
```

`init --gate` 才启用门控，此时必须另开 `gate`，训练写入 `modelstobetested/`。默认不加 `--gate`：`train` 直接写入 `models/`，客户下一轮就会拉到新网。

然后在**同一 basedir** 各开一个窗口（可 `--once` 只跑一轮）：

```bash
python gkt_dist.py shuffle --basedir ../dist_run
python gkt_dist.py train --basedir ../dist_run --device cuda
python gkt_dist.py serve --basedir ../dist_run --host 0.0.0.0 --port 8877 --token 与init相同
# 可选：python gkt_dist.py gate --basedir ../dist_run --device cuda
# 可选：python gkt_dist.py selfplay --basedir ../dist_run --device cuda --workers 2
```

`serve` **默认只绑 `127.0.0.1`**，局域网客户连不上。多机必须 `--host 0.0.0.0`（或这台网卡的 LAN IP）。Windows 防火墙放行 8877。不要把该端口暴露到公网；`--token` 写入 `X-Token`，空 token 则不校验。

#### 其他电脑：contribute

每台克隆同一份仓库（图 key、样本元组必须一致），安装同系列依赖（GPU 客户要能 `import torch` 且能加载该 `--net`）。**不要**在客户机再 `init`。把 `服务器LAN_IP` 换成资源管理器里能 ping 通的地址：

```bash
cd scr
python gkt_dist.py contribute --url http://服务器LAN_IP:8877 --token 与init相同 --device cuda
```

无 GPU 则 `--device cpu`。`--cache` 可指到本机目录，避免每次重下整份权。循环：`GET /api/task` → 必要时 `GET /api/model/…` → 下一局 → `POST /api/games`。服务器把 npz 写进 basedir 的 `selfplay/`，由正在跑的 `shuffle` 收走。

#### 共享盘方案（不用 HTTP）

多台都能读写同一个 `dist_run`（SMB/NFS）时，客户机也可跑 `selfplay --basedir 映射路径`，不必 `serve`/`contribute`。仍应**只有一个** `shuffle` 和一个 `train`。Windows 网络盘上多进程写 `.npz` 偶发锁文件，出问题就改回 HTTP 客户。

#### 和官方 KataGo 的对应

| 官方 | 本仓库 |
|---|---|
| 自对弈进程写 `selfplay/` | `gkt_dist.py selfplay` 或 `contribute` |
| shuffle 窗口 | `gkt_dist.py shuffle`（只混最近 selfplay；丢掉更旧的 raw / shuffled 包） |
| train + export | `gkt_dist.py train`（写出 `.pt`/`.npz`，无单独 TF export） |
| gatekeeper | `gkt_dist.py gate`（Arena；可选） |
| `katago contribute` | `gkt_dist.py contribute` |

搜索仍是每进程自己的 `predict_batch`，没有参数服务器。checkpoint 用现有 `load_net` / `load_cpu_net`。各 loop 可 `--poll` 控制空转间隔。

---

## 8. Web UI

`python web_ui/server.py`：标准库 HTTP，单线程，状态在 `STATE`。  
选手 2/3/4 人，每人真人或模型。下拉只列出与本局人数一致的 checkpoint（标签 kP）；对不上拒挂。无匹配权则该席随机（均匀合法着；图围棋含虚手，五子棋 / 反五子棋不含），sim 禁用。模型席 MCTS 恰好 `sim` 次，根 Dirichlet 与自对弈相同（\(f=0.25\)），并做随机顶点重编号（GNN 同步换邻接），走子仍为访问 `argmax`。若存在 `models/elo.json`，下拉可显示 Elo（展示用）。2DCNN 仅网格图。终局画布展示 **finalize 填空之前** 的占领；图围棋分数栏在 `finalize()` 之上另加对局贴目（三人默认约 7.5/4.5/0，四人约 7.5/4.5/2.5/0，末位 0）。规则下拉：图围棋 / 五子棋 / 反五子棋（后两者两人、无虚手条）。界面配色图围棋偏红、五子棋偏绿、反五子棋偏蓝紫。

布局：矩形网格；图 3 / 5.5 可拖动 wrap（按屏幕铺格点、占用周期取模、未平移主区外虚化，同一套路）；图 5 三角菱形；图 6 为 19³ 可拖旋转立方体（空交叉点不画圆点；点外侧后只亮三平面；蓝圈为选点；Pass 下 Confirm / Cancel；棋盘下可改 x/y/z，W/S·Q/E·A/D 微调）；图 7 路图画成 M；图 4 / 4.5 罗宾逊投影（可拖旋转）；无 LAYOUT 的随机图用力导向。图 1 / 5 / 5.5 / 7 默认横放。图 2（61×61）可下：默认训练不含此图，给 2DCNN / GNN 做训练后网格泛化检查。回放时 AI 着可叠该手 MCTS 访问占比与落子前 ownership（人着无网分析）。

### 8.1 随机图上的讲解、对话与 drone 指挥（规划）

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

**未做：** 无对话 API、无命令解析、无区域约束搜索；future / opp / belief 不进画布。回放已能叠 AI 着的访问占比与 ownership（另一次 `forward`，不进 MCTS）。`predict_batch` 给搜索回 policy + value + stdev。在第一个有显著棋力的模型出世前不组装讲解 / 对话 / drone。规划顺序仍是「完整 forward → 张量摘要 → 带工具的对话」，再接 drone 掩码。

---

## 9. 技巧亮点（实现层，不是规则）

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
12. **图级可复现 shuffle + `summary.json` resume**：中断后续跑下一张图。每图另有 `progress/summary_{key}.json` 累积各 round。
13. **鸭子类型网络**：MCTS 不知道背后是 GNN、2DCNN、MLP 还是 1DCNN。
14. **掩码用有限 \(-10^9\)**：避免 `0 * log_softmax(-inf)` 变 NaN。
15. **虚拟损失语义**：inflight 深度为一个 batch；Python 异步评估器无法保证每节点虚拟损失只计一次。
16. **连续分数差做 Arena / Elo 信号**：对局不记 0/1 胜负，而记归一化子数差 `score_lead`（value 头同一目标）。先手优势 δ 是加性项，交替先后手后精确抵消（`E[lead]=棋力差/n`），强弱比较不依赖贴目；0/1 胜负在 δ 巨大的图（线图/乱图）上会因「谁先谁赢」截断棋力差。
17. **大快照面板 Arena**：对手不止 best，而是 best + 上一轮小快照 + 最近 N 个大快照——短程抓「单轮崩」、长程抓「缓慢漂移」；默认 `--arena-lead-threshold 0.0`（不退步即接受）。

---

## 10. 内置图（实现元数据）

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

## 11. 检查点

| 产物 | 含义 |
|---|---|
| `new.pt` / `new.npz` | 最近完成的一张图之后的共享权（`--resume` 用这个） |
| `roundN.pt` / `roundN.npz` | 整轮结束；默认只保留最近 `--model-snapshot-rounds`（5）轮，更早的自动清理（`new.*` / `best.*` 不动） |
| `bigN.pt` / `bigN.npz` | 长程大快照：每 `--big-snapshot-interval`（默认 10）轮存一份，默认只留最近 `--big-snapshot-rounds`（5）个；兼作 Arena 面板的长程对手 |
| `best.pt` / `best.npz` | Arena 接受后的最佳 |
| `elo.json` | 共享 Elo 榜（`gkt_elo.py` 写出，UI 只读） |
| `distill_meta.json` | 蒸馏产物旁的小侧记（`distill.py` 写出）：数据源、图 key、epoch、样本数、lr |
| `summary.json` | 跨图、跨 round 的 ploss / vloss / oloss，供曲线与 resume |
| `progress/summary_{key}.json` | 该图所有 round 的统计列表（每训完此图追加一行） |
| `replay/{key}/unused.npz` | 该图尚未用于 SGD 的样本；下次再训此图会再加载，直到被抽中或超过 `buffer_capacity` |
| `replay/{key}/snapshots/round%06d.npz` | 该图每 round 的 buffer 快照（回滚点），默认保留最近 `--buffer-snapshot-rounds`（5）轮 |
| `progress.wN.txt` | `outdir`：每个 worker 一份自对弈细粒度心跳（`progress.w0.txt` …）；每张图 cycle 开始时清空。CPU/GPU 均为 worker 下标 `0..workers-1`，每 worker 打 `gpw` 局，逐局 `game_start` / 每步 `move` / 每 MCTS batch `batch`（约 1s 节流）/ `game_end` |
| `progress.txt` | 自对弈 `outdir`：**粗粒度**心跳，只打每局 `start` / `end`（不打 move/batch），与 `progress.wN.txt` 同时写，供快速看进度不看细节。蒸馏 `base/<arch>/progress.txt`：逐步心跳，每个 epoch 开始时清空，逐 train/val step 打 `pl/vl/ol` |
| `progress.arena.txt` | `outdir`：**Arena 细粒度**心跳，每轮门控开始时清空；逐对手×逐图×逐局打 `game_start`/`move`/`batch`/`game_end`（复用自对弈同一套 `make_move_heartbeat`，`move`/`batch` 经 `native.search` 新增的 `on_batch` 回调透出） |

加载 GPU 权：`load_net`（`net_type`、`num_players`、`n_features` 等；`label` 若在则须与类型一致）。CPU：`load_cpu_net`（`_net_type`、`_num_players`）。缺键、非有限权、或 \(F\neq\texttt{feature_dim}(k)\) 拒载。

---

## 12. 五子棋 / 反五子棋引擎

规则与实验主张：[`gomoku.md`](gomoku.md)（2DCNN 完备性对照图围棋的 GNN 完备性）。C++ `Game(..., rules="gomoku"|"antigomoku", win_length=5)`，必须带 `.grid` 且 \(R,C\ge 2\)、\(R\cdot C=n\)。合法着是空点；`play(n)` 非法，`apply_pass` 恒等。胜负扫描坐标四向连珠，不读 CSR。反五子棋把成线方判负（`winner` 为对方）。MCTS / 自对弈同一套，价值 \(\{+1,0,-1\}\)，手数上限 \(n\)。轮训图 `GOMOKU_TRAIN_KEYS`；入口 `train_gomoku_gnn.bat` / `train_antigomoku_gnn.bat` 等（经 `start_train.bat` 加 `--rules`），默认目录 `cur_mod_gomoku_*` / `cur_mod_antigomoku_*`。图围棋默认图列表排除 `G*`。

