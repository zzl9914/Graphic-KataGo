# Graphic-KataGo · 训练方法

本文件讲 **蒸馏、分阶段自对弈与分布式**：怎么产第一个有棋力的网、怎么过渡、怎么跨图续训、怎么多机加速。  
模型 / MCTS / 多头定义见 [`algorithm.md`](algorithm.md)。规则见 [`rules.md`](rules.md)、[`gomoku.md`](gomoku.md)。

命题仍是 **图无关性与权重可迁移**：同一套权能否不依赖棋盘拓扑、在任意有向图上维持棋力。训练是分阶段把这个命题做出来，不是从随机权硬冲。

---

## 0. 分阶段总览

围棋主线（四种架构各有一套 bat）：

| 阶段 | 入口 | 产物 | 做什么 |
|---|---|---|---|
| ① 蒸馏（基础培养） | `base/distill_{gnn,mlp,cnn1d,cnn2d}.bat` | `base/<arch>/new.pt` 或 `new.npz` | KataGo 标签监督；三阶段 10+10+10 epoch |
| ② 基础培养2 | `base/cultivate2_*.bat` | `base/cultivate2/<arch>/` | 只训图 `0`，20 轮后停；前 10 轮冻 policy 头 |
| ③ 正式跨图 | `starter/train_*.bat` | `cur_mod_*` | 默认全图、`--infinite`、Arena 开 |

①②③ 共用 13 元组样本、F=8 特征、同一 checkpoint 格式。后一阶段 `--resume` 前一阶段的 `new.pt` / `new.npz`。

**不要**把②的「只图 0 / 冻政策头 / 20 轮停 / 关 Arena」写进③。  
**不要**用 3000/25 去乘 value/own MSE：那是把蒸馏阶段 2/3 的头部 lr 倍率误叠进联训，`clip_grad_norm=1` 下会把 policy 梯度挤没。现行权重与蒸馏联合阶段相同：**abs 0.08 / rto 5 / cons 0.01 / own 5**。

五子棋 / 反五子棋不走 KataGo 蒸馏，直接 `starter_gomoku/` / `starter_anti_gomoku/`（见 §10）。

---

## 1. 环境

先编译 native：`python cpp/build.py`（C++17 + `pip install pybind11`；Windows 走 setuptools/MSVC）。**pip 一律走清华源**（`start_ui.bat` 已写死）：

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

Windows 入口钉死 **Python 3.14**（缺则 PATH 上的 `python`，须为 3.14 非 free-threading）。不要用 `py -3` 或 3.13：扩展是 `gkt_native.cp314-win_amd64.pyd`。编译 native：`python cpp/_build_with_sdk.py`，再 `python cpp/_deploy_pyd.py`（见 [`algorithm.md`](algorithm.md) §9）。

```bash
python cpp/build.py
cd scr
python distill.py --data ../distill_data/m2_19x19.jsonl
# 然后：base/cultivate2_gnn.bat
# 然后：starter/train_gnn.bat
python web_ui/server.py --device cpu
```

---

## 2. 自对弈样本

一律 13 元组。缺 `value_rto` 键的 npz 直接拒载。

`(features, legal_mask, policy, me, value_abs, ownership, q, opp_policy, opp_weight, future, lead, weight, value_rto)`

- `value_abs`：图围棋是终局子数差 \(z\)（二人 \(\mathrm{我}-\mathrm{对方}\)，**不** `/n`、**不** mix）。五子棋 / 反五子棋与 `value_rto` 相同（已经是 \([-1,1]\) 的 mix）
- `value_rto`：mix \(\lambda\cdot\mathrm{lead}+(1-\lambda)\cdot Q\)，lead 是不含贴目的 \((k\cdot s_{\mathrm{me}}-S)/((k-1)n)\)（二人即 \((\text{我}-\text{对方})/n\)）。搜索 / JIT `out["value"]` 学这个
- `ownership[i]\in[-1,1]`：终局该点从 `me` 视角的归属
- `q`：该步根节点 MCTS Q（与 lead 同量纲），给 stdev 头
- `opp_policy`：下一手（下一家）的访问分布；最后一步 `opp_weight=0`
- `future`：约 4 手后的占子图（C++ `FUTURE_PLIES`；+1 我 / −1 对方 / 0 空）
- `lead`：终局 \(z/n\)，给 score-belief，不随 mix 搅浑
- `weight`：policy surprise \(\mathrm{KL}(\pi_{\mathrm{MCTS}}\|\pi_{\mathrm{nn}})\) 映射到 \([0.5,8]\)，SGD 按样本加权
- GPU 损失：`(policy, value, own, aux, total)`，其中报告的 `value` 是 **abs** MSE；`total` 含 `vw*abs + vr*rto + wc*cons + ow*own +` 加权辅助项（cons 进 `aux`）
- CPU `backward` 返回四元组，报告的 value 同样是 abs MSE，`total` 含辅助损失与 cons

两个头用很低的一致性项 \((\mathrm{abs}-n\cdot\mathrm{rto})^2\)（`--value-cons-weight` 默认 0.01；五子棋 / 反五子棋是 \((\mathrm{abs}-\mathrm{rto})^2\)）。这只是软耦合，abs 仍可以跟 mix 不完全一致。abs 给 SGD 一个不被小图 `/n` 放大的监督；rto 保持 MCTS 的 \([-1,1]\) 标定。

### 2.1 价值语义（KataGo 标签 vs 学生网）

**同一类走子方领先、无贴目；两个头两个尺度。**

| | 蒸馏教师 | 自对弈学生 |
|---|---|---|
| value_abs | KataGo `scoreLead` 子数（查询 `komi: 0`、`tromp-taylor`，`SIDETOMOVE`） | 图围棋：`finalize` 后的 MC 子数差（无 Q mix）。k-in-a-row：与 rto 同一 mix |
| value_rto | `scoreLead / n`，clip 到 \([-1,1]\) | mix \(\lambda z/n+(1-\lambda)Q\) |
| 噪声 | 很小 | 大（学生棋 + 随机撒子开局 + 截断） |

搜索头是 `tanh`，学 rto 尺度。监督头是无界线性求和，学子数。差不只是估计方式（期望 vs 实现），单位也不同。自对弈 rto 固定 mix，`--q-lambda` 默认 0.5。不要把 \(\lambda\) 设成 0：rto 若还贴 0，纯 Q 会自举锁死。

---

## 3. 蒸馏（基础培养）

`scr/distill.py` 把 KataGo 强标签监督拟合到图无关网络，产出第一个有棋力的 checkpoint（GPU `new.pt`、CPU `new.npz`）。`--net`：`gnn`（图无关主线）、`2dcnn`（网格对照）、`mlp` / `1dcnn`（消融）。

- **三阶段**（各 `--epochs` 轮，默认 10+10+10）：① 联合 policy + **两个 value 头** + own，**不冻**任何参数，Adam \(lr=1\mathrm{e}{-4}\)，loss = `pl + 0.08*abs + 5*rto + 0.01*cons + 5*ol`（报告的 pl / vl / ol 仍未加权，**vl = abs MSE**；cons 进 aloss / total）；② 只训 own 头（冻 trunk，Adam lr ×25，unweighted MSE）；③ 只训 **两个 value 头**（冻 trunk，Adam lr ×100，仍用 `vw`/`vr`/`wc`）。GPU 用 Adam（β=0.9/0.999）；CPU 蒸馏同样用 NumPy Adam。辅助头（opp / soft / belief / stdev / future）蒸馏阶段无教师信号，保持初始化，到自对弈再训。
- **特征逐位一致**：`build_sample` 用 native `extract_features`，与自对弈同源；abs 用石头差、rto 用 `/n`。
- **训练增强、验证集固定**：训练集每个 batch 按网络分派与自对弈 SGD 相同的置换（见 [`algorithm.md`](algorithm.md) §8.5）。**课程打乱**：默认 `--aug-from-epoch 4`，全局 epoch 1–3 用原编号，从第 4 轮起才打乱（`1` = 第一轮就打乱，`0` = 全程不打乱）。**验证集**按**局**切：`--val-frac`（默认 5%）是局数比例，用 `--seed` 打乱局号后整局划入 val，不做增强、不重采样。现有无 `game_id` 的 JSONL 用空盘当局界；新生成的记录带 `game_id`。心跳行带 `aug=on/off`。
- **逐步心跳**：`outdir/progress.txt`，每个 epoch 开始时清空；逐 train/val step 打 `pl/vl/ol`，行首带 `stage=policy|own|value e{k}/10 g{G}/30`。
- **数据流**：`distill_data/gen_m2_19x19.bat` 跑 `gen_katago_data.py`（`katago/katago.exe analysis`，**300 局、`--visits 350`**）写出 `distill_data/m2_19x19.jsonl` → `distill.py`。坐标转换在 `distill_katago.py`。终局截断见 §3.1。蒸馏启动器默认读该 JSONL。中途停掉再跑同一 bat：`--resume` 从最高 `round*` 续。写出 `new.pt` / `new.npz` 之后会删掉 epoch 快照；再跑 `--resume` 发现已有 `new`、没有 `round*`，直接保留终态、不重蒸。
- **内存**：19×19 稠密邻接在 6 GB 卡上 batch 上限约 **16**（32 OOM）。`distill.py` 对 gnn/2dcnn 默认 `--batch-size 16`，mlp/1dcnn 默认 64。`--hidden 512 --n-blocks 20` 与自对弈默认一致。CPU 1DCNN 蒸馏 mini-batch 默认 64。
- **启动器**只传 `--net`（gnn 可省）和 `--data`；其余（`--resume`、`--outdir ../base/<arch>`、`--graph-key 0`、`--value-weight 0.08 --value-rto-weight 5 --value-cons-weight 0.01 --own-weight 5`、`--aug-from-epoch 4`、GPU `--device auto`→cuda）是代码默认。

蒸馏打破 value 塌缩死锁；接下来接基础培养2，再接正式跨图。不要把刚蒸完的网直接丢进 3000/25。

### 3.1 生成截断（图围棋终局 ≠ KataGo 终局）

教师是 KataGo `analysis`（查询里 `rules: tromp-taylor`），但 JSONL 的 `board` 必须是 GKT 引擎上的合法续谱。两边**终局条件不同**，规则对照见 [`rules.md`](rules.md) §5.1。

KataGo / Tromp–Taylor：棋盘上**连续两手 ply 都是虚手** → 整盘结束。  
图围棋：某方**自己的连续两手**都虚手 → 该方出局；二人时一人出局即整盘结束。中间可以夹着对方落子。

`scr/gen_katago_data.py` 因此按 GKT 截断，不能只用「连两手 ply 虚手」停局：

1. 每步先 `Game.play`，只有 `legal` 才把该手写入 KataGo `moves`。
2. `game.game_over()`（出局或协议结束）为真 → **停止本局**，进入下一局。
3. **已经写入的局面保留**（写入时两边仍同步）。不把整局从文件里删掉，也**不再写终局之后**的盘面。
4. KataGo 采样着 GKT 判非法：该手不进 `moves`，同样停本局、开下一局。
5. 「连两手 ply 都是 pass」仍可作额外停条件（与教师规则一致），但是附加项，不是唯一停条件。
6. KataGo `stderr` 由后台线程排空。只读 `stdout`、把 `stderr` 留在 `PIPE` 里会在 Windows 约 4 KiB 管道缓冲满后把分析进程卡死。

若忽略 1–2：GKT 已终局后仍向 KataGo 要填空政策，拒子后 GKT 盘面冻结而教师 `moves` 继续，写出同盘同 `to_move` 的脏标签。旧 `distill_data/m2_19x19.jsonl` 因此已删除；重蒸前须重新生成。

---

## 4. 基础培养2

插在蒸馏和正式跨图之间，**不是**正式训练。启动器 `base/cultivate2_{gnn,mlp,cnn1d,cnn2d}.bat`，产物 `base/cultivate2/<arch>/`。

| 项 | 设定 |
|---|---|
| 图 | 只训 `0`（19×19） |
| 轮次 | `--rounds 20`，无 `--infinite`，然后停止 |
| 权重 | 默认 `--value-weight 0.08 --value-rto-weight 5 --value-cons-weight 0.01 --own-weight 5` |
| 价值目标 | abs = MC 子数差；rto = mix（`--q-lambda 0.5`） |
| Arena | **关**（`--no-arena`）。此阶段要让 value/own 在学生自对弈上动起来；回滚会丢掉刚学到的 value |
| replay | `--buffer-drop-from-round 21`（20 轮内不丢） |
| 快照 | `--model-snapshot-rounds 25` |
| 其余 | 与正式跨图同一套训练器默认（`--sim 256 --gpw 32 --steps 16 --lr 1e-4 --temperature 0.1`） |

日程：

- 轮 1–10：`--freeze-policy-until-round 10`。冻结已脱离随机的 **policy 读出**（GNN `policy_head`、2DCNN `policy_conv`、CPU `Wp`/`bp`/`W_pass`/`b_pass`）。读出权重不动；policy CE **仍回传到 trunk**。训 trunk + value + own + aux。
- 轮 11–20：解冻 policy，仍只训图 0。

CLI：`gkt_train_gpu.py` / `gkt_train_cpu.py` 的 `--freeze-policy-until-round N`（默认 0 = 不冻）。GPU 每张图新建 `GktTrainer` 时按当前轮决定；CPU 在每轮开头设 `net.freeze_policy`。

**续训**：再跑同一个 bat。已有 `cultivate2/<arch>/new.pt|.npz` 则从它续，并用该目录 `summary.json` 接到下一轮；否则 `--resume` 蒸馏终态。没有蒸馏产物则报错，不去随机初始化。中途 Ctrl+C 停在某一轮打到一半时，从上一轮已存盘之后重来。20 轮已经跑完再点一次会多进第 21 轮再因 `--rounds 20` 停。

Windows `.bat` 里 `if (...)` 块中的 `echo` **不能写裸括号**（会被当成语法）；汉字若用 UTF-8 无 BOM，在默认 GBK 的 `cmd` 里会乱码，不影响 Python 训练命令。

---

## 5. 正式跨图

入口 `starter/train_*.bat` → `start_train.bat` → `gkt_train_gpu.py` / `gkt_train_cpu.py`，目录 `cur_mod_*`。启动器只选架构；`start_train.bat` 补 `--net`、`--infinite` 和已有 checkpoint 的 `--resume`。

| 项 | 设定（训练器默认 = 围棋正式线） |
|---|---|
| 图 | 默认训练图（排除 oversized `2`/`6` 与五子棋 `G*`） |
| 轮次 | `--infinite`（`start_train.bat` 传入） |
| 权重 | `--value-weight 0.08 --value-rto-weight 5 --value-cons-weight 0.01 --own-weight 5` |
| 价值目标 | abs = MC 子数差；rto = mix（`--q-lambda 0.5`） |
| Arena | **开**（默认） |
| replay | `--buffer-drop-from-round 5` |
| 温度 | `--temperature 0.1` |
| 其余 | `--sim 256 --workers 1 --gpw 32 --steps 16 --lr 1e-4`；GPU `--device cuda --selfplay-device cuda` |

有 `cur_mod_*/new.pt` 则续训。要从培养2 接过来，把 `base/cultivate2/<arch>/new.pt` 拷进 `cur_mod_*` 或显式 `--resume`。

### 5.1 随机让子开局

仅**自对弈训练**（`make_training_game`）。Arena / Web / 评估从空盘开始。

每局先抽 \(p\sim\mathcal{U}(0,\,p_{\max})\)，每个顶点独立以概率 \(p\) 放子（颜色在 \(1..k\) 上均匀），否则为空。图围棋 \(p_{\max}=0.25\)。五子棋 \(p_{\max}=0.12\)。然后用规则引擎对静态盘提子：从末座到首座依次提无气块。子数最少的一方先行。Zobrist 按整盘占领重算，历史为空。

### 5.2 跨图循环、课程、Arena、replay

- 排除超大图 `2`、`6`。二者都不进默认训练，但 UI 都可下。
- 每轮图顺序走 `gkt.shuffle_graph_keys`：`random.Random(int(rnd)).shuffle`（GPU / CPU 同一公式、私有 RNG）。这是为了跨图访问均匀，不是为了复现对局；自对弈 / SGD / 顶点置换的不可复现是设计（见 [`algorithm.md`](algorithm.md) §8.6）。`--resume` 读 `summary.json` 末行，**cycle 序号接续**，用同一公式切到下一张图（key 按 `str` 比对）。**若末行恰好是该轮最后一张图且 Arena 开启**，先补跑这一次门控（图列表置空；buffer 快照已在该轮第一次写过），再进下一轮；`best_weights` 从 `best.*` 恢复（无则回退当前权重）。`--no-arena` 续训则直接进下一轮，不补门控。图集合对不上时打日志并进下一 round。
- 课程（**默认关闭**，`--curriculum-rounds 0` = 全程全长）。从 0 时显式传 `--min-moves 40 --curriculum-rounds 20000` 可重启：名义手数上限从 `--min-moves` 涨到 \(n\cdot\mathrm{factor}+\)min-moves。自对弈每局再对数均匀抽 \([\mathrm{cap}/2,\,2\cdot\mathrm{cap}]\)。停表后 `finalize` 提供中局领地监督。图围棋自对弈在前 \(\texttt{forbid_pass_after}=\min(\mathrm{cap},\,\max(\lfloor n/8\rfloor,16))\) 手若仍有合法落子则 **MCTS 去掉虚手**（规则层虚手仍合法；UI / Arena / Elo 不禁）。
- 每图结束（CUDA）：`gc.collect()` + `empty_cache()`。
- 自对弈：同一套权打满 \(k\) 个座位。
- Arena（**默认开**，`--no-arena` 关）：新 vs 面板跨图门槛，**据此接受或回滚**；固定 `--arena-sim`，根 Dirichlet \(f=0\)（纯访问 argmax，与自对弈 \(0.25\) 不同）。面板 = best-so-far + 上一轮**已接受**的小快照 `round{rnd-1}` + 最近 `--big-snapshot-rounds` 个大快照 `big*`。新网轮流占一座、对手占其余；每局记连续归一化 `score_lead`，交替先后手使先手优势 δ 作为加性项抵消，再与 `--arena-lead-threshold`（默认 0.0）比。**0 局不接受、不回滚**。接受后才写 `roundN` / `bigN`；拒绝则权回 `best`，把本轮误留下的 `roundN`/`bigN` 挪到 `rejected/`，`unused.npz` 从上一份 buffer 快照恢复。带心跳：`arena: round N gate starting …`、逐「图×对手」、结束汇总。
- `--players {2,3,4}`：自对弈人数与 \(F=\texttt{feature_dim}(k)\) 一致；2P/3P/4P 分网。
- `--lr` 默认 \(1\mathrm{e}{-4}\)。非有限 loss / 梯度跳过该次 SGD；拒绝把非有限权写入 `new.pt`。
- **每图 unused 队列**：`outdir/replay/{key}/unused.npz`。SGD 成功用过的样本从 buffer 删除、不写回。未用样本下次再训同一张图时仍会加载。条数上限 `buffer_capacity`（默认 20 万）。图与图不混。`--replay-rounds 1` 不读盘上队列。从 round `--buffer-drop-from-round`（围棋默认 5；五子棋 / 反五子棋未传时填 11）起，每训完一张图按 `--buffer-drop` 丢掉最旧样本（默认 auto = 本轮未使用量 `new - consumed`）。
- Elo（`gkt_elo.py`）：四种架构 + 均匀随机共用一张榜，**只供展示**，不替代 Arena。

```bash
cd scr
python gkt_elo.py --models-dir ../models --sim 100 --games 2 --device cpu
```

---

## 6. 从随机权自对弈（规划）

**组装门槛：** 不蒸馏、随机初始化的自对弈，在**当前主线（蒸馏 → 基础培养2 → 正式跨图）被显著棋力跑通之前不考虑组装**。在那之前本节只作规划：不设启动器、不设里程碑、不当对照实验来跑。

命题上，「同一套图无关权从均匀随机自己下出来」是理论完备形态。实现上单卡吞吐距 AlphaZero 量级差很远，value 头还会塌成常数——主线先蒸馏，正是因为这条路现在走不通。

**未做：** 无专用 bat、无对照实验目录、不维护一套与主线脱节的超参。`gkt_train_*` 仍允许不 `--resume` 随机开训，那只是训练器还能随机初始化，不是一条要跑的线。

---

## 7. 损失与价值目标（现行）

| 旗标 | 蒸馏① | 基础培养2 | 正式跨图 | 五子棋 / 反五子棋（未传则填） |
|---|---|---|---|---|
| `--value-weight` | 0.08 | 0.08 | 0.08 | 1 |
| `--value-rto-weight` | 5 | 5 | 5 | 1 |
| `--value-cons-weight` | 0.01 | 0.01 | 0.01 | 0.01 |
| `--own-weight` | 5 | 5 | 5 | 1 |
| abs 目标 | 教师 `scoreLead`（子数） | MC 子数差 | MC 子数差 | 与 rto 同一 mix |
| rto 目标 | 教师 `scoreLead/n` | mix | mix | mix |
| `--q-lambda` | — | 0.5（\(z/n\) 的权重） | 0.5 | 0.5 |

报告的 pl / vl / ol **始终未加权**；vl 是 **abs** MSE；cons 进 aloss。不要用 3000/25。一致性项权重保持很低（默认 0.01），不是硬约束。

---

## 8. 分布式

官方 KataGo 是异步进程靠共享目录衔接：selfplay → shuffle → train → 可选 gatekeeper。本仓库对应 `scr/gkt_dist.py`。启动步骤只写在本节，不写进 [`algorithm.md`](algorithm.md)。

### 8.1 与本机训练共用同一套权（设计，不是事故）

本机 `gkt_train_*` / `start_train.bat`（`cur_mod_*`）和分布式 `gkt_dist` **故意用同一套模型**（同一 `net_type`、同一人数、同一规则、同一 checkpoint 格式）。多机自对弈给同一条权产样本，吞吐比单机循环快。`init --from ../cur_mod_*/new.pt` 把本机权灌进 `dist_run`；门控接受后的 `models/` 也可以拷回 `cur_mod_*` 再 `--resume`。

**更新不会打架：** 本机跨图训练（`gkt_train_gpu.py` 的自对弈 + SGD）和把**本机当教练机**的分布式训练（`gkt_dist.py train`，以及可选 `gate`）**互斥**——两边都要占 GPU，同一张卡上不会同时写权。教练机典型组合是 `shuffle` + `train` + `serve`（`serve` 只吃 CPU）；其它机器只跑 `contribute` 做自对弈。不要在同一 GPU 上并行开 `gkt_train_*` 和 `gkt_dist.py train`。

同一 basedir 仍只对应**一种**网、**一种**人数、**一种规则**。2P GNN 和 3P、GNN 和 2DCNN、或图围棋和五子棋，不要混在一个 `dist_run` 里——那是架构混用，不是「本机/分布式共用权」。

只在本机接着训、暂时不开客户机：用 `gkt_train_*` 即可。分布式 basedir 默认 `../dist_run`，和 `cur_mod_*` 分目录，权通过拷贝 / `--from` 对齐。

### 8.2 启动

Windows 入口与 `start_train.bat` / `start_ui.bat` 并列：教练机 `start_dist_main.bat`，客户机 `start_dist_cont.bat`（同样钉死 Python 3.14）。不要和本机 `start_train.bat` 抢同一张 GPU。

| 入口 | 做什么 |
|---|---|
| `start_dist_main.bat` | 可选首参 `gnn` / `mlp` / `cnn1d` / `cnn2d`（默认 gnn），其后是 `init` 旗标。无 `dist_run/run.json` 时 `init`（种子优先 `cur_mod_*`，围棋可回退培养2 / 蒸馏）；然后各开一个窗口跑 `shuffle`、`train`、`serve --host 0.0.0.0 --port 8877` |
| `start_dist_cont.bat` | 参数 `url`、`token`、`cuda` 或 `cpu`。缺参则读 `GKT_DIST_URL` / `GKT_DIST_TOKEN` / `GKT_DIST_DEVICE` 或提示。设备默认 **cuda**（Python 子命令默认是 cpu） |

`init` 的 HTTP token：环境变量 `GKT_DIST_TOKEN`，否则提示输入，至少 16 个字符；明文不写进 `run.json`。可选：`GKT_DIST_BASEDIR`（相对仓库根，默认 `dist_run`）、`GKT_DIST_HOST` / `GKT_DIST_PORT`、`GKT_DIST_GATE=1`（再开 `gate` 窗口；须在首次 init 带 `--gate`）、客户机 `GKT_DIST_INSECURE=1`（自签名 HTTPS）。

**多机：** 只有一台当教练机写权，其余当自对弈客户。客户不必挂共享盘，走 HTTP（`serve` + `contribute`）。`selfplay` / `shuffle` / `train` / `gate` 必须看见同一个 `--basedir`。

| 机器 | 跑什么 |
|---|---|
| 教练机（建议最强 GPU） | `start_dist_main.bat`（`init` 一次；长期 `shuffle`、`train`、`serve`；可选 `gate`） |
| 教练机上也可再开 | 仅当 GPU 空闲时 `selfplay --device cuda`（与 `train` 互斥） |
| 其他电脑 | `start_dist_cont.bat http://教练机LAN_IP:8877` |

手工等价命令（非 Windows 或要拆开跑时）如下。五子棋种子必须是 2P；默认图列表为 `GOMOKU_TRAIN_KEYS`；手数上限固定为 \(n\)：

```bash
cd scr
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gomoku_gnn/new.pt --net gnn --rules gomoku --token 自己设至少16位密文
```

等价于 `start_dist_main.bat gnn --rules gomoku`。图围棋：

```bash
cd scr
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gnn/new.pt --net gnn --token 自己设至少16位密文
python gkt_dist.py shuffle --basedir ../dist_run
python gkt_dist.py train --basedir ../dist_run --device cuda
python gkt_dist.py serve --basedir ../dist_run --host 0.0.0.0 --port 8877
```

等价于 `start_dist_main.bat`（默认 gnn）。`serve` 的 token 默认读 `run.json` 里 init 写下的哈希，不必再传明文。局域网客户必须 `--host 0.0.0.0`（Python 默认只绑 `127.0.0.1`；bat 绑 `0.0.0.0`）。Windows 防火墙放行 8877。

`init --gate` 才启用门控，此时必须另开 `gate`，训练写入 `modelstobetested/`。默认不加 `--gate`：`train` 直接写入 `models/`。

客户机：

```bash
cd scr
python gkt_dist.py contribute --url http://教练机LAN_IP:8877 --token 与init相同 --device cuda
```

等价于 `start_dist_cont.bat http://教练机LAN_IP:8877`。

无 GPU 则 `--device cpu`。循环：`GET /api/task` → 必要时 `GET /api/model/…` → 下一局 → `POST /api/games`。客户机**不能**上传权重，只能上传自对弈 `.npz`。

多台都能读写同一个 `dist_run`（SMB/NFS）时，客户机也可跑 `selfplay --basedir 映射路径`，不必 HTTP。仍应只有一个 `shuffle` 和一个 `train`。Windows 网络盘上多进程写 `.npz` 偶发锁文件，出问题就改回 HTTP。

目录（`--basedir`）：

| 目录 | 角色 |
|---|---|
| `run.json` | 图列表、sim、窗口、是否门控、`token_sha256`（无明文 token） |
| `models/` | 已接受网络（仅教练机 `train` / `gate` / `init` 写入） |
| `selfplay/` | 自对弈 13 元组 `.npz`；shuffle 只保留每图最近 `window_files` 个 |
| `shuffled/` | 按图滑窗打乱后的训练包 |
| `modelstobetested/` | 训练刚写出、待门控（仅 `init --gate`） |
| `rejectedmodels/` | 门控未过 |
| `client_cache/` | 服务器本机缓存；客户机用 `contribute --cache` |

| 官方 KataGo | 本仓库 |
|---|---|
| 自对弈进程写 `selfplay/` | `gkt_dist.py selfplay` 或 `contribute` |
| shuffle 窗口 | `gkt_dist.py shuffle` |
| train + export | `gkt_dist.py train`（写出 `.pt`/`.npz`） |
| gatekeeper | `gkt_dist.py gate`（Arena；可选） |
| `katago contribute` | `gkt_dist.py contribute` |

搜索仍是每进程自己的 `predict_batch`，没有参数服务器。

### 8.3 HTTP 安全

`serve` / `contribute` **fail-closed**：没有 token 拒开。token 至少 16 个字符；`init` 只把 SHA-256 写入 `run.json`，明文不落盘。比对走 `hmac.compare_digest`（先哈希再比，长度固定）。`/api/status` 不回 token 字段。

客户上传：必须带 `Content-Length`，默认上限 64 MiB（`--max-upload-mb`），样本条数上限 20000；`_graph` 必须在本次 run 的图列表里；文件名只允许 `[A-Za-z0-9._-]`，写入前做目录约束，防止 `../` 写进 `models/`。下载权重只从 `models/` 白名单扩展名（`.pt` / `.pth` / `.npz`）。`latest_model` 仍按 mtime，但客户写不进 `models/`，不能靠改时间戳冒充 accepted 网。

默认绑定 `127.0.0.1`。绑 `0.0.0.0` 时打警告：无 TLS 则 token 是唯一认证，只给局域网或 SSH/VPN 隧道，**不要暴露公网**。可选 `--tls-cert` / `--tls-key`（TLS 1.2+）；自签名证书时客户加 `--insecure`（仅 HTTPS）。

旧 `run.json` 若仍有明文 `"token"`，`serve` 会现算哈希，不把明文回给 API。新 init 不再写该字段。

---

## 9. 检查点

| 产物 | 含义 |
|---|---|
| `new.pt` / `new.npz` | 最近完成的一张图之后的共享权（`--resume` 用这个） |
| `roundN.pt` / `roundN.npz` | 整轮**接受后**（或 `--no-arena`）的权；默认只保留最近 `--model-snapshot-rounds`（5）轮（`new.*` / `best.*` 不动） |
| `bigN.pt` / `bigN.npz` | 长程大快照：每 `--big-snapshot-interval`（默认 10）轮、且 Arena 接受（或关 Arena）时存一份，默认只留最近 `--big-snapshot-rounds`（5）个；兼作 Arena 长程对手 |
| `rejected/` | Arena 拒绝后从工作目录挪走的 `roundN` / `bigN`，不进下一轮面板 |
| `best.pt` / `best.npz` | Arena 接受后的最佳 |
| `elo.json` | 共享 Elo 榜（`gkt_elo.py` 写出，UI 只读） |
| `distill_meta.json` | 蒸馏产物旁的小侧记：数据源、图 key、epoch、样本数、lr |
| `summary.json` | 跨图、跨 round 的 ploss / vloss / oloss，供曲线与 resume |
| `progress/summary_{key}.json` | 该图所有 round 的统计列表 |
| `replay/{key}/unused.npz` | 该图尚未用于 SGD 的样本 |
| `replay/{key}/snapshots/round%06d.npz` | 该图每 round 的 buffer 快照，默认保留最近 `--buffer-snapshot-rounds`（5）轮 |
| `progress.wN.txt` | 每个 worker 一份自对弈细粒度心跳；每张图 cycle 开始时清空 |
| `progress.txt` | 自对弈粗粒度心跳（每局 start/end）。蒸馏目录下是逐步 `pl/vl/ol` |
| `progress.arena.txt` | Arena 细粒度心跳，每轮门控开始时清空 |

加载 GPU 权：`load_net`。CPU：`load_cpu_net`。缺键、非有限权、或 \(F\neq\texttt{feature_dim}(k)\) 拒载。

---

## 10. 五子棋 / 反五子棋训练入口

规则见 [`gomoku.md`](gomoku.md)。入口 `train_gomoku_*.bat` / `train_antigomoku_*.bat`（经 `start_train.bat` 加 `--rules`），默认目录 `cur_mod_gomoku_*` / `cur_mod_antigomoku_*`。不走 KataGo 蒸馏。启动器只传 `--rules`；未出现在命令行的项由 `apply_krow_train_defaults` 填：`--sim 800 --gpw 16 --steps 4 --lr 1e-3 --temperature 1 --value-weight 1 --value-rto-weight 1 --own-weight 1 --no-arena`，以及 `--buffer-drop-from-round 11`。`--infinite` 仍由 `start_train.bat` 加上。
