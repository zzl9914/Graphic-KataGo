# Graphic-KataGo · 训练方法

本文件讲 **蒸馏与分阶段自对弈**：怎么产第一个有棋力的网、怎么过渡、怎么跨图续训。  
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

①②③ 共用 12 元组样本、F=8 特征、同一 checkpoint 格式。后一阶段 `--resume` 前一阶段的 `new.pt` / `new.npz`。

**不要**把②的「只图 0 / 冻政策头 / 20 轮停 / 关 Arena」写进③。  
**不要**用 3000/25 去乘 value/own MSE：那是把蒸馏阶段 2/3 的头部 lr 倍率误叠进联训，`clip_grad_norm=1` 下会把 policy 梯度挤没。现行 value/own 权重与蒸馏联合阶段相同：**30 / 5**。

对照线（不是主进度）：从随机初始化自对弈，只留 `cur_mod_gnn/experiment/M0.bat`（7×7 sanity gate）。单卡距 AlphaGo Zero 量级差两三个数量级，不作为当前主推。

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
python distill.py --net gnn --data ../distill_data/m2_19x19.jsonl --outdir ../base/gnn
# 然后：base/cultivate2_gnn.bat
# 然后：starter/train_gnn.bat
python web_ui/server.py --port 8765 --device cpu
```

---

## 2. 自对弈样本

一律 12 元组：

`(features, legal_mask, policy, me, score_lead, ownership, q, opp_policy, opp_weight, future, lead, weight)`

- `score_lead`：不含贴目的 \((k\cdot s_{\mathrm{me}}-S)/((k-1)n)\)（二人即 \((\text{我}-\text{对方})/n\)）。这一槽写的是 mix 目标 \(\lambda\cdot\mathrm{lead}+(1-\lambda)\cdot Q\)（belief 仍用下面的 `lead`）
- `ownership[i]\in[-1,1]`：终局该点从 `me` 视角的归属
- `q`：该步根节点 MCTS Q（与 lead 同量纲），给 stdev 头
- `opp_policy`：下一手（下一家）的访问分布；最后一步 `opp_weight=0`
- `future`：约 4 手后的占子图（C++ `FUTURE_PLIES`；+1 我 / −1 对方 / 0 空）
- `lead`：终局同一公式，给 score-belief，不随 `mix` 搅浑
- `weight`：policy surprise \(\mathrm{KL}(\pi_{\mathrm{MCTS}}\|\pi_{\mathrm{nn}})\) 映射到 \([0.5,8]\)，SGD 按样本加权
- GPU 损失：`(policy, value, own, aux, total)`，`total` 含加权辅助项
- CPU `backward` 返回四元组，`total` 含辅助损失

### 2.1 价值语义（KataGo 标签 vs 学生网）

**同一类量，不是两种分数。** 都是当前走子方领先、无贴目、除以 \(n\)、落在 \([-1,1]\)：

| | 蒸馏教师 | 自对弈学生 |
|---|---|---|
| 来源 | KataGo `scoreLead / n`（查询 `komi: 0`、`tromp-taylor`，`SIDETOMOVE`） | `finalize` 后的图围棋 `score_lead` 与根 Q 的 mix |
| 含义 | 强教师搜索下的**期望**终局差 | \(\lambda z+(1-\lambda)Q\)（\(z\) 是这一局实现值） |
| 噪声 | 很小 | 大（学生棋 + 随机撒子开局 + 截断） |

网络头是 `tanh` 输出，学的就是这个尺度。差在估计方式（期望 vs 实现），不在单位。自对弈固定 mix，`--q-lambda` 默认 0.5，让目标更接近「搜索期望」，而不是纯 MC 抽奖。不要把 \(\lambda\) 设成 0：value 若还贴 0，纯 Q 会自举锁死。

---

## 3. 蒸馏（基础培养）

`scr/distill.py` 把 KataGo 强标签监督拟合到图无关网络，产出第一个有棋力的 checkpoint（GPU `new.pt`、CPU `new.npz`）。`--net`：`gnn`（图无关主线）、`2dcnn`（网格对照）、`mlp` / `1dcnn`（消融）。

- **三阶段**（各 `--epochs` 轮，默认 10+10+10）：① 联合 policy + value + own，**不冻**任何参数，Adam \(lr=1\mathrm{e}{-4}\)，loss = `pl + 30*vl + 5*ol`（报告的 pl / vl / ol 仍未加权）；② 只训 own 头（冻 trunk，Adam lr ×25，unweighted MSE）；③ 只训 value 头（冻 trunk，Adam lr ×100）。GPU 用 Adam（β=0.9/0.999）；CPU 蒸馏同样用 NumPy Adam。辅助头（opp / soft / belief / stdev / future）蒸馏阶段无教师信号，保持初始化，到自对弈再训。
- **特征逐位一致**：`build_sample` 用 native `extract_features`，与自对弈同源；value/own 用 `/n`，跨图可比。
- **训练增强、验证集固定**：训练集每个 batch 按网络分派与自对弈 SGD 相同的置换（见 [`algorithm.md`](algorithm.md) §8.5）。**课程打乱**：默认 `--aug-from-epoch 4`，全局 epoch 1–3 用原编号，从第 4 轮起才打乱（`1` = 第一轮就打乱，`0` = 全程不打乱）。**验证集**是 JSONL 顺序前 `val-frac`（默认 5%）、不做增强、不重采样。心跳行带 `aug=on/off`。
- **逐步心跳**：`outdir/progress.txt`，每个 epoch 开始时清空；逐 train/val step 打 `pl/vl/ol`，行首带 `stage=policy|own|value e{k}/10 g{G}/30`。
- **数据流**：`gen_katago_data.py`（驱动 `katago/katago.exe analysis`）→ `distill_katago.py`（KataGo 坐标 → gkt 顶点顺序 JSONL）→ `distill.py`。启动器默认读 `distill_data/m2_19x19.jsonl`。再跑同一 bat 会 `--resume` 最高 `round*.pt`。
- **内存**：19×19 稠密邻接在 6 GB 卡上 batch 上限约 **16**（32 OOM）。`--hidden 512 --n-blocks 20` 与自对弈默认一致。CPU 1DCNN 蒸馏 mini-batch 默认 64。

蒸馏打破 value 塌缩死锁；接下来接基础培养2，再接正式跨图。不要把刚蒸完的网直接丢进 3000/25。

---

## 4. 基础培养2

插在蒸馏和正式跨图之间，**不是**正式训练。启动器 `base/cultivate2_{gnn,mlp,cnn1d,cnn2d}.bat`，产物 `base/cultivate2/<arch>/`。

| 项 | 设定 |
|---|---|
| 图 | 只训 `0`（19×19） |
| 轮次 | `--rounds 20`，无 `--infinite`，然后停止 |
| 权重 | `--value-weight 30 --own-weight 5` |
| 价值目标 | mix（`--q-lambda 0.5`） |
| Arena | **关**（`--no-arena`）。此阶段要让 value/own 在学生自对弈上动起来；回滚会丢掉刚学到的 value |
| replay | `--buffer-drop-from-round 21`（20 轮内不丢） |
| 快照 | `--model-snapshot-rounds 25` |
| 其余 | `--sim 256 --workers 1 --gpw 32 --steps 16 --lr 1e-4 --temperature 0.1` |

日程：

- 轮 1–10：`--freeze-policy-until-round 10`。冻结已脱离随机的 **policy 读出**（GNN `policy_head`、2DCNN `policy_conv`、CPU `Wp`/`bp`/`W_pass`/`b_pass`）。读出权重不动；policy CE **仍回传到 trunk**。训 trunk + value + own + aux。
- 轮 11–20：解冻 policy，仍只训图 0。

CLI：`gkt_train_gpu.py` / `gkt_train_cpu.py` 的 `--freeze-policy-until-round N`（默认 0 = 不冻）。GPU 每张图新建 `GktTrainer` 时按当前轮决定；CPU 在每轮开头设 `net.freeze_policy`。

**续训**：再跑同一个 bat。已有 `cultivate2/<arch>/new.pt|.npz` 则从它续，并用该目录 `summary.json` 接到下一轮；否则 `--resume` 蒸馏终态。没有蒸馏产物则报错，不去随机初始化。中途 Ctrl+C 停在某一轮打到一半时，从上一轮已存盘之后重来。20 轮已经跑完再点一次会多进第 21 轮再因 `--rounds 20` 停。

Windows `.bat` 里 `if (...)` 块中的 `echo` **不能写裸括号**（会被当成语法）；汉字若用 UTF-8 无 BOM，在默认 GBK 的 `cmd` 里会乱码，不影响 Python 训练命令。

---

## 5. 正式跨图

入口 `starter/train_*.bat` → `start_train.bat` → `gkt_train_gpu.py` / `gkt_train_cpu.py`，目录 `cur_mod_*`。

| 项 | 设定 |
|---|---|
| 图 | 默认训练图（排除 oversized `2`/`6` 与五子棋 `G*`） |
| 轮次 | `--infinite` |
| 权重 | `--value-weight 30 --own-weight 5` |
| 价值目标 | mix（`--q-lambda 0.5`） |
| Arena | **开**（默认） |
| replay | `--buffer-drop-from-round 5` |
| 温度 | `--temperature 0.1` |
| 其余 | `--sim 256 --workers 1 --gpw 32 --steps 16 --lr 1e-4` |

GPU 加 `--device cuda --selfplay-device cuda`。有 `cur_mod_*/new.pt` 则续训。要从培养2 接过来，把 `base/cultivate2/<arch>/new.pt` 拷进 `cur_mod_*` 或显式 `--resume`。

### 5.1 随机让子开局

仅**自对弈训练**（`make_training_game`）。Arena / Web / 评估从空盘开始。

每局先抽 \(p\sim\mathcal{U}(0,\,p_{\max})\)，每个顶点独立以概率 \(p\) 放子（颜色在 \(1..k\) 上均匀），否则为空。图围棋 \(p_{\max}=0.25\)。五子棋 \(p_{\max}=0.12\)。然后用规则引擎对静态盘提子：从末座到首座依次提无气块。子数最少的一方先行。Zobrist 按整盘占领重算，历史为空。

### 5.2 跨图循环、课程、Arena、replay

- 排除超大图 `2`、`6`。二者都不进默认训练，但 UI 都可下。
- 每轮 `random.Random(rnd).shuffle` 图顺序（GPU / CPU 相同，可复现）。`--resume` 读 `summary.json` 末行，**cycle 序号接续**，并从下一张图接着训。**若末行恰好是该轮最后一张图且 Arena 开启**，先补跑这一次门控（图列表置空、跳过 round-end 产物，直接落入 Arena），再进下一轮；`best_weights` 从 `best.*` 恢复（无则回退当前权重）。
- 课程（**默认关闭**，`--curriculum-rounds 0` = 全程全长）。从 0 时显式传 `--min-moves 40 --curriculum-rounds 20000` 可重启：名义手数上限从 `--min-moves` 涨到 \(n\cdot\mathrm{factor}+\)min-moves。自对弈每局再对数均匀抽 \([\mathrm{cap}/2,\,2\cdot\mathrm{cap}]\)。停表后 `finalize` 提供中局领地监督。图围棋自对弈在前 \(\texttt{forbid_pass_after}=\min(\mathrm{cap},\,\max(\lfloor n/8\rfloor,16))\) 手若仍有合法落子则 **MCTS 去掉虚手**（规则层虚手仍合法；UI / Arena / Elo 不禁）。
- 每图结束（CUDA）：`gc.collect()` + `empty_cache()`。
- 自对弈：同一套权打满 \(k\) 个座位。
- Arena（**默认开**，`--no-arena` 关）：新 vs 面板跨图门槛，**据此接受或回滚**；固定 `--arena-sim`。面板 = best-so-far + 上一轮小快照 `round{rnd-1}` + 最近 `--big-snapshot-rounds` 个大快照 `big*`。新网轮流占一座、对手占其余；每局记连续归一化 `score_lead`，交替先后手使先手优势 δ 作为加性项抵消，再与 `--arena-lead-threshold`（默认 0.0）比。带心跳：`arena: round N gate starting …`、逐「图×对手」、结束汇总。
- `--players {2,3,4}`：自对弈人数与 \(F=\texttt{feature_dim}(k)\) 一致；2P/3P/4P 分网。
- `--lr` 默认 \(1\mathrm{e}{-4}\)。非有限 loss / 梯度跳过该次 SGD；拒绝把非有限权写入 `new.pt`。
- **每图 unused 队列**：`outdir/replay/{key}/unused.npz`。SGD 成功用过的样本从 buffer 删除、不写回。未用样本下次再训同一张图时仍会加载。条数上限 `buffer_capacity`（默认 20 万）。图与图不混。`--replay-rounds 1` 不读盘上队列。从 round `--buffer-drop-from-round`（代码默认 11；围棋正式 starter 传 5）起，每训完一张图按 `--buffer-drop` 丢掉最旧样本（默认 auto = 本轮未使用量 `new - consumed`）。
- Elo（`gkt_elo.py`）：四种架构 + 均匀随机共用一张榜，**只供展示**，不替代 Arena。

```bash
cd scr
python gkt_elo.py --models-dir ../models --sim 100 --games 2 --device cpu
```

---

## 6. 从 0 对照

`gkt_train_gpu.py` / `gkt_train_cpu.py` 也可以随机初始化、不 `--resume`。这是命题的理论完备形态，单卡不可行，只留 `M0`（7×7、`--no-arena`）作 sanity gate。默认 CLI 的 `--value-weight 1 --own-weight 1 --temperature 1.0` 是这条对照线的底（value 仍是 mix）；围棋主线 bat 会改写成 30/5、\(\tau=0.1\)。

---

## 7. 损失与价值目标（现行）

| 旗标 | 蒸馏① | 基础培养2 | 正式跨图 | 从 0 / 五子棋默认 |
|---|---|---|---|---|
| `--value-weight` | 30 | 30 | 30 | 1 |
| `--own-weight` | 5 | 5 | 5 | 1 |
| value 目标 | 教师 `scoreLead/n`（监督） | mix | mix | mix |
| `--q-lambda` | — | 0.5（\(z\) 的权重） | 0.5 | 0.5 |

报告的 pl / vl / ol **始终未加权**。不要用 3000/25。

---

## 8. 分布式

官方 KataGo 是异步进程靠共享目录衔接：selfplay → shuffle → train → 可选 gatekeeper。本仓库对应 `scr/gkt_dist.py`。

**单机、只想接着训 `cur_mod_*`：** 用 `gkt_train_*` 或 `start_train.bat`。不要和 `gkt_dist` 抢同一套权；分布式 basedir 默认 `../dist_run`。

**多机：** 只有一台当服务器写权，其余当自对弈客户。客户不必挂共享盘，走 HTTP（`serve` + `contribute`）。`selfplay` / `shuffle` / `train` / `gate` 必须看见同一个 `--basedir`。

| 机器 | 跑什么 |
|---|---|
| 服务器（建议最强 GPU） | `init` 一次；长期 `shuffle`、`train`、`serve`；可选 `gate` |
| 服务器上也可再开 | `selfplay --device cuda` |
| 其他电脑 | 只跑 `contribute --url http://服务器LAN地址:8877` |

同一 basedir 只对应**一种**网、**一种**人数、**一种规则**。2P GNN 和 3P、GNN 和 2DCNN、或图围棋和五子棋，不要混在一个 `dist_run` 里。

五子棋：

```bash
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gomoku_gnn/new.pt --net gnn --rules gomoku --token 自己设一串密文
```

默认图列表为 `GOMOKU_TRAIN_KEYS`；手数上限固定为 \(n\)。种子必须是 2P。

目录（`--basedir`）：

| 目录 | 角色 |
|---|---|
| `run.json` | 图列表、sim、窗口、是否门控、token |
| `models/` | 已接受网络 |
| `selfplay/` | 自对弈 12 元组 `.npz`；shuffle 只保留每图最近 `window_files` 个 |
| `shuffled/` | 按图滑窗打乱后的训练包 |
| `modelstobetested/` | 训练刚写出、待门控（仅 `init --gate`） |
| `rejectedmodels/` | 门控未过 |
| `client_cache/` | 服务器本机缓存；客户机用 `contribute --cache` |

```bash
cd scr
python gkt_dist.py init --basedir ../dist_run --from ../cur_mod_gnn/new.pt --net gnn --token 自己设一串密文
python gkt_dist.py shuffle --basedir ../dist_run
python gkt_dist.py train --basedir ../dist_run --device cuda
python gkt_dist.py serve --basedir ../dist_run --host 0.0.0.0 --port 8877 --token 与init相同
```

`init --gate` 才启用门控，此时必须另开 `gate`，训练写入 `modelstobetested/`。默认不加 `--gate`：`train` 直接写入 `models/`。

`serve` **默认只绑 `127.0.0.1`**，局域网客户必须 `--host 0.0.0.0`。Windows 防火墙放行 8877。不要把该端口暴露到公网；`--token` 写入 `X-Token`。

客户机：

```bash
cd scr
python gkt_dist.py contribute --url http://服务器LAN_IP:8877 --token 与init相同 --device cuda
```

无 GPU 则 `--device cpu`。循环：`GET /api/task` → 必要时 `GET /api/model/…` → 下一局 → `POST /api/games`。

多台都能读写同一个 `dist_run`（SMB/NFS）时，客户机也可跑 `selfplay --basedir 映射路径`，不必 HTTP。仍应只有一个 `shuffle` 和一个 `train`。Windows 网络盘上多进程写 `.npz` 偶发锁文件，出问题就改回 HTTP。

| 官方 KataGo | 本仓库 |
|---|---|
| 自对弈进程写 `selfplay/` | `gkt_dist.py selfplay` 或 `contribute` |
| shuffle 窗口 | `gkt_dist.py shuffle` |
| train + export | `gkt_dist.py train`（写出 `.pt`/`.npz`） |
| gatekeeper | `gkt_dist.py gate`（Arena；可选） |
| `katago contribute` | `gkt_dist.py contribute` |

搜索仍是每进程自己的 `predict_batch`，没有参数服务器。

---

## 9. 检查点

| 产物 | 含义 |
|---|---|
| `new.pt` / `new.npz` | 最近完成的一张图之后的共享权（`--resume` 用这个） |
| `roundN.pt` / `roundN.npz` | 整轮结束；默认只保留最近 `--model-snapshot-rounds`（5）轮（`new.*` / `best.*` 不动） |
| `bigN.pt` / `bigN.npz` | 长程大快照：每 `--big-snapshot-interval`（默认 10）轮存一份，默认只留最近 `--big-snapshot-rounds`（5）个；兼作 Arena 长程对手 |
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

规则见 [`gomoku.md`](gomoku.md)。入口 `train_gomoku_*.bat` / `train_antigomoku_*.bat`（经 `start_train.bat` 加 `--rules`），默认目录 `cur_mod_gomoku_*` / `cur_mod_antigomoku_*`。不走 KataGo 蒸馏；默认 `--lr 1e-3 --no-arena --infinite`，value/own 权重保持训练器默认 1。
