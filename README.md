# Graphic-KataGo

**Graphic-KataGo**（**GKT Go**）把围棋从匀质网格推广到**任意有向图**：气是空入邻居，规则与图同构。代码与模块名里的 `gkt` / `Gkt*` 即此缩写。

要检验的是：围棋策略有多少绑在棋盘形状上，有多少是「占子—气—围歼」本身。同一套与 \(n\) 无关的权重，应能在乱图、有向边、网格之间迁移。

- 图围棋主网是 **GNN**（拓扑换了还能不能下）。
- **2DCNN** 是网格对照（形状还在时还剩多少）；五子棋 / 反五子棋反过来检验它。
- **MLP / 1DCNN** 是消融，不是认真选手。

热路径在 C++（`gkt_native`：规则、MCTS、自对弈）；Python 只做调度与训练。搜索用的 **value_rto** 是 mix \(\lambda z+(1-\lambda)Q\)（`--q-lambda` 默认 0.5）；监督用的 **value_abs** 是子数差。二者有很低的一致性项（`--value-cons-weight` 默认 0.01）。Windows 入口钉死 **Python 3.14**（非 free-threading）。

---

## 仓库目录

| 路径 | 做什么 |
|---|---|
| `cpp/` | 规则引擎、MCTS、自对弈；`python cpp/_build_with_sdk.py` 再 `_deploy_pyd.py` → `scr/` |
| `scr/` | 网络、蒸馏、跨图训练、Web UI |
| `ref/` | 规则与算法文档（下表） |
| `base/` | 蒸馏与基础培养2 的启动器、产物 |
| `starter/` | 图围棋正式跨图（`train_*.bat` → `start_train.bat`） |
| `starter_gomoku/` `starter_anti_gomoku/` | 五子棋 / 反五子棋，不走 KataGo 蒸馏 |
| `katago/` | 本机 KataGo，给蒸馏产教师标签 |
| `distill_data/` | 蒸馏 JSONL |
| `cur_mod_*` | 正式跨图输出 |
| `dist_run/` | 分布式 basedir（`start_dist_main.bat` 默认） |
| `models/` | Web UI / Elo 用的已部署网 |
| `start_train.bat` | 本机跨图训练（`starter/` 调用） |
| `start_ui.bat` | 打开对弈界面 |
| `start_dist_main.bat` | 分布式教练机：init + shuffle / train / serve |
| `start_dist_cont.bat` | 分布式客户机：`contribute` |
| `eol.py` | `.bat` 换行：写完后 `python eol.py`（LF→CRLF）；`--to lf` 相反 |

---

## 怎么跑

围棋主线，四种架构各有一套 bat：

1. 蒸馏：`base/distill_{gnn,mlp,cnn1d,cnn2d}.bat` → `base/<arch>/new.pt` 或 `new.npz`
2. 基础培养2：`base/cultivate2_*.bat` → `base/cultivate2/<arch>/`（只图 0，20 轮停）
3. 正式跨图：`starter/train_*.bat` → `cur_mod_*`（全图、无限轮、Arena 开）

环境、旗标、检查点见 [`ref/training_method.md`](ref/training_method.md)。对弈：`start_ui.bat`（或 `cd scr` 后 `python web_ui/server.py`）。多机：教练机 `start_dist_main.bat`，客户机 `start_dist_cont.bat`（见 [`ref/training_method.md`](ref/training_method.md) §8）。

---

## 文档

| 文件 | 读什么 |
|---|---|
| [`ref/rules.md`](ref/rules.md) | 图围棋规则：盘、气、提子、禁手、终局、多人 |
| [`ref/gomoku.md`](ref/gomoku.md) | 五子棋 / 反五子棋规则与实验角色 |
| [`ref/algorithm.md`](ref/algorithm.md) | 特征、MCTS、多头、四种网络、并行、UI、内置图 |
| [`ref/training_method.md`](ref/training_method.md) | 蒸馏 → 培养2 → 跨图；双 value 头与 mix；分布式 |
| [`ref/go_glossary.md`](ref/go_glossary.md) | 围棋术语 ↔ 网络头 |
