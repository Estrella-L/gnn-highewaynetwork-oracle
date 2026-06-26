# NeurSC

基于图神经网络的**节点对最短路距离回归**项目。给定图上两个节点 `(s, t)`，
用「highway（高速骨架）分解」的三段式 GNN 预测它们之间的近似最短路距离 `d̃(s, t)`。

> 当前可运行主线只有 distance 任务。历史的子图匹配/计数路线（`filtering / graph_coarsen /
> graph_operation / viz_grf / utils` 等独立文件，以及 `model.py`/`gnn.py` 中的
> `BasicCountNet / AttentiveCountNet / CoarsenNet / GIN / GAT` 等类）已于 v0.5.0 全部删除。

---

## 核心思想

最短路距离可以按「高速公路」分解：

```
d(s, t) ≈ d(s → 入口s) + d_highway(入口s → 入口t) + d(入口t → t)
          \___本地接入___/   \_____高速长途_____/   \___本地接入___/
```

模型沿用这一分解，由三段网络分别建模并融合：

1. **Inner-graph GNN（本地段）**：对 `s`、`t` 各自的局部子图用 GraphSAGE 编码，取查询点嵌入。
2. **Inter-graph GNN（高速段）**：在固定的高速骨架图上加入代表 `s`、`t` 的两个虚拟节点，
   分别连到各自最近的高速入口，message passing 后得到 `s`、`t` 两块跨区嵌入。
3. **Fusion MLP（融合回归段）**：拼接 4 块嵌入
   `[h_s_inner | h_t_inner | h_s_inter | h_t_inter]`（可选再拼接 highway 分解距离特征），
   经 MLP + `Softplus` 输出非负距离。

详细的逐模块说明见 `docs/项目说明.md`，版本演进见 `docs/CHANGELOG.md`。

---

## 目录结构

```
.
├── main.py                # 训练入口（distance 主线，输入 .off）；导出 test 点对 CSV
├── infer_distance.py      # 单对节点推理入口
├── generate_terrain.py    # 生成规则网格地形 .off（带起伏，适配距离分解算法；仅需 numpy）
├── baseline.py            # 非学习基线（常数/直线距离），读 test 点对文件与模型对比
├── build_highway.py       # .off → 全局图 + 真正的四叉树分区 + 高速骨干（对齐 EAR-Oracle）
├── preprocess.py          # 样本采样、特征构造、highway 上下文与距离预计算
├── gnn.py                 # InnerGNN / InterGNN / DistancePredictor
├── model.py               # DistanceRegressionNet 包装 + 评估指标
├── data/                  # 输入地形数据
│   ├── sample_terrain.off #   默认示例地形（来自 EAR-Oracle HorseMount，793 顶点）
│   └── generated/         #   generate_terrain.py 生成的规则网格地形
├── outputs/               # 所有运行产物
│   ├── models/            #   训练得到的 .pt 权重
│   ├── results/           #   测试指标 + <run_name>_test_pairs.csv + 基线结果
│   └── params/            #   运行参数快照
├── docs/                  # 文档
│   ├── 项目说明.md        #   实现细节说明
│   ├── 流程文档.md        #   端到端流程与流程图
│   ├── 训练文档.md        #   GPU 实操：1600 点图全流程命令
│   ├── CHANGELOG.md       #   版本更新文档
│   └── note.txt           #   零散笔记
├── README.md              # 本文件
└── requirements.txt
```

> 路径约定：脚本默认在**项目根目录**下解析路径——输入地形相对 `data/` 解析（`--file_folder` 默认 `data`），
> 训练产物固定写入 `outputs/{models,results,params}/`，生成地形默认写入 `data/generated/`。
> 这些路径按脚本所在目录（项目根）解析，因此**在任意工作目录运行都能正确定位**。

---

## 环境安装

需要 Python 3.8+ 与 PyTorch、PyTorch Geometric：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 `numpy / tqdm / torch / torchvision / torchaudio / torch-geometric`。
若 `torch-geometric` 安装失败，请参考其官方文档按 CUDA / PyTorch 版本选择对应 wheel。

---

## 数据格式

输入为 **`.off` 2-manifold 三角网格**（与 EAR-Oracle 一致）：
```
OFF
<nV> <nF> <nE>
<x> <y> <z>            # nV 行，顶点 3D 坐标
3 <v0> <v1> <v2>       # nF 行，三角面（首列为该面顶点数）
```
- 由网格自动构造**全局图**：节点 = 顶点，边 = 网格边，边权 = 顶点间 3D 欧氏距离。
- 用顶点 (x,y) 做四叉树分区与位置特征；不再需要单独的坐标文件。
- 可用自带的 `data/sample_terrain.off`（真实地形示例），或用 `generate_terrain.py` 生成规则网格地形。

---

## 使用方法

### 工作流总览（三步独立、按需运行）

三个环节是**相互解耦的独立脚本**，谁都不会自动触发谁：

```
①（按需）生成地形    ②（核心）训练            ③（可选）基线对比
generate_terrain.py → main.py            → baseline.py
   ↓ data/generated/    ↓ outputs/models/      ↓ outputs/results/
   .off                 .pt + 指标 + test点对   基线指标
```

- **① 地形生成是可选、一次性的**：只在你需要新的规则网格地形时才跑。生成的 `.off` 存到
  `data/generated/`，之后反复训练都直接复用，不会重新生成。用现成的 `data/sample_terrain.off`
  时这一步**完全跳过**。`generate_terrain.py` 只依赖 numpy，不碰 torch。
- **② 训练只读 `--off_file` 指向的 `.off`**，不 import 也不调用生成逻辑，启动开销与地形生成无关。
- **③ 基线在训练导出的 test 点对上评估**，纯 Python、无需 torch，可在训练后任意时刻单独跑。

即"造地形 → 训练 → 评估"按需逐步进行，互不强制依赖。各步骤命令见下。

#### 完整示例：在生成地形上跑一遍（生成 → 训练 → 推理 → 基线）

```bash
# ① 生成一张 400 点地形 → data/generated/terrain_grid_20x20_400v.off
python generate_terrain.py --grid 20

# ② 训练（--off_file 默认相对 data/ 解析，故写 generated/...）
python main.py \
  --off_file generated/terrain_grid_20x20_400v.off \
  --max_depth 3 --capacity 32 \
  --num_epoch 30 --batch_size 32 --device cuda

# ③ 推理（参数须与训练一致；<run> 看 outputs/models/ 下生成的 .pt 名）
python infer_distance.py \
  --model_path outputs/models/<run>.pt \
  --off_file generated/terrain_grid_20x20_400v.off \
  --max_depth 3 --capacity 32 --s 10 --t 300 --device cuda

# ③ 基线对比（读训练导出的 test 点对，与模型同口径）
python baseline.py \
  --off_file generated/terrain_grid_20x20_400v.off \
  --test_pairs_file outputs/results/<run>_test_pairs.csv
```

> `<run>` 是训练自动生成的运行名 `<图名>_distance_<时间戳>`，到 `outputs/models/` / `outputs/results/` 查看实际文件名。
> 换规模：`--grid 10`→100 点、`--grid 40`→1600 点，命令里的 `terrain_grid_NxN_<V>v.off` 同步替换即可。

### 训练
```bash
python main.py \
  --off_file sample_terrain.off \
  --max_depth 3 --capacity 32 \
  --loss_type log_l1 \
  --num_epoch 50 \
  --batch_size 32 \
  --device cuda
```
产物：`outputs/models/<name>.pt`、`outputs/results/<name>.txt`、`outputs/params/<name>.txt`，
以及 test 点对 `outputs/results/<name>_test_pairs.csv`。
模型按验证集 `val_mae` 早停并保存最佳权重。

> GPU 提示：训练支持 `--device cuda` + `--batch_size`，mini-batch 把多个样本的子图/高速图合并成
> 一张大图做**一次**消息传递与优化器步进，显著提升 GPU 利用率。注意启动时的四叉树分区与 Dijkstra
> 预处理是纯 CPU 图算法（与 GPU 无关）；GPU 加速的是 GNN 训练部分。`--device cuda` 在无 GPU 时自动回退 CPU。
> `generate_terrain.py` / `baseline.py` 不涉及神经网络，纯 CPU 运行，无需也无法用 GPU。

> 缓存提示：四叉树分区 + 高速 Dijkstra 预处理结果会缓存到 `outputs/cache/`（按 图名+参数 命名）。
> 首次训练某张图算一次、存盘；之后**相同图与四叉树参数**的训练/推理直接加载缓存，跳过这段等待。
> 同时导出两份审查 CSV：`<key>_partition.csv`（每点的叶子/是否高速/坐标）和 `<key>_highway_edges.csv`
> （高速图的边）。改了 `--max_depth/--capacity/--uniform/--in_feat` 会自动用新缓存；加 `--no_cache` 强制重算。

### 推理（单对顶点）
```bash
python infer_distance.py \
  --model_path outputs/models/<ckpt>.pt \
  --off_file sample_terrain.off \
  --max_depth 3 --capacity 32 \
  --s 10 --t 500 \
  --device cpu
```
输出 `predicted_distance(10->500) = ...`。

> ⚠️ 推理时的四叉树参数（`--max_depth/--capacity/--uniform`）与
> `--disable_highway_distance_feature` 必须与训练时一致，否则分区/高速上下文与模型结构不匹配。

### 仅派生 / 查看四叉树分区 + 高速骨干（不需要 torch）
`build_highway.py` 是 C++ 项目 **EAR-Oracle (SIGMOD'2023)** 四叉树分区 + 边界点高速方案的
图层面实现，直接吃 `.off`：
```bash
python build_highway.py --off_file data/sample_terrain.off --max_depth 3 --capacity 32 --out_prefix terrain
```
输出 `terrain_partition.csv`（每个顶点的叶子编号、是否为高速节点、坐标），并打印分区/高速统计。
- 自适应四叉树（默认）：叶内点数 > `capacity` 且深度 < `max_depth` 才继续四分。
- 加 `--uniform` 则一律分到 `max_depth`（等大叶子，对应 EAR-Oracle 非自适应模式）。
- 细网格 + 小盒子会让高速节点占比偏高；用更小 `max_depth` / 更大 `capacity` 可得更稀疏的高速骨架。

### 生成地形（适配算法的规则网格）
`generate_terrain.py` 仅需 numpy，生成带真实起伏的 `.off`（使图最短路明显偏离直线，适配距离分解）：
```bash
# 一次生成 100 / 400 / 1600 点三张图（命名与云端一致）
python generate_terrain.py --grid 10 20 40
# 平地对照组（z=0，消融用）
python generate_terrain.py --grid 20 --mode flat --out_suffix _flat
```
默认输出到 `data/generated/`；`--mode {flat,smooth,mountains,ridges,mixed}`、`--relief` 控制起伏。

### 非学习基线（与模型对比）
训练时 `main.py` 会把 test 点对导出到 `outputs/results/<run_name>_test_pairs.csv`。
`baseline.py`（纯 Python，无需 torch）读取该文件，在**同一批 test 点对**上算朴素基线：
```bash
python baseline.py \
  --off_file generated/terrain_grid_20x20_400v.off \
  --test_pairs_file outputs/results/<run_name>_test_pairs.csv \
  --out_file outputs/results/baseline_400v.txt
```
输出 `mean_constant / euclidean_2d / euclidean_3d` 三个基线的 mae/rmse/relative_error；
拿其中的 `relative_error` 和模型的 `test_relative_error` 并排比较即可。

---

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--off_file` | （必填） | 输入 `.off` 三角网格路径 |
| `--max_depth` | 3 | 四叉树最大深度 |
| `--capacity` | 32 | 每叶最大点数（自适应模式生效） |
| `--uniform` | 关 | 均匀四叉树（分到 max_depth）替代自适应 |
| `--in_feat` / `--hidden_dim` / `--out_dim` | 64 / 128 / 64 | 特征、隐藏、输出维度 |
| `--learning_rate` | 0.001 | 学习率 |
| `--num_epoch` | 20 | 最大训练轮数 |
| `--batch_size` | 16 | mini-batch 大小（每次 GNN 前向/反向/优化器步的样本数）；`1` = 旧逐样本行为 |
| `--distance_samples` | 3000 | `(s,t)` 对数量上限；`<=0` 表示用全部唯一可达对（大图会自动设保护上限） |
| `--highway_k` | 3 | 每个端点连接的高速入口数 |
| `--inner_mode` | `partition` | Inner-GNN 子图：`partition`=四叉树叶子盒子图(对齐 G1~G4) / `ego`=2-hop ego(消融) |
| `--loss_type` | `log_l1` | `l1` / `log_l1` / `relative` / `huber`；后两者跨尺度归一化误差 |
| `--disable_highway_distance_feature` | 关 | 关闭 highway 分解距离特征（回到纯 GNN 嵌入融合） |
| `--early_stop_patience` | 10 | 早停耐心值 |
| `--device` | `cpu` | `cpu` 或 `cuda` |
| `--cache_dir` | `outputs/cache` | 高速上下文缓存目录；首次算完存盘，之后直接读（跳过分区+Dijkstra） |
| `--no_cache` | 关 | 禁用缓存，每次重新计算分区+高速 |

---

## 评估指标

`model.compute_distance_metrics` 输出：
- `mae`：平均绝对误差
- `rmse`：均方根误差
- `relative_error`：平均相对误差 `mean(|ŷ − y| / (y + ε))`

---

## 已知局限

- 距离监督用网格图最短路（折线测地近似）；EAR-Oracle 用 Snell 加权测地距离更精确。
- 细网格 + 小盒子会让高速节点占比偏高；用更小 `max_depth` / 更大 `capacity` 可得更稀疏的高速骨架。
- Inner-GNN 已改吃**四叉树叶子盒子图**（`--inner_mode partition`，对齐论文 G1~G4）；可用 `--inner_mode ego` 做消融。
- 训练支持 mini-batch（`--batch_size`，批内合并子图/高速图）；Steiner Points / WSPD spanner 尚未建模。

后续计划见 `docs/CHANGELOG.md` 的 TODO 部分。

---

## 常见问题（FAQ）

### Q1. 如何验证「从全局图分解出的四叉树分区子图、高速骨干图」与原图严格一致，且能无损拼回去？

这是分解类方法的正确性基础。我们把它形式化为一组**可执行不变量（invariants）**，核心原则是：
**分解必须是信息守恒且可逆的——拆开再拼回去要严格等于原图**。建议落成一个独立校验脚本
（纯图层面，不依赖 torch），对每份 `.off` 逐条断言，任一条失败即可定位到具体节点/边。

**A. 四叉树分区是「真划分」**

1. 完整覆盖：`set(leaf_of.keys()) == set(graph_info[0])`，`len(leaf_of) == |V|`，无节点遗漏。
2. 互斥：`leaf_of` 为 `node → 单个 leaf_id` 的映射，天然不重叠；重点验证「不漏」。
3. 几何包含：每个节点的 `(x, y)` 必须落在其所属叶子盒的包围盒内（含边界归属规则 `x ≥ px → 右`）。

> 代码证据：分区与归属规则见 `build_highway.py` 的 `build_quadtree`（边界归属 `ix = 1 if x >= px else 0`，约第 188 行）。

**B. 分区子图是「诱导子图」（可映射回原图）**

4. 边来源与权值一致：子图每条边经 local→global 还原后必须在原图存在，且权值相等。
5. 诱导完整性：原图中两端同属一叶子的边必须**全部**出现在该叶子子图中（不丢边）。
6. id 映射可逆：`global_to_local` 与其逆构成双射，`local → global → local` 为恒等。

> 代码证据：诱导子图构造见 `preprocess.py` 的 `_build_inner_subgraph` 与 `_reindex_edges_from_nodes`
> （遍历全图边、两端都在叶子成员集合内才保留）；id 映射见 `build_highway_context` 中
> `global_to_local = {g: i for i, g in enumerate(highway_global_ids)}`。

**C. 高速骨干的点、边确实来自原图**

7. 高速节点 ⊆ 原图节点；独立重算「有跨叶子邻边的节点」，结果集合须与 `boundary` 完全相等。
8. 高速边分两类分别校验：原始边界间边权值 == 原图边权；盒内 transit 边权值 == 在原图独立重跑
   Dijkstra 得到的真实最短路。
9. `access_dist` 与 `highway_pair_dist` 复算一致：分别用原图、高速子图独立重算并逐元素比对。

> 代码证据：高速节点定义见 `build_highway.py:find_boundary_nodes`（约第 212 行，节点存在跨叶子邻边即为边界点）；
> 高速边两类见 `build_highway_edges`（约第 225 行，原始边界间边 ∪ 盒内 boundary→boundary 全图最短路 transit 边）；
> 预计算见 `build_pipeline_inputs` 中 `dist_from_boundary = {b: _dijkstra(adj, b) ...}`（约第 281 行）与
> `highway_pair_dist = [_dijkstra(hadj, s) ...]`。

**D. 「拆开 → 拼回去 == 原图」（round-trip 恒等性，最关键）**

10. 将原图边集按归属拆为 `intra`（两端同叶子）与 `inter`（跨叶子），须满足：
    `intra ∪ inter == 原图边集`、`intra ∩ inter == ∅`、`inter` 端点集合 == `boundary`。
11. 合并所有叶子诱导子图 + 所有跨界边，得到的图（节点集、边集、权值逐一比对）须与原始
    `graph_info` 完全一致，即「分解—重组 = identity」。

**E. 语义一致性（高速距离是原图中的真实路径）**

12. 高速 oracle 的分解距离对应原图中一条真实路径，故对任意采样对 `(s, t)` 应有
    `access(s) + highway_path(入口s, 入口t) + access(t) ≥ d(s, t)`（永不短于真值）。
    若出现某估计 < 真值，说明某段距离并非真来自原图，可立即捕获。

> 代码证据：高速图的每条边权要么是原图真实边权、要么是原图真实最短路（`build_highway_edges`），
> `access_dist` 亦为原图最短路，故三段之和构成 `s→t` 的一条真实游走长度，必然 ≥ 最短路。
> 分解距离特征的拼装见 `preprocess.py:build_synthetic_partition_inputs`（`est = access_s + seg + access_t`）。

---

## 学术审稿常见质疑与回应

以下为面向严谨读者与论文审稿人的问题与正式回答，包含本项目当前**已支持**与**尚存局限**的诚实区分。

### Q2. 实验是否存在数据泄漏？train/val/test 的划分是否可信？

样本为**无放回采样的唯一无向节点对** `(s, t)`（`s < t`），划分前去重，因此不存在「同一对同时出现在
训练集与测试集」的对级泄漏。但需明确：本设置为**单图、直推式（transductive）**——训练与测试共享同一张
全局图、同一套四叉树分区与高速上下文。因此泛化性的主张应严格限定为
「在**同一张图**上对**未见节点对**的距离预测」，而非跨图泛化。跨图归纳泛化（在图 A 训练、图 B 测试）
属当前局限，见 Q7。

> 代码证据：唯一无向对采样见 `preprocess.py:build_distance_samples`（`seen` 集合去重、`if undirected and s > t: s, t = t, s`，约第 425 行）；
> 划分见 `split_distance_dataset`（对已去重样本切片）；单图上下文在 `main.py` 中只构建一次后被 train/val/test 共享。

### Q3. 监督信号（真值距离）是否准确？是否存在系统性偏差？

真值为加权网格图上的 Dijkstra 最短路，其边权为顶点间 **3D 欧氏距离**，因此它是沿网格棱边的
**折线测地近似**，而非连续曲面上的真实测地距离，更非 EAR-Oracle 采用的 Snell 加权测地距离。
这会引入一个**非负的系统性高估偏差**（折线长度 ≥ 真实测地长度），其大小随网格分辨率提高而减小。
本项目的目标是「学习逼近该图最短路 oracle」，监督信号与评估指标在同一度量下自洽；若需对标连续测地真值，
应替换为 Steiner 点加密图或 Snell 距离作为监督，这一项已列入后续计划。

> 代码证据：边权构造见 `build_highway.py:build_mesh_graph`
> （`w = math.sqrt((xu-xv)**2 + (yu-yv)**2 + (zu-zv)**2)`，约第 90 行，即 3D 欧氏边权）；真值由该加权图上的
> Dijkstra 给出（`preprocess.py:_dijkstra_single_source`）。

### Q4. 与哪些基线比较？如何证明 GNN 部分确有贡献，而非仅拟合 highway 距离特征？

本项目提供两条可控对照：
（i）`--disable_highway_distance_feature` 关闭显式 highway 分解距离特征，迫使模型仅凭三段 GNN 嵌入回归，
用于度量该特征的边际贡献；
（ii）`--inner_mode ego` 将 Inner 子图从四叉树叶子盒退化为 2-hop ego 图，用于消融分区结构的作用。
此外，**非学习基线**——直接用 `access(s) + highway_path + access(t)` 作为距离估计（即纯 oracle 上界）——
应作为强基线报告；学习模型若不能显著优于该非学习上界，则其价值存疑。当前仓库尚未内置该基线脚本，
建议补充，这是一个合理且必要的审稿要求。

> 代码证据：两个消融开关见 `main.py` 参数 `--disable_highway_distance_feature` 与 `--inner_mode {partition,ego}`；
> ego 回退为 2-hop 子图见 `preprocess.py:_collect_hop_subgraph(max_hops=2)`；非学习上界的中间量
> `est = access_s + seg + access_t` 已在 `build_synthetic_partition_inputs` 中算出，但未作为独立基线指标输出。

### Q5. 模型输出是否满足距离的度量性质（非负、对称、三角不等式）？

- **非负性**：输出层 `Softplus` 保证 `d̃ ≥ 0`，严格成立。
- **对称性**：当前模型对 `s`、`t` 的处理**不对称**（分别编码 `h_s_inner`、`h_t_inner` 并按固定顺序拼接），
  因此**不保证** `d̃(s, t) = d̃(t, s)`。在无向图任务中这是一个建模缺陷。可通过对称化读出
  （如对 `[h_s, h_t]` 与 `[h_t, h_s]` 取平均或使用对称池化）来强制满足，并以
  `mean|d̃(s,t) − d̃(t,s)|` 作为对称性违反度指标进行报告。
- **三角不等式**：无任何结构性保证；如确有需求需引入度量嵌入约束，当前不主张该性质。

诚实结论：本方法是「距离回归器」，而非「合法度量」；不应宣称其输出构成度量空间。

> 代码证据：非负性见 `gnn.py` `self.output_activation = nn.Softplus()`（约第 223 行）；
> s/t 按固定顺序拼接见 `gnn.py` `fusion_parts = [h_s_inner, h_t_inner, st_virtual_emb]`（约第 275 行），
> 故前向对调 s/t 不保证输出相等。

### Q6. 预处理与训练的计算复杂度、可扩展性如何？

预处理对**每个高速（边界）节点**各跑一次全图 Dijkstra，复杂度约 `O(K · (E log V))`，`K` 为高速节点数；
在细网格 + 小盒子配置下 `K` 可占 `V` 的较大比例，预处理开销显著。高速内部两两最短路为
`O(K · (E_h log K))`。训练支持 **mini-batch**（`--batch_size`，批内把多个样本的子图/高速图合并成
一张大图做一次前向/反向），提升 GPU 利用率与梯度稳定性。可扩展性改进方向包括：
限制每盒边界点数、对 transit 边做稀疏化（如 WSPD spanner）。预处理仍为纯 CPU 图算法（已知工程局限，
不影响方法正确性，但影响大规模适用性）。

> 代码证据：预处理每个边界点一次全图 Dijkstra 见 `build_highway.py` `dist_from_boundary = {b: _dijkstra(adj, b) ...}`（约第 281 行）；
> mini-batch 训练见 `main.py:run_distance_epoch`（按 `--batch_size` 分块，`distance_model.forward_batch(...)` 单次前向/反向/`optimizer.step()`）。

### Q7. 结论的统计可靠性如何？是否报告了方差与多次重复？

当前默认使用固定随机种子（`seed=42`）做单次划分与单次训练，便于复现，但**未报告跨随机种子的均值±方差**，
也**未做跨图的留一验证**。严格的实证评估应：（i）多种子重复并报告置信区间；（ii）多张地形上交叉验证；
（iii）报告对关键超参（`max_depth`、`capacity`、`highway_k`）的敏感性曲线。这是后续实验完善项。

> 代码证据：`main.py` 中采样与划分均使用固定 `seed=42`（`build_distance_samples(..., seed=42)`、`split_distance_dataset(..., seed=42)`），
> 且仅训练一次、单次划分，未对多种子/多图循环报告方差。

### Q8. 与 EAR-Oracle 的关系是什么？本工作的增量贡献在哪？

EAR-Oracle（SIGMOD'2023）是基于 CGAL 的**精确/近似测地距离 oracle**（Steiner 点 + Snell 距离 +
WSPD spanner），无学习成分。本项目在**图抽象层面**复现其四叉树分区与边界点高速骨干，并在其上叠加
一个**三段式 GNN 回归器**，将「本地接入 + 高速长途 + 本地接入」的分解从确定性查询替换为可学习的
表示融合。增量主张应表述为「在 oracle 分解框架内引入可学习距离估计」，而非「超越 EAR-Oracle 的
测地精度」——后者因监督信号为折线近似（Q3）而不成立。对应关系详见 `项目说明.md` 第 6 节。
