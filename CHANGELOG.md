# 版本更新文档 (CHANGELOG)

本文件按版本记录 NeurSC 距离回归主线（highway 分解方案）的演进。
最新版本在最上方。新增条目请复制文末「条目模板」。

格式约定：
- **动机**：为什么要改
- **改动**：做了什么（按文件）
- **接口/参数变化**：新增或语义变化的 CLI 参数、函数签名、模型结构
- **兼容性**：是否影响旧 checkpoint / 旧调用
- **验证**：如何确认本次改动正确

---

## [v0.4.1] - 2026-06-12 — Inner-GNN 改吃四叉树分区子图（落地论文 G1~G4）

### 动机
此前 Inner-GNN 用 2-hop ego 子图编码 s/t 的局部结构，与论文「按 highway 切分的分区 G_i」不一致。
本版本让 Inner-GNN 直接吃 **s/t 所在四叉树叶子盒的诱导子图**，真正对齐论文结构。

### 改动
**`preprocess.py`**
- `build_highway_context` 新增 `leaf_of` 参数，构造并缓存 `leaf_members`（叶子盒 → 成员节点）；
  上下文新增 `leaf_of / leaf_members / inner_cache`。
- 新增 `_build_inner_subgraph(node, context, ...)`：
  - `partition`（默认）：用 node 所在叶子盒的**诱导子图**（节点=盒内顶点，边=盒内边）；
    每个盒子的子图张量按 `cell_id` **缓存复用**（大量 (s,t) 对共享盒子，避免重复构建，显著提速）。
  - `ego`：退回 2-hop ego 子图（消融对比用）。
- `build_synthetic_partition_inputs` 新增 `inner_mode` 参数，s/t 的 Inner 输入改由该辅助函数产出。

**`build_highway.py`**：`build_pipeline_inputs` 把 `leaf_of` 传入 `build_highway_context`。

**`main.py` / `infer_distance.py`**：新增 `--inner_mode {partition, ego}`（默认 partition，推理需与训练一致）。

### 兼容性
- 默认行为变化：Inner-GNN 从 ego 子图变为分区子图。`--inner_mode ego` 可恢复旧行为。
- 旧 checkpoint 仍可加载（模型结构未变），但训练分布不同，建议重训。

### 验证
- `py_compile` 通过。
- 不依赖 torch 的分区子图自检（sample_terrain.off, max_depth=2/capacity=64）：
  16 个占用叶子，盒大小 27~70（均值 49）；抽查节点的盒子诱导子图均包含该节点、局部索引正确、
  诱导边端点都在盒内 → `PARTITION_SUBGRAPH_OK`。

### 已知局限 / 后续 TODO
- 叶子盒大小受 `max_depth/capacity` 控制；盒子过大→ Inner-GNN 子图大、过小→ 难以包含足够局部结构。
- 仍是逐样本训练；Steiner 点 / WSPD spanner / Snell 测地距离未建模。

---

## [v0.4.0] - 2026-06-12 — 整体改造为 .off 输入 + 真正的四叉树流水线

### 动机
按 EAR-Oracle 的真实形态对齐：**输入直接采用 .off 2-manifold 三角网格**，并把
`build_highway.py` 从均匀网格升级为**真正的递归四叉树**。旧的手工 `.grf`/坐标 `.csv` 输入
被淘汰；旧模型权重失效，需重新训练（"重新来过"）。

### 改动
**`build_highway.py`（重写）**
- `load_off`：读取 `.off`（`OFF` 头 + `nV nF nE` + `x y z` + 面），鲁棒处理排版差异。
- `build_mesh_graph`：网格 → 无向加权图（节点=顶点，边=网格边，**权=3D 欧氏距离**），
  产出 `graph_info` + 全节点坐标 `coords`（不再需要外挂坐标）。
- `build_quadtree`：**真正的递归四叉树**，每节点四分为 SW/SE/NW/NE：
  - 自适应（默认）：盒内点数 > `capacity` 且深度 < `max_depth` 才继续四分（叶子大小不均）；
  - 均匀（`--uniform`）：一律分到 `max_depth`（4^depth 个等大叶子，对应 EAR-Oracle 非自适应模式）。
- `find_boundary_nodes` / `build_highway_edges`：有跨叶子邻边的节点 = 高速节点；
  高速边 = 原始跨界边 ∪ 盒内 boundary-to-boundary 全图最短路 transit 边。
- `build_pipeline_inputs(off_path, ...)`：一站式产出 `(graph_info, coords, leaf_of, num_leaves, highway_context)`，供训练/推理调用（纯图部分不依赖 torch，张量化延迟导入）。

**`preprocess.py`**
- 删除 `.grf`/`.csv` 读取（`load_external_highway_context`、`load_node_coordinates`、`_resolve_path`、`_build_highway_local_adj`）。
- 新增 `build_highway_context(...)`：根据派生好的高速节点/边/距离组装张量与特征。
- `build_distance_samples` 重写为**对大图高效**：无放回抽样唯一对 → 按源点分组 → 每源点只跑一次 Dijkstra（Dijkstra 次数 = 不同源点数，避免对数千点网格做全 APSP）；大图自动设上限保护。
- `_build_weighted_adj_list`：边权不再被 `max(1.0, w)` 截断，支持真实浮点距离。

**`main.py` / `infer_distance.py`（重写）**
- 入口参数改为 `--off_file --max_depth --capacity [--uniform]`，移除 `--graph_file/--highway_file/--coords_file`。
- 调用 `build_pipeline_inputs` 一次性构建流水线；距离采样固定 `weighted=True, undirected=True`。

### 接口/参数变化
- 训练/推理输入：`.grf + .csv` → **单个 `.off`** + 四叉树参数。
- `build_distance_samples(..., weighted=True, undirected=True)` 默认值变化（网格场景）。

### 兼容性
- **不兼容旧数据与旧模型**：`.grf/.csv` 输入、`saved_models/*.pt` 均已删除，需用 `.off` 重新训练。
- 已删除工作区数据/模型：`cross*.grf`、`cross15_coords.csv`、`cross30_auto_*`、`saved_models|saved_params|saved_results/*`。
- 新增示例数据 `sample_terrain.off`（来自 EAR-Oracle `datasets/small/HorseMount.off`，793 顶点）。

### 验证
- `py_compile` 全部通过。
- `build_highway.py` CLI 在 `sample_terrain.off` 上：793 顶点/1488 面，四叉树派生分区 + 高速节点成功。
- 纯图流水线端到端自检（不依赖 torch）：793 顶点、16 叶子、338 高速节点、7916 高速边、
  highway_pair_dist 计算完成，0.62s。
- ⚠️ 端到端训练未在本机执行：环境未安装 `torch / torch_geometric`。安装后用下方命令复跑。

### 运行
```bash
# 训练
python main.py --off_file sample_terrain.off --max_depth 3 --capacity 32 --num_epoch 30
# 仅派生/查看分区+高速（不需要 torch）
python build_highway.py --off_file sample_terrain.off --max_depth 3 --capacity 32 --out_prefix terrain
# 推理（参数需与训练一致）
python infer_distance.py --model_path saved_models/<ckpt>.pt --off_file sample_terrain.off \
    --max_depth 3 --capacity 32 --s 10 --t 500
```

### 已知局限 / 后续 TODO
- 距离监督用网格图最短路（折线测地近似）；EAR-Oracle 用 Snell 加权测地距离更精确。
- 细网格 + 小盒子会让高速节点比例偏高（边界点过多）；用更少/更大盒子（小 `max_depth`、大 `capacity`）可得更稀疏的高速骨架。
- Inner-GNN 仍用 2-hop ego 子图；可改为吃四叉树叶子盒子图（真正对齐论文的 G1~G4 分区）。
- 未做 Steiner 点 / WSPD spanner（依赖几何，留待后续）。

---

## [v0.3.0] - 2026-06-12 — 从全局图自动派生分区与高速骨干（对齐 EAR-Oracle）

### 动机
本项目是在 C++ 项目 **EAR-Oracle (SIGMOD'2023, weighted_distance_oracle)** 之上加 GNN 的拓展。
EAR-Oracle 的核心是：在 terrain 网格上用 **ζ×ζ 网格/四叉树盒子**做分区，盒子**边界顶点**即为
**highway 节点**，再用盒内接入距离 + 盒间高速网络回答距离查询。此前本项目的 `cross15.grf`
高速图是**手工**编写的，没有从全局图自动生成；本版本补上这一步。

### 改动
**新增 `build_highway.py`**（纯 Python，不依赖 torch）：从「全局图 + 节点坐标」自动派生
分区与高速骨干图，是 EAR-Oracle 网格/边界点方案的**图层面适配版**：
- `assign_grid`：对应 EAR-Oracle 的 `fastRetrieveGridBoundary`，按坐标把包围盒切成 `side×side` 个盒子（`grid_num=side²`）。
- `find_boundary_nodes`：有跨盒子邻边的节点 = 高速(边界)节点（对应 `boundary_points_id`）。
- `build_highway_edges`：高速节点之间用「原始跨界边 ∪ 盒内全图最短路 transit 边」连成骨架。
- `propagate_coords`：缺坐标节点继承图上最近已知坐标。
- 输出 `<prefix>_highway.grf`（与 `load_external_highway_context` 兼容）和 `<prefix>_partition.csv`。

CLI：
```bash
python build_highway.py --graph_file cross30_full.grf --coords_file cross15_coords.csv \
    --grid_num 4 --out_prefix cross30_auto
```

### 验证
- 在 `cross30_full.grf` 上运行成功：30 节点 → 2×2 网格 → 高速(边界)节点 `{0,4,11}`，
  正是十字交叉处跨盒子的节点；输出文件可被现有 highway 加载逻辑直接读取。

### 已知局限 / 后续 TODO
- 这是**图层面适配**，非 1:1 移植：EAR-Oracle 的 Steiner 点、Snell 加权测地距离、WSPD spanner
  依赖网格几何与 CGAL，无法在抽象图上复刻。这里保留其核心结构（网格分区 + 边界点高速 + 盒内接入距离）。
- 当前用均匀网格；可扩展为四叉树（递归细分）以贴合 EAR-Oracle 的多层结构。
- 下一步可把 `--graph_file` 直接换成 EAR-Oracle 导出的 base graph（Steiner 点图 + 坐标），实现端到端对齐。

### 关于图文件格式的结论
- EAR-Oracle 输入是 **`.off` 三角网格**（`OFF` → `nV nF nE` → 每点 `x y z` → 面 `3 v0 v1 v2`），**自带 3D 几何**。
- 本项目 `.grf` 是**无几何**的抽象标注图，坐标靠旁边的 CSV 拼接、边权语义含糊、有向/无向靠约定且零校验。
- 因此若要严格延续 EAR-Oracle，`.grf` **不是理想输入**：分区(网格/四叉树)与距离本质都依赖坐标与权重。
  建议让坐标与边权成为一等公民（或直接消费 EAR-Oracle 的 base graph 导出），并加加载期校验。

---

## [v0.2.1] - 2026-06-12 — 修复样本采样/分割的重复与数据泄漏

### 动机
30 节点的 `cross30_full` 图最多只有 870 个有向对（无向 435 个），但历史训练用了
`distance_samples=2000`（甚至更多），逐样本更新导致每轮上千次梯度步——明显不合理。
排查发现 `build_distance_samples` 采用**有放回随机采样且不去重**，且每次试探都重跑一次
单源 Dijkstra：
1. 同一 `(s,t)` 被重复抽样，浪费算力（要 2000 个样本就抽 2000 次、各跑一次 Dijkstra）；
2. 重复对同时落入 train/val/test → **数据泄漏**，测试指标虚高、不可信。

### 改动
**`preprocess.py`**
- 重写 `build_distance_samples`：改用 **APSP**（每个源点跑一次 Dijkstra，共 n 次）一次性枚举
  **全部唯一可达对**，每个 `(s,t)` 只出现一次；`num_samples` 改为「上限」，需要时做**无放回**下采样。
- 新增 `undirected` 选项：只取 `s<t` 的无向对，避免 `(s,t)/(t,s)` 标签相同造成的近泄漏。
- `split_distance_dataset`：输入已唯一，train/val/test 不再重叠同一对（补充了文档说明）。

**`main.py`**
- `--distance_samples` 语义改为「上限」，默认 `0` = 使用全部唯一对（不再 `max(100, ...)` 强制放大）。
- 新增 `--undirected_pairs`。

### 接口/参数变化
- `build_distance_samples(graph_info, num_samples=None, weighted=False, seed=42, undirected=False)`：
  `num_samples` 由「目标数量（有放回）」变为「上限（无放回）」，`None/<=0` 表示全部。
- `--distance_samples` 默认由 `870` 改为 `0`（全部）。

### 兼容性
- 不影响模型结构与 checkpoint；仅改变训练数据的构造方式。
- 行为变化：相同 `--distance_samples` 下样本数可能变少（去重后不超过唯一对总数）。

### 验证
- `py_compile` 通过。
- 数据规模自检：`cross30_full` 有向唯一对 = 870、无向 = 435，与枚举结果一致；
  去重后切分不再有同一对跨 train/test。

---

## [v0.2.0] - 2026-06-12 — highway 分解方案问题修复

### 动机
对照论文架构图与项目设想，发现当前实现虽然搭起了「双 Inner-GNN + Inter-GNN + 融合 MLP」三段式骨架，
但有 5 处问题导致它无法真正预测距离（详见 `项目说明.md` 与架构评审）：

1. 融合向量只有 3 块（`[s_inner, t_inner, h_st_inter]`），其中 Inter 段被一个 readout MLP
   塌缩成单一向量，与图中「s(绿) / t(棕) 两块独立」不符。
2. Inner/全局特征没有真实位置信息，模型无法定位 s、t（GNN 测距的根本障碍）。
3. s/t 接入高速按 **node id 差**近似最近，与真实图距离无关（在十字玩具图上 30 个节点错 14 个）。
4. Inter-GNN 仅 2 层，感受野无法覆盖两个虚拟节点之间的高速长程距离 → 长途段学不出来。
5. 训练损失为纯 L1，未做尺度归一化（对应 `note.txt` 的两条待办）。

### 改动

**`preprocess.py`**
- 新增 `load_node_coordinates(coords_file, base_dir)`：解析 `node_id,x,y[,comment]` 坐标 CSV。
- 新增 `FeatureBuilder`：统一节点特征构造器，基础特征改为
  `[label_norm, degree_norm, x_norm, y_norm]` 重复填充至 `feat_dim`，把**真实坐标注入节点特征**。
- 新增 `_dijkstra_multi_source` 与 `_nearest_k_local_by_access`：按**全图最短路**选最近高速入口。
- 重写 `load_external_highway_context(...)`：加载时一次性预计算并缓存
  - `access_dist[N_full][K_highway]`：每个图节点到每个高速入口的真实最短路；
  - `highway_pair_dist[K_highway][K_highway]`：高速图内部两两最短路；
  - `node_coords`：高速节点用真实坐标，其余节点继承「图上最近高速入口」的坐标；
  - `feature_builder`。
- 重写 `build_synthetic_partition_inputs(...)`：
  - s/t 高速连接点按 `access_dist` 选最近 k 个（修复问题 3）；
  - 全局特征改用真实归一化坐标（修复问题 2）；
  - 新增返回 `highway_dist_feat = log1p([access(s), highway(入口s,入口t), access(t), 三者之和])`（缓解问题 4）。
- 删除已废弃的 `_build_node_features` / `_build_highway_nodes` / `_nearest_k_highways`（id 近似版）。

**`gnn.py` — `DistancePredictor`**
- 融合改为 **4 块** `[h_s_inner | h_t_inner | h_s_inter | h_t_inter]`，直接使用 InterGNN 的
  `st_virtual_emb`（s/t 两个虚拟节点嵌入的拼接），不再经 readout 塌缩（修复问题 1）。
- 新增构造参数 `use_highway_distance_feature` / `highway_distance_feat_dim`，
  `forward` 新增可选入参 `highway_dist_feat`，启用时拼入融合 MLP。

**`model.py` — `DistanceRegressionNet`**
- 透传上述两个新构造参数与 `highway_dist_feat`。

**`main.py`**
- 新增参数：`--coords_file`（默认 `cross15_coords.csv`）、
  `--loss_type {l1,log_l1,relative,huber}`（默认 `log_l1`）、
  `--disable_highway_distance_feature`。
- 新增 `build_loss(loss_type)`：`log_l1` 在 `log1p` 空间做 MAE、`relative` 为相对误差，跨尺度归一化（修复问题 5）。
- `run_distance_epoch` 改为接收外部传入的 `criterion`。
- 加载 context 时传入 `coords_file` 与 `weighted`；构建模型时传入距离特征开关。

**`infer_distance.py`**
- 新增 `--coords_file`、`--disable_highway_distance_feature`，与训练配置对齐。

### 接口/参数变化
- 模型融合层输入维度变化：旧 `2*inner_out + inter_out` → 新 `2*inner_out + 2*inter_out + [4]`。
- `run_distance_epoch(...)` 新增必填参数 `criterion`。
- `load_external_highway_context(...)` 新增参数 `coords_file`、`weighted`。
- `build_synthetic_partition_inputs(...)` 返回字典新增键 `highway_dist_feat`。

### 兼容性
- **旧 checkpoint 不兼容**：融合层维度已变，`saved_models/*.pt`（v0.1）无法直接 `load_state_dict`，需重新训练。
- 若要对齐论文「纯 GNN 嵌入 → MLP」的形态，可加 `--disable_highway_distance_feature`
  关闭距离特征（此时融合为纯 4 块嵌入）。
- 默认损失由 `l1` 改为 `log_l1`；如需复现旧行为请显式 `--loss_type l1`。

### 验证
- `python -m py_compile main.py preprocess.py gnn.py model.py infer_distance.py` 通过。
- 用独立脚本（不依赖 torch）在 `cross30_full` 上枚举全部 870 个可达节点对，验证
  `est = access(s) + highway(入口s,入口t) + access(t)`：
  - decomposition MAE = 0.0000，平均相对误差 = 0.0000（十字图上分解为精确解）；
  - id-近似最近高速 vs 图最短路最近高速：30 个节点中 14 个不一致 → 证实问题 3 的 bug 已修复。
- ⚠️ 端到端训练**未在本机执行**：当前环境（anaconda base）未安装 `torch / torch_geometric`，无法跑通完整训练。
  安装依赖后建议执行下方命令复跑确认。

### 运行示例
```bash
# 训练
python main.py --graph_file cross30_full --highway_file cross15.grf \
  --coords_file cross15_coords.csv --loss_type log_l1 --num_epoch 50

# 推理
python infer_distance.py --model_path saved_models/<ckpt>.pt \
  --graph_file ./cross30_full.grf --highway_file cross15.grf \
  --coords_file cross15_coords.csv --s 18 --t 27 --device cpu
```

### 已知局限 / 后续 TODO
- `highway_dist_feat` 在十字玩具图上是精确解，会让任务过于简单；
  在真实图（局部路网更复杂、入口非唯一）上才能体现 GNN 段的价值。建议补一个更复杂的数据集验证。
- Inner-GNN 仍用 2-hop ego 子图，未实现论文图中「按 highway 切分的 G1~G4 分区」；
  真正的分区机制留待 v0.3.0。
- 训练仍是逐样本更新（非 mini-batch），速度与梯度稳定性可进一步优化。
- Steiner Points 尚未建模。

---

## [v0.1.0] - 2026-04-24 — 距离回归主线初版（基线）
- 三段式 `DistancePredictor`（双 Inner-GNN + Inter-GNN + 融合 MLP），输出经 Softplus 保证非负。
- Dijkstra 采样 `(s,t,d)` 监督样本；按 `val_mae` 早停。
- 已知问题见 v0.2.0「动机」。

---

## 条目模板（复制到本节最上方使用）
```
## [vX.Y.Z] - YYYY-MM-DD — 简述
### 动机
### 改动
### 接口/参数变化
### 兼容性
### 验证
### 已知局限 / 后续 TODO
```
