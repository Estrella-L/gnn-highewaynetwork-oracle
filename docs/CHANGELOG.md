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

## [v0.12.0] - 2026-07-10 — 8:1:1 划分验证（导出 train/val 点对 + check_split.py）

### 动机
实验计划第 5 步"8:1:1 节点/边重叠"需要验证 train/val/test 划分的比例与**无对级泄漏**。
此前 `main.py` 只导出 test 点对，无法核对三集的重叠。

### 改动
**`main.py`**：切分后**同时导出 train/val/test 三份**点对 CSV
（`<run>_{train,val,test}_pairs.csv`），test 文件名不变（`baseline.py` 仍兼容）。

**新增 `check_split.py`**（纯 Python）：读三份 CSV，报告
① 划分比例（是否 ≈ 8:1:1）；② **对级泄漏**（同一 (s,t) 跨集出现，必须为 0）；
③ 集合内重复对自检；④ 节点重叠（单图直推式下高重叠属正常，非泄漏）。

### 接口/参数变化
- `main.py` 新增产物 `<run>_train_pairs.csv`、`<run>_val_pairs.csv`（test 不变）。
- 新增脚本 `check_split.py`（`--run_prefix` 或分别 `--train/--val/--test`）。

### 兼容性
- 不改训练逻辑与模型；仅多导出两份 CSV。旧 run（只导了 test）需重训或重跑采样才能全量校验。

### 验证
- `main.py` / `check_split.py` `ast` 解析通过。
- 本地合成 CSV 冒烟：正确报出划分比例、并检出故意植入的 1 处对级泄漏（train∩test=1）。

---

## [v0.11.0] - 2026-07-10 — 新增点对跨分区/同分区比例分析脚本

### 动机
需要用**实测**代替估算，确认采样的 `(s,t)` 点对里跨分区/同分区各占多少
（此前对"跨分区占比≈94%"只是均匀假设下的估算，真实地形叶子可能不均衡）。

### 改动
**新增 `analyze_pairs.py`**（纯 Python，无需 torch）：
- 读 `outputs/cache/<key>_partition.csv`（每点 leaf_id）→ 计算叶子大小分布、`Σpᵢ²`、
  全图所有对中同分区的**精确**比例 `Σ C(nᵢ,2)/C(N,2)`；
- 可选读 `<run>_test_pairs.csv` → 统计**实际采样**点对中同分区/跨分区的实测比例。

### 验证
- `ast` 解析通过；本地 900 点规则网格实测：16 叶（49~64），同分区 6.20%，跨分区 93.80%
  （均匀图下与理论 1/16 吻合）。真实地形（如 EP_low）需在云端用其 partition/test_pairs CSV 实测。

### 已知局限 / 后续 TODO
- 均匀网格 intra≈6% 已证；真实地形因顶点分布不均，intra 可能更高，需实测确认。
- 如需按分区控制采样，可加 `--pair_mode {any,cross,intra}`（见 Roadmap 的分层采样相关项）。

---

## [v0.10.0] - 2026-07-10 — 新增 LR scheduler + 训练日志目录 + 超参数消融文档

### 动机
超参数消融实验（LR / dropout / scheduler / ε）需要学习率调度支持；同时需要一个统一存放
不同参数/版本训练日志的地方，以及一份消融实验蓝图文档。

### 改动
**`main.py`**
- 新增 `--lr_scheduler {none,plateau}` + `--lr_patience` / `--lr_factor` / `--min_lr`：
  plateau 用 `torch.optim.lr_scheduler.ReduceLROnPlateau`，按 `val_mae` 触发降 LR。
- 每轮日志追加 `lr=...`；降 LR 时打印 `lr reduced: a -> b`。

**目录**：新增顶层 `logs/`（训练日志归档，手动 `tee` 保存，跨参数/版本对比）；
首个 Baseline 日志已归档其中。

**文档**：新增 `docs/训练文档2.md`（超参数消融实验蓝图：Baseline + Exp-1~5，含可执行命令与结果表模板）。

### 接口/参数变化
- `main.py` 新增 `--lr_scheduler / --lr_patience / --lr_factor / --min_lr`（默认 `none`，行为不变）。

### 兼容性
- 默认 `--lr_scheduler none`，不启用时训练行为与之前完全一致；模型结构与 checkpoint 不变。

### 验证
- `main.py` `ast` 解析通过。scheduler 仅在 `plateau` 时创建，`step(val_mae)` 在每轮 val 后调用。

### 已知局限 / 后续 TODO
- Exp-4/5（ε 稀疏化 / 测试集重构）仍需先实现 WSPD spanner（见 Roadmap / `项目说明.md` §9.1）。
- 数据泄漏检查脚本（8:1:1 节点/边重叠）尚未实现。

---

## [v0.13.0] - 2026-07-10 — 预处理内存优化（修复大图 depth=3 OOM）

### 动机
EP_low（|V|=164238）在 `--max_depth 3` 下预处理阶段被系统 OOM killer 杀死（CPU RAM 爆，非 GPU）。
根因：`build_pipeline_inputs` 里两个稠密距离结构 `dist_from_boundary`（字典推导同时持有 K 份长度 N
的数组）与 `access_dist`（N×K，且是前者的转置、完全冗余），均用 Python float list 存储（每个数 ~32B）。
K（高速节点数）随分区变细快速增长：depth=2 K≈4856 峰值 ~51GB（勉强能跑），depth=3 K 翻几倍 →
峰值 ~130GB → OOM。

### 改动
**`build_highway.py:build_pipeline_inputs`**
- **流式化边界 Dijkstra**：不再用字典推导一次性持有 K 份全长距离；改为每个高速点算一次即
  (1) 填 `access_dist` 对应列、(2) 就地生成该盒 transit 边，随即丢弃 → 峰值从 O(K×N) 降到 O(N)。
- **float32 numpy 存储**：`access_dist` 与 `highway_pair_dist` 从 Python list-of-list 改为
  `np.float32` 数组（每数 4B，省约 8×）。
- 效果（depth=3 估算）：峰值从 ~130GB 降到 ~9GB（约 14×）。

### 接口/参数变化
- 无 CLI 变化。`access_dist` / `highway_pair_dist` 由嵌套 list 变为 numpy float32 数组（内部表示）。

### 兼容性
- 语义等价（距离值不变，仅 float32 舍入，对 nearest-k 选择与 log1p 距离特征无实质影响）。
- 消费端（`_nearest_k_local_by_access` 排序、距离特征算术、`highway_pair_dist[i][j]` 索引、
  `== inf` 比较）经本地 numpy 兼容性测试全部通过。
- **缓存需重算**：内部表示变了，旧 `.pt` 缓存不应复用（用 `--no_cache` 或删 `outputs/cache/` 重算）。

### 验证
- `build_highway.py` `ast` 解析通过；numpy float32 消费逻辑本地单测通过。
- ⚠️ 未在真实大图上端到端跑（本地无 torch/大图）。**强烈建议先用 depth=2 重训一次**，
  确认 test 指标与旧结果（rel_err≈20.01%）一致（等价性验证），再上 depth=3。

### 已知局限 / 后续 TODO
- 若 depth=3 的 access_dist（~8GB）仍偏大，可进一步"只存每节点最近 k 个高速入口"（[N,k] 而非 [N,K]）
  —— 需改 nearest-k 的存取方式，是下一步更大的内存优化。
- transit 边仍是盒内全连接 O(K²/盒)，大图上 edge_w 也占内存 + InterGNN 变重 → 见 WSPD spanner（Roadmap）。

---

## 可提升工作（Roadmap，未实现）

按优先级记录尚未落地、但有价值的改进方向：

- **高速网络用 WSPD spanner 稀疏化（可选）**：当前 `build_highway_edges` 的盒内
  transit 边是**全连接**（盒内高速点两两相连），边数 `O(K²/盒数)`，大图上爆炸——EP_low
  （高速点 4856）盒内 transit 边约 **70 万条**，是每轮 ~570s 的主因。应改为 EAR-Oracle 的
  **WSPD（Well-Separated Pair Decomposition）spanner**，用参数 **ε（近似精度）** 控制边数：
  产出 (1+ε)-spanner，边数降到 `O(K/ε²)`（大图少 30~50 倍），ε 即"精度—边数/速度"旋钮。
  过渡方案可先做 `--transit_k` 的 k-近邻稀疏化验证收益。详细实现拆解见 `项目说明.md` 第 9 节。
- **按距离分层采样监督**：当前 `build_distance_samples` 对节点对**均匀随机抽样**，导致距离标签集中在
  中等距离、极近/极远样本稀少；而评估指标 `relative_error` 对小距离最敏感。可加一个可选开关：
  把可达距离分档（如按分位数），每档抽相近数量，使各距离段训练更均衡。默认保留现有均匀采样以兼容。
- **highway 分解强基线**：`baseline.py` 补 `access(s)+highway(入口s,入口t)+access(t)` 这条非学习上界，
  正式回应"GNN 相对分解本身有多少增益"（README Q4）。
- **多种子 / 跨图评估**：固定配置多种子重复报均值±方差；多张地形交叉验证（当前单种子单次）。
- **对称化输出**：强制 `d̃(s,t)=d̃(t,s)`（对称读出），并报告对称性违反度（README Q5）。
- **分区/高速可视化**：把分区盒子 + 高速点画成 PNG，便于审查与论文配图（当前仅 CSV 审查文件）。
- **Steiner 点 / Snell 加权测地距离**：贴近 EAR-Oracle 的更精确**监督信号**（依赖几何，较重）。
  （注：WSPD spanner 属高速网络稀疏化，已单列为上面的高优先级项，不在此监督精度条目内。）

---

## [v0.9.0] - 2026-06-26 — 记录训练耗时（预处理 / 每轮 / 总时长）

### 动机
此前只有运行时间戳命名，没有任何耗时记录，无法判断"等待主要花在 CPU 预处理还是 GPU 训练"。
租 GPU 做实验时这点很关键。

### 改动
**`main.py`**
- 预处理（`build_pipeline_inputs_cached`）计时：打印 `preprocess(分区+高速/缓存) done in X.XXs`。
- 每轮训练日志追加 `time=XX.XXs`（单轮耗时）。
- 训练结束打印 `timing: preprocess=...s, train=...s, best_epoch=...`。
- 结果文件 `outputs/results/<run>.txt` 增加字段：`best_epoch / preprocess_seconds / train_seconds`。

**`docs/训练文档.md`**：示例图规模从 1600 点改为 **900 点（`--grid 30`，`terrain_grid_30x30_900v.off`）**，
新增「关于耗时记录」一节，顶点编号上限与推理 `--s/--t` 同步调整。

### 接口/参数变化
- 无新增 CLI 参数；仅新增日志与结果文件字段（向后兼容）。

### 兼容性
- 不影响训练逻辑、模型结构与 checkpoint；纯计时打点 + 多写几个结果字段。

### 验证
- `main.py` `ast` 解析通过。计时用 `time.time()` 差值，无外部依赖。

---

## [v0.8.0] - 2026-06-26 — 高速上下文磁盘缓存 + 审查文件导出

### 动机
分区 + 每个高速节点的全图 Dijkstra 预处理在每次训练/推理启动时都重算，大图要等数十秒；
反复实验时纯属浪费。同时需要可供**审查**的分区/高速结构记录（可追溯性）。

### 改动
**`build_highway.py`**
- 新增 `build_pipeline_inputs_cached(...)`：按 `图名 + .off内容指纹 + max_depth/capacity/模式/feature_dim` 作缓存键，
  首次计算后用 `torch.save` 存 `outputs/cache/<key>.pt`（CPU 张量，cpu/gpu 通用），命中则直接加载、
  跳过分区与 Dijkstra 预处理。`cache_dir=None` 时禁用。
- 新增审查导出：`<key>_partition.csv`（node, leaf_id, is_highway, x, y）与
  `<key>_highway_edges.csv`（高速边还原成全局 id，无向去重）。

**`main.py` / `infer_distance.py`**
- 改用 `build_pipeline_inputs_cached`；新增 `--cache_dir`（默认 `outputs/cache`，按项目根解析）与 `--no_cache`。

**`.gitignore`**：忽略 `outputs/cache/*.pt`、`outputs/cache/*.csv`（派生物）。

### 接口/参数变化
- `main.py` / `infer_distance.py` 新增 `--cache_dir`、`--no_cache`。

### 兼容性
- 模型结构与 checkpoint 不变；缓存仅是预处理结果的存档，命中与否产出的上下文一致（确定性）。
- 缓存键含四叉树参数与 feature_dim，参数变更自动用不同缓存，不会读到过期结果。

### 验证
- 8 个 `.py` 全部 `ast` 解析通过；`FeatureBuilder` 及上下文均为可 pickle 的纯数据。
- ⚠️ 缓存读写依赖 torch，本机未跑；GPU/有 torch 环境下首跑会生成 `.pt` 与两份审查 CSV，二次跑应打印
  `[cache] 命中...`。

### 已知局限 / 后续 TODO
- 缓存键含 `.off` 内容指纹（md5 前 8 位）+ 四叉树参数 + feature_dim，**同名但内容不同的图不会误命中**；
  改图/改参都会自动用新缓存。
- 可视化（把分区/高速画成 PNG）尚未做，目前审查靠 CSV。

---

## [v0.7.0] - 2026-06-26 — 训练支持 mini-batch（提升 GPU 利用率）

### 动机
此前训练是**逐样本**前向/反向/优化器步进，GPU 利用率低、租用 GPU 收益有限。本版本引入 mini-batch：
批内把多个样本的子图与（复制的）高速图合并成一张大图做**一次**消息传递与优化器步进。

### 改动
**`gnn.py`**
- `InnerGNN.forward_batch(x_list, edge_index_list, query_idx_list)`：把 B 个子图按节点偏移合并成
  一张不连通大图，单次消息传递后按偏移后的查询索引取出各样本嵌入，返回 `[B, out]`。
- `InterGNN.forward_batch(x_highway, edge_index_highway, s_global_feats, t_global_feats, s_connect_list, t_connect_list)`：
  共享高速图复制 B 份、各加 s/t 两个虚拟节点（特征由 `global_encoder` 批量编码），合并成大图单次消息传递，
  返回各样本 s/t 虚拟节点拼接 `[B, 2*out]`。
- `DistancePredictor.forward_batch(samples)`：组合上面两段 + 批量拼接 highway 距离特征 → Fusion MLP → `[B]`。
- 原单样本 `forward` 保留不变（推理与 `--batch_size 1` 仍走原语义）。

**`model.py`**：`DistanceRegressionNet.forward_batch` 透传到 backbone。

**`main.py`**
- 新增 `--batch_size`（默认 16；`1` = 旧逐样本行为）。
- `run_distance_epoch` 改为按 `batch_size` 分块：每块构造各样本输入 → `forward_batch` 单次前向 →
  批损失 → 一次 `backward`/`optimizer.step()`；指标仍在全量预测上汇总。

### 接口/参数变化
- `main.py` 新增 `--batch_size`（默认 16）。
- 模型新增方法 `forward_batch`（`gnn.DistancePredictor` 与 `model.DistanceRegressionNet`）。

### 兼容性
- 模型结构与 checkpoint **不变**；`forward_batch` 与单样本 `forward` 共享同一套权重。
- `--batch_size 1` 复现旧逐样本更新语义；`infer_distance.py` 仍用单样本 `forward`，不受影响。

### 验证
- 8 个 `.py` 全部 `ast` 解析通过。
- ⚠️ **端到端未在本机执行**：本机无 `torch / torch_geometric`。请在 GPU 机上做冒烟测试确认：
  ```bash
  python main.py --off_file sample_terrain.off --max_depth 2 --capacity 64 \
    --distance_samples 40 --num_epoch 1 --batch_size 8 --device cuda
  ```

### 已知局限 / 后续 TODO
- InterGNN 批量化对每个样本复制一份高速图（节点数 ≈ B×K），batch 过大时显存上升；K 大时建议适中 batch。
- 启动预处理（四叉树分区 + 边界点 Dijkstra）仍为纯 CPU，不随 GPU/batch 加速。
- `baseline.py` 暂未含 highway 分解强基线；多种子方差、跨图验证仍待补。

---

## [v0.6.0] - 2026-06-26 — 工程目录集成化（data / outputs / docs）

### 动机
此前源码、数据、产物、文档全平铺在根目录，难以查找、产物无固定归处。本版本按职责分目录，
让「每个生成文件都有地方存放和查找」，同时保持源码在根目录、运行命令基本不变。

### 改动
**目录结构**（源码 `.py` 仍在根目录，命令不变）：
- `data/`：输入地形。`data/sample_terrain.off`、`data/sample_terrains/`（真实地形）、
  `data/generated/`（`generate_terrain.py` 输出）。
- `outputs/`：运行产物。`outputs/models/`(.pt) / `outputs/results/`(指标 + `<run>_test_pairs.csv` + 基线) / `outputs/params/`(参数快照)。
- `docs/`：`项目说明.md` / `流程文档.md` / `CHANGELOG.md` / `note.txt`（`README.md` 留在根目录）。

**路径解析（关键）**：`main.py` / `infer_distance.py` / `baseline.py` / `generate_terrain.py`
统一以**脚本所在目录（项目根）**为基准解析路径——
- `--file_folder` 默认 `data`，故 `--off_file sample_terrain.off` 解析为 `data/sample_terrain.off`；
- 训练产物固定写入 `outputs/{models,results,params}/`；
- `generate_terrain.py` 默认输出 `data/generated/`。
因此**在任意工作目录运行都能正确定位**，不再依赖当前目录。

**`.gitignore`**：忽略路径同步为 `outputs/**`、`data/generated/*.off`。

### 接口/参数变化
- `--file_folder` 默认 `./` → `data`（main/infer/baseline）；`generate_terrain.py --out_dir` 默认 `data/generated`。
- 训练产物目录 `saved_models|saved_results|saved_params/` → `outputs/{models,results,params}/`。

### 兼容性
- 模型结构与 checkpoint **不受影响**；训练/推理逻辑不变，仅路径默认值与产物位置变化。
- 旧的 `saved_*` 目录内容已迁移至 `outputs/`；历史 `saved_params/` 快照现位于 `outputs/params/`。
- 云端命令 `python main.py --off_file sample_terrain.off ...` 仍可直接用（解析到 `data/`）。

### 验证
- 8 个 `.py` 全部 `ast` 解析通过。
- `generate_terrain.py` 从子目录运行，正确输出到 `data/generated/`（root-relative 解析生效）。
- 全局检索确认源码与文档已无遗留旧路径引用（历史 changelog 条目保留原貌）。

### 已知局限 / 后续 TODO
- `build_highway.py` CLI 仍按当前目录解析 `--off_file`（用 `data/sample_terrain.off`），未做 root-relative。
- `baseline.py` 暂未含 highway 分解强基线；多种子方差、跨图验证仍待补。

---

## [v0.5.0] - 2026-06-26 — 新增地形生成器与非学习基线 + 仓库瘦身

### 动机
1. 云端实验用的规则网格地形（`terrain_grid_NxN`）没有同步到仓库，需要可复现的生成脚本。
2. 缺少非学习基线，无法回答「GNN 比简单方法好多少」（README Q4）。
3. 仓库里残留历史子图匹配/计数路线的文件，与 distance 主线无关，影响可读性。

### 改动
**新增 `generate_terrain.py`**（仅依赖 numpy）：生成规则网格地形 `.off`，带真实起伏（山丘/山谷/带缺口山脊），
使「图最短路 ≠ 直线距离」，从而适配 highway 分解算法。`--grid` 支持一次生成多种规模，
`--mode {flat,smooth,mountains,ridges,mixed}` 控制高度场，命名与云端一致（`terrain_grid_NxN_<V>v.off`）。

**新增 `baseline.py`**（纯 Python，无需 torch）：在 `main.py` 导出的 test 点对文件上评估三个朴素基线
（`mean_constant` 最优常数 / `euclidean_2d` / `euclidean_3d`），直接与模型的 `test_*` 指标并排比较。
test 集**唯一来源是导出的点对文件**（`--test_pairs_file` 必填），不依赖种子复现。

**`main.py`**：切分后导出 test 查询点对 + 真实距离到 `saved_results/<run_name>_test_pairs.csv`
（最小版 `s,t,true_distance`），用于审计与基线对齐，消除跨环境复现 test 集的隐患。

**全部 `.py` 文件**：在首行加一句用途注释，便于一眼区分主线/历史。

**仓库瘦身（删除历史文件与死代码）**：删除 `filtering.py`、`graph_coarsen.py`、`graph_operation.py`、
`viz_grf.py`、`utils.py`（历史子图匹配/计数与 `.grf` 路线，主线不依赖）；
连带从 `preprocess.py` 移除仅被 `graph_coarsen` 使用的死代码类 `SampleSubgraph`
及其对 `graph_operation.graph_depth` 的导入。
进一步清理 `model.py` 的历史计数网络类（`BasicCountNet`/`AttentiveCountNet`/`WasserstainDiscriminator`/
`QErrorLoss`/`QErrorLikeLoss`/`CoarsenNet`），以及 `gnn.py` 中仅被它们使用的 `GIN`/`GAT` 类；
`model.py` 现仅保留 `DistanceRegressionNet` + `compute_distance_metrics`，`gnn.py` 仅保留三段式网络。

### 接口/参数变化
- 新增脚本 `generate_terrain.py`、`baseline.py`（均带 CLI）。
- `main.py` 新增产物 `saved_results/<run_name>_test_pairs.csv`（不改变任何训练逻辑）。
- `preprocess.py` 不再导出 `SampleSubgraph`、不再 import `graph_operation`。
- `model.py` 不再导出历史计数网络类；`gnn.py` 不再导出 `GIN`/`GAT`（主线无引用）。

### 兼容性
- 模型结构与 checkpoint **不受影响**；训练/推理行为不变。
- 若有外部代码引用已删除的历史模块或 `SampleSubgraph`，需自行移除（主线无引用）。

### 验证
- 全部保留的 `.py`（baseline/build_highway/generate_terrain/gnn/infer_distance/main/model/preprocess）
  通过 `ast` 解析。
- `generate_terrain.py` 本地生成 100/400 点 `.off`，经 `build_highway.py` 读取分区成功
  （400 点 → 16 叶、204 高速点，与云端结构一致）。
- `baseline.py` 本地（无 torch）在小型 test 点对文件上跑通，三个基线指标输出正常。
- 全局搜索确认无任何残留文件引用已删除模块。

### 已知局限 / 后续 TODO
- `baseline.py` 暂未含 highway 分解强基线（`access+highway+access`），后续按需补充以正式回应审稿。
- 分区每次训练重算（无缓存）；多种子方差、跨图验证仍待补。

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
