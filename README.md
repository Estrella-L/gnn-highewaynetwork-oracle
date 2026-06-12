# NeurSC

基于图神经网络的**节点对最短路距离回归**项目。给定图上两个节点 `(s, t)`，
用「highway（高速骨架）分解」的三段式 GNN 预测它们之间的近似最短路距离 `d̃(s, t)`。

> 当前可运行主线只有 distance 任务。仓库中的 `Filtering / CoarsenNet / BasicCountNet /
> AttentiveCountNet` 等模块属于历史的子图匹配/计数路线，不在本主线内。

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

详细的逐模块说明见 `项目说明.md`，版本演进见 `CHANGELOG.md`。

---

## 目录结构

```
.
├── main.py                # 训练入口（distance 主线，输入 .off）
├── infer_distance.py      # 单对节点推理入口
├── build_highway.py       # .off → 全局图 + 真正的四叉树分区 + 高速骨干（对齐 EAR-Oracle）
├── preprocess.py          # 样本采样、特征构造、highway 上下文与距离预计算
├── gnn.py                 # InnerGNN / InterGNN / DistancePredictor
├── model.py               # DistanceRegressionNet 包装 + 评估指标
├── utils.py               # 辅助工具（含历史 .grf 读取，已不在主线使用）
├── sample_terrain.off     # 示例地形网格（来自 EAR-Oracle datasets/small，793 顶点）
├── saved_models/          # 训练得到的 .pt 权重
├── saved_results/         # 测试指标
├── saved_params/          # 运行参数快照
├── 项目说明.md            # 实现细节说明
└── CHANGELOG.md           # 版本更新文档
```

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
- 可用自带的 `sample_terrain.off`，或 `sample_terrains/` 下的其它示例地形（793~3696 顶点）。

---

## 使用方法

### 训练
```bash
python main.py \
  --off_file sample_terrain.off \
  --max_depth 3 --capacity 32 \
  --loss_type log_l1 \
  --num_epoch 50 \
  --device cpu
```
产物：`saved_models/<name>.pt`、`saved_results/<name>.txt`、`saved_params/<name>.txt`。
模型按验证集 `val_mae` 早停并保存最佳权重。

### 推理（单对顶点）
```bash
python infer_distance.py \
  --model_path saved_models/<ckpt>.pt \
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
python build_highway.py --off_file sample_terrain.off --max_depth 3 --capacity 32 --out_prefix terrain
```
输出 `terrain_partition.csv`（每个顶点的叶子编号、是否为高速节点、坐标），并打印分区/高速统计。
- 自适应四叉树（默认）：叶内点数 > `capacity` 且深度 < `max_depth` 才继续四分。
- 加 `--uniform` 则一律分到 `max_depth`（等大叶子，对应 EAR-Oracle 非自适应模式）。
- 细网格 + 小盒子会让高速节点占比偏高；用更小 `max_depth` / 更大 `capacity` 可得更稀疏的高速骨架。

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
| `--distance_samples` | 3000 | `(s,t)` 对数量上限；`<=0` 表示用全部唯一可达对（大图会自动设保护上限） |
| `--highway_k` | 3 | 每个端点连接的高速入口数 |
| `--inner_mode` | `partition` | Inner-GNN 子图：`partition`=四叉树叶子盒子图(对齐 G1~G4) / `ego`=2-hop ego(消融) |
| `--loss_type` | `log_l1` | `l1` / `log_l1` / `relative` / `huber`；后两者跨尺度归一化误差 |
| `--disable_highway_distance_feature` | 关 | 关闭 highway 分解距离特征（回到纯 GNN 嵌入融合） |
| `--early_stop_patience` | 10 | 早停耐心值 |
| `--device` | `cpu` | `cpu` 或 `cuda` |

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
- 训练为逐样本更新（非 mini-batch）；Steiner Points / WSPD spanner 尚未建模。

后续计划见 `CHANGELOG.md` 的 TODO 部分。
