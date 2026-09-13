# baseline_ep_high：三篇 baseline 在 EP_high 上的独立对比工程

目标：用**三篇论文的方法**（GeGnn / NeuroGF / LiteGE）跑**我们的 terrain 主数据集 EP_high**，
与我们的三段式方法在同一份真值、同一套切分上比较。本工程**不依赖**主项目里被改过的 `main.py`，
只从原版核心代码借用 `build_highway.load_off` 读网格。

## 一、协议（与主实验完全对齐）

| 项目 | 取值 |
|---|---|
| 网格 | EP_high.off（1,392,236 顶点） |
| 图 | 三角网格无向去重边，边权 = 3D 欧氏边长 |
| 真值标签 | flip-out 精确表面测地距离（surface_random.csv，50,000 对） |
| 切分 | 80/10/10，random.seed(42)，按 CSV 行序（与 preprocess.split_distance_dataset 等价） |
| 指标 | MRE = mean(|pred-true|/true)、MAE、RMSE |
| 纯欧氏参照 | 同一 test 集上的 3D 直线距离 |

## 二、三篇的实现与适配（论文对比表中需标注）

| 方法 | 官方要点 | 本实现 | 适配说明 |
|---|---|---|---|
| GeGnn | GeoConv（max 聚合 + 边长/相对位置消息）、GroupNorm(4)、U-Net 层级池化 | 全分辨率 GeoConv + GroupNorm，**保持官方宽度 256 / 4 层**，用梯度检查点控显存 | 省略 hgraph 层级池化（在全分辨率上做）——与消融实验一致，需在论文中标注 |
| NeuroGF | 逐点 lifting + GDF 差分头 | 与官方一致（不使用图结构） | 无 |
| LiteGE | CoordMLP + UDF-PCA 形状描述子 | CoordMLP 同构；UDF-PCA 适配为「按 (x,y) 切 patch 当形状集」 | 单场景开曲面没有"一批形状"，patch 化适配需标注 |

输出参数化（对齐主实验的可比性设计）：
- `native`：论文原样线性输出
- `euclidean_residual`：d_3D · (1 + 0.5·tanh(raw))，与我们的方法同构

## 三、跑法

    # 单方法
    python train.py --off <EP_high.off> --labels_csv <surface_random.csv> \
        --arch gegnn --out_mode euclidean_residual --epochs 20 --out_json results/gegnn_er.json

    # 全部（3 方法 × 2 输出参数化）+ 自动汇总
    bash run_all.sh

## 四、显存与速度

- EP_high：V=1.39M、E≈4.2M。GeGnn 宽度 256 时每层消息张量 E×256 ≈ 4.3GB，
  用 `torch.utils.checkpoint` 逐层重算把峰值压到 2 层量级，24GB 卡可跑；`--no_checkpoint` 会更吃显存。
- NeuroGF / LiteGE 为逐点 MLP，显存主要来自节点嵌入（1.39M×256 ≈ 1.4GB）。
- 标签加载后按 (s,t) 索引取嵌入，无需对每个 pair 前向整图。
