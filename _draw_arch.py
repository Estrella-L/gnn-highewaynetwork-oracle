# -*- coding: utf-8 -*-
"""生成当前 NeurSC 网络结构信息流图 → docs/网络结构图_v1.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 找可用中文字体
def _pick_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            return c
    return "DejaVu Sans"

plt.rcParams["font.sans-serif"] = [_pick_font()]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(17, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# 颜色
C_DATA   = "#f0f4c3"
C_SAMPLE = "#e1f5fe"
C_INNER  = "#c8e6c9"
C_INTER  = "#bbdefb"
C_FUSION = "#e1bee7"
C_OUT    = "#ffe0b2"


def box(x, y, w, h, text, color, fs=9, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", linewidth=1.2,
                        facecolor=color, edgecolor="#333333", alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def region(x, y, w, h, title, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", linewidth=1.5,
                        facecolor=color, edgecolor="#666666", alpha=0.35)
    ax.add_patch(p)
    ax.text(x + 0.5, y + h - 0.9, title, ha="left", va="top", fontsize=11, weight="bold")


def arrow(x1, y1, x2, y2, style="-|>", color="#333333", lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                  linewidth=lw, linestyle=ls, mutation_scale=12,
                                  shrinkA=2, shrinkB=2))


# ============ Region 1: 预处理产物（顶部一行） ============
region(2, 82, 96, 16, "🗄  预处理产物（一次性算好，不参与训练）", C_DATA)
box(4,   85, 12, 8, ".off 地形网格\n|V|=N", C_DATA)
box(19,  85, 14, 8, "四叉树分区\nleaf_of[node]→leaf_id", C_DATA)
box(36,  85, 14, 8, "高速骨架图\nK 个边界点 + transit 边", C_DATA)
box(53,  85, 14, 8, "access_dist[N,K]\n节点→高速入口最短路", C_DATA)
box(70,  85, 12, 8, "highway_pair_dist[K,K]\n高速图两两最短路", C_DATA)
box(84,  85, 13, 8, "nearest_k_local[N,k]\n预计算最近 k 入口", C_DATA)

# 数据流箭头
arrow(16, 89, 19, 89)
arrow(33, 89, 36, 89)
arrow(50, 89, 53, 89)
arrow(60, 89, 74, 89)   # access → nearest_k
arrow(70, 89, 70, 89)   # 占位
arrow(83, 89, 84, 89)

# ============ Region 2: 单样本输入构造 ============
region(2, 62, 46, 18, "📥 单样本输入构造 (s, t)", C_SAMPLE)
box(4,  73, 8, 5, "查询点 s", C_SAMPLE)
box(4,  66, 8, 5, "查询点 t", C_SAMPLE)

box(15, 74, 14, 5, "s 叶子盒诱导子图\nx_s, edge_index_s", C_SAMPLE, fs=8)
box(15, 67, 14, 5, "t 叶子盒诱导子图\nx_t, edge_index_t", C_SAMPLE, fs=8)

box(32, 76, 15, 3, "s_connect_idx (查 nearest_k)", C_SAMPLE, fs=8)
box(32, 72, 15, 3, "s_global_feat=(x,y)_s", C_SAMPLE, fs=8)
box(32, 68, 15, 3, "t_connect_idx (查 nearest_k)", C_SAMPLE, fs=8)
box(32, 64, 15, 3, "t_global_feat=(x,y)_t", C_SAMPLE, fs=8)

arrow(12, 76, 15, 76)
arrow(12, 69, 15, 69)
arrow(12, 76, 32, 78)   # s → connect
arrow(12, 76, 32, 74)   # s → global
arrow(12, 69, 32, 70)   # t → connect
arrow(12, 69, 32, 66)   # t → global

# highway_dist_feat 特征（跨到融合层）
box(50, 68, 12, 4, "highway_dist_feat [4]\nlog1p([access_s, seg, access_t, sum])", C_SAMPLE, fs=7)

# ============ Region 3: 模型三段（并行） ============
region(2, 30, 96, 30, "🧠 DistancePredictor 前向（三路并行 → Fusion 汇合）", "#ffffff")

# Inner-s
region(3, 33, 22, 25, "🟩 InnerGNN — 局部段 (s)", C_INNER)
box(5, 50, 18, 4, "SAGEConv layer 1  (盒内消息传递)", C_INNER, fs=8)
box(5, 44, 18, 4, "ReLU + Dropout", C_INNER, fs=8)
box(5, 38, 18, 4, "SAGEConv layer 2  (2 跳感受野)", C_INNER, fs=8)
box(5, 33.5, 18, 3, "取 s 位置 → h_s_inner ∈ ℝ^O", C_INNER, fs=8)
arrow(14, 50, 14, 48)
arrow(14, 44, 14, 42)
arrow(14, 38, 14, 36.5)

# Inner-t
region(27, 33, 22, 25, "🟫 InnerGNN — 局部段 (t)  [共享权重]", C_INNER)
box(29, 50, 18, 4, "SAGEConv layer 1  (盒内消息传递)", C_INNER, fs=8)
box(29, 44, 18, 4, "ReLU + Dropout", C_INNER, fs=8)
box(29, 38, 18, 4, "SAGEConv layer 2  (2 跳感受野)", C_INNER, fs=8)
box(29, 33.5, 18, 3, "取 t 位置 → h_t_inner ∈ ℝ^O", C_INNER, fs=8)
arrow(38, 50, 38, 48)
arrow(38, 44, 38, 42)
arrow(38, 38, 38, 36.5)

# Inter
region(51, 33, 46, 25, "🟦 InterGNN — 高速段（含 s、t 虚拟节点）", C_INTER)
box(53, 53, 20, 4, "global_encoder MLP\n(x,y) → 虚拟节点特征", C_INTER, fs=8)
box(75, 53, 20, 4, "构造增广图：\nx_highway + s_virt + t_virt", C_INTER, fs=8)
box(53, 47, 42, 4, "加虚拟边：s_virt ↔ s_connect_idx,  t_virt ↔ t_connect_idx", C_INTER, fs=8)
box(53, 42, 42, 4, "SAGEConv layer 1", C_INTER, fs=8)
box(53, 37, 42, 4, "⚠  SAGEConv layer 2 (感受野=2，高速图直径远>2)", C_INTER, fs=8)
box(53, 33.5, 42, 3, "取 s_virt / t_virt → st_virtual_emb = [h_s_inter | h_t_inter] ∈ ℝ^{2O}", C_INTER, fs=8)
arrow(63, 53, 63, 51)
arrow(85, 53, 85, 51)
arrow(74, 47, 74, 46)
arrow(74, 42, 74, 41)
arrow(74, 37, 74, 36.5)

# ============ Region 4: Fusion 融合段 ============
region(15, 6, 70, 22, "🟪 Fusion 融合回归段", C_FUSION)
box(18, 20, 64, 6,
    "concat: [ h_s_inner | h_t_inner | h_s_inter | h_t_inter | highway_dist_feat(4) ]",
    C_FUSION, fs=9)
box(18, 13, 64, 5, "Fusion MLP  (dim → hidden → hidden/2 → 1)", C_FUSION, fs=9)
box(30, 8, 40, 4, "Softplus  →  ŷ = d̃(s,t)  ≥ 0", C_OUT, fs=10)
arrow(50, 20, 50, 18)
arrow(50, 13, 50, 12)

# ============ 跨区连接 ============
# Inner outputs → concat
arrow(14, 33.5, 30, 26, ls="-", color="#2e7d32")   # h_s_inner
arrow(38, 33.5, 38, 26, ls="-", color="#5d4037")   # h_t_inner
# Inter output → concat
arrow(74, 33.5, 60, 26, ls="-", color="#1565c0")   # st_virtual_emb
# highway_dist_feat → concat
arrow(56, 68, 70, 26, ls="--", color="#1565c0")

# nearest_k_local 表查 → connect_idx（虚线：查表式依赖）
arrow(90, 85, 47, 78, ls="--", color="#666")
arrow(90, 85, 47, 70, ls="--", color="#666")
# access_dist 查 → highway_dist_feat
arrow(60, 85, 56, 72, ls="--", color="#666")
# hpd 查 → highway_dist_feat
arrow(76, 85, 58, 72, ls="--", color="#666")

# 顶部标题
ax.text(50, 99.5, "NeurSC 网络结构信息流图  (v1 · 基线架构)",
        ha="center", va="top", fontsize=14, weight="bold")

# 底部图例
ax.text(2, 3, "图例:  → 张量流(实线)   -- ~ 查表/间接依赖   ⚠ 已知瑕疵点",
        fontsize=9, color="#444")
ax.text(2, 1, "特征维度约定: O = out_dim(默认 64), hidden = hidden_dim(默认 128), k = highway_k(默认 3)",
        fontsize=8, color="#666")

out_path = "docs/网络结构图_v1.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved: {out_path}")
