# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'STHeiti', 'Songti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT = "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local/docs/figures"
os.makedirs(OUT, exist_ok=True)

# ================= 图1：版本演进时间线 =================
fig, ax = plt.subplots(figsize=(14, 5.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
cards = [
    (1,  "仓库版\nv0.15.0 (2026-07-17)", "#c0c4cc",
     ["· 只有 direct（从零回归）", "· 无任何残差修正", "· 无预测模式参数", "· 缓存键无 tk", "· torch.load 旧写法"]),
    (35, "Highway版\n(2026-08-19)", "#9dc3e6",
     ["· 新增 highway_residual ★", "· 基线=高速分解(依赖已有工作)", "· 采样策略3选1", "· 分组误差评估", "· weights_only=False"]),
    (69, "欧式距离版（现在这版）\n(2026-08-21)", "#a9d18e",
     ["· 新增 euclidean_residual ★★", "· 基线=两点3D直线(不依赖已有)", "· 有界修正0.5·tanh", "· 本地91配置调参验证", "· 云平台16万点训练中"]),
]
for x, title, color, items in cards:
    box = FancyBboxPatch((x, 18), 30, 50, boxstyle="round,pad=1.2",
                         linewidth=2, edgecolor='#404040', facecolor=color, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + 15, 62, title, ha='center', va='center', fontsize=12.5, fontweight='bold')
    y = 48
    for it in items:
        ax.text(x + 2.5, y, it, ha='left', va='center', fontsize=9.5)
        y -= 5.4
for x in (32.5, 66.5):
    ax.add_patch(FancyArrowPatch((x, 43), (x + 3.2, 43), arrowstyle='-|>', mutation_scale=22, color='#333333', lw=2.2))
ax.text(50, 7, "每一步都在上一版基础上增强 —— 欧式距离版 = 仓库版 + Highway版的全部 + 欧氏基线残差", ha='center', fontsize=11.5, color='#555555', style='italic')
ax.set_title("三个版本演进", fontsize=18, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(f"{OUT}/01_version_evolution.png", dpi=150, bbox_inches='tight')
plt.close()

# ================= 图2：三种预测模式 =================
fig, axes = plt.subplots(3, 1, figsize=(14, 8.5))
modes = [
    ("仓库版 · direct（纯回归）", "y = softplus( MLP(x) )", "无基线, 模型硬猜完整距离", "#dae3f3", "本地实测 57%~99%（几乎崩坏）", "无限制(可能跑飞)"),
    ("Highway版 · highway_residual", "y = 高速分解距离 × (1 + 0.5·tanh(MLP))", "基线=高速网络（依赖已有工作 EAR-Oracle）", "#fde9d9", "强基线但依赖高速已有工作", "0.5x ~ 1.5x 基线"),
    ("欧式距离版 · euclidean_residual ★", "y = 欧氏直线距离 × (1 + 0.5·tanh(MLP))", "基线=两点 3D 直线距离（任何图都有，不依赖已有工作）", "#e2efda", "本地实测 1.67%（最优）", "0.5x ~ 1.5x 基线"),
]
for ax, (name, formula, base, color, note, rng) in zip(axes, modes):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    box = FancyBboxPatch((1, 6), 98, 88, boxstyle="round,pad=0.8", linewidth=1.8, facecolor=color, edgecolor='#666666')
    ax.add_patch(box)
    ax.text(4, 84, name, fontsize=14, fontweight='bold')
    ax.text(4, 60, formula, fontsize=17, fontweight='bold', color='#1a1a1a')
    ax.text(4, 38, base, fontsize=11, color='#333333')
    ax.text(4, 16, note, fontsize=11, color='#8b0000' if '崩坏' in note else '#1a6b1a', fontweight='bold')
    ax.add_patch(mpatches.Rectangle((66, 56), 20, 6, facecolor='#cccccc', edgecolor='#888888'))
    ax.text(76, 63.5, "修正范围", fontsize=8.5, ha='center', color='#555')
    ax.text(66, 48, rng, fontsize=10, color='#444444')
plt.tight_layout()
plt.savefig(f"{OUT}/02_prediction_modes.png", dpi=150, bbox_inches='tight')
plt.close()

# ================= 图3：功能矩阵 =================
feats = [
    ("欧氏基线+GNN修正 (euclidean_residual)",      False, False, True),
    ("有界修正 0.5·tanh",                          False, False, True),
    ("highway_residual 模式",                       False, True,  True),
    ("预测模式可选 (direct/hw/euclidean)",          False, True,  True),
    ("3D坐标注入 (node_coords3d)",                  False, False, True),
    ("采样策略 (random/oracle_mix/distance_gap)",   False, True,  True),
    ("transit_k 稀疏化 + tk缓存键",                  False, True,  True),
    ("torch.load(weights_only=False)",              False, True,  True),
    ("分组误差评估 (same/cross/short/long)",        False, True,  True),
    ("deepcopy 冻结最佳权重",                       False, True,  True),
    ("baseline: highway_decomp 强基线",             False, True,  True),
    ("scipy 分块Dijkstra + 采样缓存",               True,  True,  True),
    ("nearest-k 预计算",                            True,  True,  True),
    ("8:1:1 划分 + check_split",                    True,  True,  True),
    ("LR plateau 调度",                             True,  True,  True),
]
labels = ["仓库版\nv0.15.0", "Highway版\n8-19", "欧式距离版\n8-21 ★"]
fig, ax = plt.subplots(figsize=(9.5, 10.5))
data = [[1 if f[i+1] else 0 for i in range(3)] for f in feats]
ax.imshow(data, cmap=plt.cm.Greens, aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=12.5)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([f[0] for f in feats], fontsize=10.5)
for i in range(len(feats)):
    for j in range(3):
        v = "✓" if data[i][j] else "✗"
        color = "#14532d" if data[i][j] else "#b91c1c"
        ax.text(j, i, v, ha='center', va='center', fontsize=13, fontweight='bold', color=color)
ax.set_title("功能清单对比（绿✓=已有）", fontsize=15, fontweight='bold', pad=12)
ax.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/03_diff_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("OK")
