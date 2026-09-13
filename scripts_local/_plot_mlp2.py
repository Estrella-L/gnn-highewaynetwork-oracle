# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'STHeiti', 'Songti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
OUT = "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local/docs/版本对比资料/figures"
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(0, 100); ax.set_ylim(-26, 104); ax.axis('off')

def box(x, y, w, h, label, color, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", linewidth=1.6, facecolor=color, edgecolor='#555555'))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18, color='#333333', lw=1.8))

# 标题
ax.text(50, 100, "融合 MLP（fusion_mlp）数据流 —— 从输入到预测（hidden=64, out=32 配置）", fontsize=16, fontweight='bold', ha='center')
ax.text(50, 93, "输入拼接 z = [ h_s_inner | h_t_inner | st_virtual | euclidean ]  → 129 维", fontsize=11, ha='center', color='#444')

# 输入块
box(2, 68, 19, 18, "h_s_inner\n[32]\ns 本地分区子图嵌入", "#e8f1fb")
box(27, 68, 19, 18, "h_t_inner\n[32]\nt 本地分区子图嵌入", "#e8f1fb")
box(52, 68, 21, 18, "st_virtual_emb\n[64]\ns/t 高速图虚拟节点", "#fdf0e2")
box(79, 68, 19, 18, "euclidean_feat\n[1]\n直线距离 log1p", "#e2efda")

# MLP 层
box(10, 38, 80, 12, "1️⃣ Linear(129 → 64)：z·W + b ，W=[64×129] 矩阵乘法 + 偏置 → 压成 64 维", "#f7d9d9")
box(10, 24, 80, 12, "2️⃣ ReLU：负数→0，保留正值（神经元只传正激活）", "#f7d9d9")
box(10, 10, 80, 12, "3️⃣ Linear(64 → 32) + ReLU：再压缩成 32 维并激活", "#f7d9d9")
box(10, -4, 80, 12, "4️⃣ Linear(32 → 1)：输出一个实数 raw（无界，MLP 的原始答案）", "#f7d9d9")

# 输入箭头
arrow(11, 68, 40, 50); arrow(36, 68, 52, 50); arrow(62, 68, 66, 50); arrow(88, 68, 76, 50)

# 锚结合
box(6, -22, 42, 14, "raw → 修正系数 = 0.5·tanh(raw)\n(限制在 ±0.5)", "#d9f0d9")
box(56, -22, 38, 14, "最终预测\ny_hat = 欧氏直线 × (1 + 修正系数)\n(0.5x ~ 1.5x 直线)", "#a9d18e")
arrow(30, 8, 27, -8)
arrow(52, 8, 62, -8)
ax.text(50, -33, "训练时：loss( y_hat, Dijkstra标签 ) → 反向传播更新 W 和 b", fontsize=11, ha='center', color='#333')
plt.tight_layout()
plt.savefig(f"{OUT}/04_fusion_mlp.png", dpi=150, bbox_inches='tight')
plt.close()
print("OK")
