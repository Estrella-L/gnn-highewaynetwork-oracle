# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'STHeiti', 'Songti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
OUT = "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local/docs/版本对比资料/figures"
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, label, color, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                 linewidth=1.6, facecolor=color, edgecolor='#555555'))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')

def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18,
                 color='#333333', lw=1.8))
    if label:
        ax.text((x1+x2)/2 + 1, (y1+y2)/2, label, fontsize=9, color='#555555')

# 顶部：四个输入块（当前配置维度：inner_out=32, inter_out=32, euclidean=1, highway关=0）
box(2, 78, 20, 14, "h_s_inner\n[32]\n(s 本地分区子图嵌入)", "#e8f1fb")
box(27, 78, 20, 14, "h_t_inner\n[32]\n(t 本地分区子图嵌入)", "#e8f1fb")
box(52, 78, 22, 14, "st_virtual_emb\n[64]\n(s/t 高速图虚拟节点)", "#fdf0e2")
box(79, 78, 20, 14, "euclidean_feat\n[1]\n(直线距离 log1p)", "#e2efda")

box(14, 52, 72, 12,
    "1️⃣ Linear(129 → 64)\n  z·W + b   (W=64×129 矩阵乘法 + 偏置)\n  → 128 维信息压缩成 64 维", "#f7d9d9")
box(14, 37, 72, 12,
    "2️⃣ ReLU\n  负数→0，保留正值 (激活函数)", "#f7d9d9")
box(14, 22, 72, 12,
    "3️⃣ Linear(64 → 32) + ReLU\n  再压缩：64 维 → 32 维，再激活", "#f7d9d9")
box(14, 7, 72, 12,
    "4️⃣ Linear(32 → 1)\n  输出一个实数 raw（无界，MLP 的原始答案）", "#f7d9d9")

arrow(12, 78, 25, 71)          # 输入汇聚箭头示意
arrow(37, 78, 50, 71)
arrow(63, 78, 60, 71)
arrow(89, 78, 75, 71)
ax.text(45, 88, "输入拼接 z = [ h_s_inner | h_t_inner | st_virtual | euclidean ]  共 129 维", fontsize=10.5, ha='center')

# 底部：锚结合
box(10, -14, 36, 16, "raw\n↓\n修正系数 = 0.5·tanh(raw)  有限制!(±0.5)", "#d9f0d9")
box(54, -14, 38, 16, "最终预测\ny_hat = 欧氏直线 × (1 + 修正系数)\n(0.5x ~ 1.5x 直线)", "#a9d18e")
arrow(50, 13, 28, 2, "raw 流出")
arrow(50, 3, 58, 2, "锚×修正")

ax.text(50, -24, "训练时：loss( y_hat, Dijkstra标签 )  → 反向传播更新 W 和 b", fontsize=10.5, ha='center', color='#333')
ax.set_title("融合 MLP（fusion_mlp）数据流 —— 从输入到预测", fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f"{OUT}/04_fusion_mlp.png", dpi=150, bbox_inches='tight')
plt.close()
print("OK")
