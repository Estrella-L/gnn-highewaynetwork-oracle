# -*- coding: utf-8 -*-
"""画 EP_high 解剖式消融图：A) 各配置 MRE 柱状；B) 2x2 因子矩阵（分区 x 欧氏残差）。"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

for fam in ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti', 'SimHei']:
    try:
        from matplotlib import font_manager
        if any(fam.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            break
    except Exception:
        pass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

M0, A1, A2, A3, A4 = 0.032943, 0.044030, 0.041770, 0.690887, 0.193060
rows = [
    ('M0 总体方法（分区+highway+三段式+欧氏残差）', M0, '#C44E52'),
    ('A1 去掉欧氏残差（保留分区+highway）', A1, '#4C72B0'),
    ('A2 去掉分区+highway（保留欧氏残差）', A2, '#55A868'),
    ('A3 两者都去掉（单 GNN 直接预测）', A3, '#8172B2'),
    ('A4 纯 3D 欧氏直线（无学习）', A4, '#999999'),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axes[0]
y = np.arange(len(rows))
ax.barh(y, [r[1] * 100 for r in rows], color=[r[2] for r in rows])
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel('测试集平均相对误差 MRE (%)  —— 越小越好')
ax.set_title('EP_high 解剖式消融：逐个拿掉技术后的效果')
for i, (name, v, _c) in enumerate(rows):
    tag = '（基准）' if i == 0 else '（%.2fx）' % (v / M0)
    ax.text(v * 100 + 1.0, i, '%.2f%% %s' % (v * 100, tag), va='center', fontsize=9)
ax.set_xlim(0, max(r[1] for r in rows) * 100 * 1.25)
ax.grid(axis='x', alpha=0.3)

# 2x2 因子矩阵
ax2 = axes[1]
grid = np.array([[M0, A1], [A2, A3]]) * 100
im = ax2.imshow(grid, cmap='RdYlGn_r', norm=matplotlib.colors.LogNorm(vmin=grid.min(), vmax=grid.max()))
ax2.set_xticks([0, 1]); ax2.set_xticklabels(['有欧氏残差输出', '无欧氏残差输出'], fontsize=11)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(['有地形分区\n+ highway 网络', '无分区（单 GNN）'], fontsize=11)
ax2.set_title('2x2 因子设计：两个技术的交互作用（MRE %）')
cell = [['M0\n%.2f%%' % grid[0, 0], 'A1\n%.2f%%' % grid[0, 1]],
        ['A2\n%.2f%%' % grid[1, 0], 'A3\n%.2f%%' % grid[1, 1]]]
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cell[i][j], ha='center', va='center', fontsize=14, weight='bold',
                 color='black' if grid[i, j] < 20 else 'white')
ax2.set_xlabel('输出参数化'); ax2.set_ylabel('网络结构')
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='MRE (%, 对数色标)')

fig.tight_layout()
p = os.path.join(OUT, 'A_消融对比图.png')
fig.savefig(p, dpi=150)
print('[plot]', p)
