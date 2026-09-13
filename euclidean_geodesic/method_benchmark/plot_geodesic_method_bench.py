# -*- coding: utf-8 -*-
"""表面距离方案：误差与成本对比图。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

CJK = None
for f in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        CJK = f
        break
if CJK:
    plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
T = (lambda zh, en: zh) if CJK else (lambda zh, en: en)

out_dir = os.path.expanduser("~/Desktop/GNN_实验图表/figures")
os.makedirs(out_dir, exist_ok=True)

labels = [
    T("图折线Dijkstra(现做法)", "graph Dijkstra (current)"),
    T("中点细分+Dijkstra", "midpoint subdivision"),
    T("Heat Method", "heat method"),
    T("精确MMP(pygeodesic)", "exact MMP"),
    T("flip-out精确(推荐)", "flip-out exact (best)"),
]
errs = [7.95, 7.95, 1.04, 0.063, 0.0625]
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#2ecc71"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

ax = axes[0]
bars = ax.bar(range(len(labels)), errs, color=colors)
for b, v in zip(bars, errs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18, f"{v:.3f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
ax.set_ylabel(T("与解析真值(球面大圆)的平均相对误差 %", "mean rel err vs analytic sphere (%)"))
ax.set_title(T("球面解析验证：各方案精度（越低越好）", "Sphere analytic test: accuracy (lower is better)"))
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
cost_labels = [T("图折线(现做法)", "graph"), T("Heat Method", "heat"), T("精确MMP", "MMP"), T("flip-out精确", "flip-out")]
hours = [0.87 * 50000 / 32 / 3600, (46.7 / 3) * 50000 / 32 / 3600, 118.6 * 50000 / 32 / 3600, 0.589 * 50000 / 32 / 3600]
bars = ax.bar(cost_labels, hours, color=["#e74c3c", "#f1c40f", "#3498db", "#2ecc71"])
for b, h in zip(bars, hours):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.06, f"{h:.2f} h", ha="center", fontsize=9)
ax.set_ylabel(T("EP_high 5万对耗时(32核并行, 小时)", "EP_high 50k pairs, 32 cores (hours)"))
ax.set_title(T("成本对比：flip-out 精确约 15 分钟", "Cost: exact flip-out about 15 min"))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = os.path.join(out_dir, "12_表面距离方案_精度与成本对比.png")
plt.savefig(p, dpi=160)
plt.close()
print("saved", p)

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ds = [T("EP_low (16.4万点)", "EP_low (164k)"), T("EP_high (139万点)", "EP_high (1.39M)")]
mean_bias = [9.48, 10.38]
median_bias = [7.79, 10.73]
max_bias = [21.09, 23.90]
x = np.arange(len(ds))
w = 0.26
ax.bar(x - w, mean_bias, w, label=T("平均高估", "mean"), color="#e74c3c")
ax.bar(x, median_bias, w, label=T("中位高估", "median"), color="#e67e22")
ax.bar(x + w, max_bias, w, label=T("最大高估", "max"), color="#f1c40f")
for xi, vals in zip(x, zip(mean_bias, median_bias, max_bias)):
    for off, v in zip([-w, 0, w], vals):
        ax.text(xi + off, v + 0.45, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(ds)
ax.set_ylabel(T("图折线相对精确测地的高估 %", "graph vs exact overestimate (%)"))
ax.set_title(T("现做法(图折线标签)对真实表面距离的系统性高估", "Current labels overestimate true surface distance"))
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
p2 = os.path.join(out_dir, "13_图折线标签的系统偏差.png")
plt.savefig(p2, dpi=160)
plt.close()
print("saved", p2)
