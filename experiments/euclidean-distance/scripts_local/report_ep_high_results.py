# -*- coding: utf-8 -*-
"""生成 EP_high 实验结果报告（配合 grid.jsonl + 基线 CSV）。
用法: python report_ep_high_results.py <grid.jsonl> [baseline_summary.csv] [out_md]
"""
import csv
import json
import sys

results_path = sys.argv[1]
base_path = sys.argv[2] if len(sys.argv) > 2 else None
out_md = sys.argv[3] if len(sys.argv) > 3 else "ep_high_results_report.md"

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

valid = [r for r in rows if r.get("test_relative_error") is not None]
valid.sort(key=lambda r: r["test_relative_error"])

lines = []
lines.append("# EP_high 参数寻优实验结果")
lines.append("")
lines.append("## 1. 配置排名 (按 test_relative_error 升序)")
lines.append("")
lines.append("| # | config | strategy | loss | k | dim(h/o) | transit_k | best_ep | epochs | test_mae | test_rmse | test_rel |")
lines.append("|---|--------|----------|------|---|----------|-----------|---------|---------|----------|-----------|----------|")
for i, r in enumerate(valid):
    lines.append(f"| {i+1} | {r['name']} | {r.get('sample_strategy')} | {r.get('loss_type')} | {r.get('highway_k')} | "
                f"{r.get('hidden_dim')}/{r.get('out_dim')} | {r.get('transit_k')} | {int(r.get('best_epoch',-1))} | "
                f"{int(r.get('epochs_run',0))} | {r['test_mae']:.2f} | {r['test_rmse']:.2f} | {r['test_relative_error']:.5f} |")

if valid:
    top = valid[0]
    lines.append("")
    lines.append("## 2. 最优配置")
    lines.append(f"- **config**: `{top['name']}`")
    lines.append(f"- strategy={top.get('sample_strategy')}, loss={top.get('loss_type')}, highway_k={top.get('highway_k')}, "
                f"hidden={top.get('hidden_dim')}, out={top.get('out_dim')}, transit_k={top.get('transit_k')}")
    lines.append(f"- test_mae={top['test_mae']:.4f}, test_rmse={top['test_rmse']:.4f}, test_relative_error={top['test_relative_error']:.6f}")
    gkeys = sorted([k for k in top if k.startswith('group_') and 'relative_error' in k])
    if gkeys:
        lines.append("")
        lines.append("### 分组误差 (最优配置)")
        lines.append("")
        lines.append("| group | count | relative_error |")
        lines.append("|-------|-------|----------------|")
        for gk in gkeys:
            cnt_key = gk.replace('_relative_error', '_count')
            lines.append(f"| {gk} | {top.get(cnt_key, '')} | {top[gk]:.6f} |")

if base_path:
    lines.append("")
    lines.append("## 3. 几何基线 (精确 Dijkstra 对照)")
    lines.append("")
    lines.append("| scope | method | count | mae | rmse | relative_error |")
    lines.append("|-------|--------|-------|-----|------|----------------|")
    with open(base_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lines.append(f"| {row['scope']} | {row['method']} | {row['count']} | "
                        f"{float(row['mae']):.4f} | {float(row['rmse']):.4f} | {float(row['relative_error']):.6f} |")

lines.append("")
lines.append("## 4. 说明")
lines.append("- 数据集: EP_high (|V|=1,392,236, |F|=2,776,348, 四叉树 d3/c32 → 64 叶, K=33,255 高速节点)")
lines.append("- 预测模式: euclidean_residual (3D 直线锚 + 0.5*tanh 有界修正, 关闭高速距离特征)")
lines.append("- 采样: 50,000 点对, 8:1:1 划分, seed 42; 训练 12 epoch 屏幕筛选, selection_metric=relative_error")
lines.append("")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines[:40]))
print("\n[REPORT WRITTEN]", out_md)
