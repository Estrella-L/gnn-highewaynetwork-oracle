# -*- coding: utf-8 -*-
"""分析 EP_high 网格结果：读取 grid.jsonl + 几何基线 CSV，输出排序与对比总结。"""
import csv
import json
import sys

results_path = sys.argv[1] if len(sys.argv) > 1 else None
base_path = sys.argv[2] if len(sys.argv) > 2 else None

if results_path is None:
    print('usage: analyze_ep_high_results.py <grid.jsonl> [baseline_summary.csv]')
    sys.exit(1)

rows = []
with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

valid = [r for r in rows if r.get('test_relative_error') is not None]
valid.sort(key=lambda r: r['test_relative_error'])

print('== EP_high grid ranking (by test_relative_error) ==')
print(f'{"test_rel":>10} {"mae":>10} {"rmse":>10} {"best_ep":>7} {"run":>6}  config')
for r in valid:
    print(f"{r['test_relative_error']:>10.5f} {r['test_mae']:>10.2f} {r['test_rmse']:>10.2f} "
          f"{int(r.get('best_epoch', -1)):>7} {int(r.get('epochs_run', 0)):>6}  "
          f"{r['name']} (strategy={r.get('sample_strategy')} loss={r.get('loss_type')} k={r.get('highway_k')} "
          f"dim={r.get('hidden_dim')}/{r.get('out_dim')} tk={r.get('transit_k')})")

print()
print('== group metrics of top run ==')
if valid:
    top = valid[0]
    for key in sorted(top):
        if key.startswith('group_') and 'relative_error' in key:
            print(f'  {key}: {top[key]:.6f}')

if base_path:
    print()
    print('== Geometry baseline (exact Dijkstra euclidean_2d/3d) ==')
    with open(base_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"  {row['scope']:<12} {row['method']:<13} count={row['count']} "
                  f"mae={float(row['mae']):.6f} rmse={float(row['rmse']):.6f} rel={float(row['relative_error']):.6f}")
