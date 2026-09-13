# -*- coding: utf-8 -*-
"""对比【图折线标签】与【表面测地标签】两次 EP_high 实验，输出 markdown 报告 + 柱状图。

用法:
  python report_surface_vs_graph.py --graph_jsonl outputs/cloud_ep_high_results/ep_high_d3c32_grid.jsonl \
      --surface_jsonl outputs/cloud_ep_high_results/ep_high_surface_grid.jsonl \
      --out_dir outputs/cloud_ep_high_results
"""
import argparse
import json
import os
import re


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def pick(rows, name):
    for r in rows:
        if r.get('name') == name:
            return r
    return None


def fmt(v, nd=5):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_jsonl', default='outputs/cloud_ep_high_results/ep_high_d3c32_grid.jsonl')
    ap.add_argument('--surface_jsonl', default='outputs/cloud_ep_high_results/ep_high_surface_grid.jsonl')
    ap.add_argument('--out_dir', default='outputs/cloud_ep_high_results')
    args = ap.parse_args()

    graph = load_jsonl(args.graph_jsonl)
    surf = []
    for part in args.surface_jsonl.split(','):
        part = part.strip()
        if part:
            surf += load_jsonl(part)
    lines = ['# EP_high：图折线标签 vs 表面测地标签', '']
    if not surf:
        lines.append('（表面标签实验结果尚未生成）')
    lines += ['## 图折线标签（原口径，13 配置）', '',
              '| 配置 | test_rel | MAE | RMSE | best_epoch |', '| --- | --- | --- | --- | --- |']
    for r in sorted(graph, key=lambda x: x.get('test_relative_error', 9e9)):
        lines.append('| {} | {} | {} | {} | {} |'.format(
            r.get('name'), fmt(r.get('test_relative_error')), fmt(r.get('test_mae'), 2),
            fmt(r.get('test_rmse'), 2), r.get('best_epoch')))
    lines += ['', '## 表面测地标签（新口径）', '',
              '| 配置 | test_rel | MAE | RMSE | best_epoch |', '| --- | --- | --- | --- | --- |']
    for r in sorted(surf, key=lambda x: x.get('test_relative_error', 9e9)):
        lines.append('| {} | {} | {} | {} | {} |'.format(
            r.get('name'), fmt(r.get('test_relative_error')), fmt(r.get('test_mae'), 2),
            fmt(r.get('test_rmse'), 2), r.get('best_epoch')))

    gb = min(graph, key=lambda x: x.get('test_relative_error', 9e9)) if graph else None
    sb = min(surf, key=lambda x: x.get('test_relative_error', 9e9)) if surf else None
    lines += ['', '## 最优配置对比', '']
    if gb and sb:
        rg = gb['test_relative_error']; rs = sb['test_relative_error']
        lines += ['| 口径 | 最优配置 | test_rel | MAE | RMSE |', '| --- | --- | --- | --- | --- |',
                  f"| 图折线 | {gb['name']} | {rg:.5f} | {gb.get('test_mae', 0):.2f} | {gb.get('test_rmse', 0):.2f} |",
                  f"| 表面测地 | {sb['name']} | {rs:.5f} | {sb.get('test_mae', 0):.2f} | {sb.get('test_rmse', 0):.2f} |",
                  '', f'相对变化：{(rs - rg) / rg * 100:+.2f}%']

    out_md = os.path.join(args.out_dir, 'EP_high_表面vs图标签_对比.md')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for fam in ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']:
            try:
                from matplotlib import font_manager
                if any(fam.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    break
            except Exception:
                pass
        names, vals, colors = [], [], []
        for r in sorted(graph, key=lambda x: -x.get('test_relative_error', 0))[:13]:
            names.append('图/' + r['name']); vals.append(r['test_relative_error']); colors.append('#4C72B0')
        for r in sorted(surf, key=lambda x: -x.get('test_relative_error', 0)):
            names.append('面/' + r['name']); vals.append(r['test_relative_error']); colors.append('#DD8452')
        if names:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.barh(range(len(names)), vals, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('test_relative_error（越小越好）')
            ax.set_title('EP_high：图折线标签 vs 表面测地标签')
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, 'EP_high_表面vs图标签_对比.png'), dpi=150)
    except Exception as e:
        print('[warn] 画图跳过:', repr(e))
    print('[report] ->', out_md)


if __name__ == '__main__':
    main()
