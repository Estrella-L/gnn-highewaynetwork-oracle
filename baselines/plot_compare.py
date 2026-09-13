# -*- coding: utf-8 -*-
"""把 EP_high 对比表画成图：主图（MRE/MAE 柱状）+ 收敛曲线图。

用法: python plot_compare.py --results results --ours_log <surface_train_res1.log> --out_dir results
"""
import argparse
import glob
import json
import os
import re

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

OURS = {'label': '我方方法（三段式+欧氏残差）', 'mre': 0.032199, 'mae': 199.40, 'rmse': 499.76}
NAME = {'gegnn': 'GeGnn', 'neurogf': 'NeuroGF', 'litege': 'LiteGE'}
MODE = {'native': '论文原样输出', 'euclidean_residual': '欧氏残差输出'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--ours_log', default='')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        with open(p, encoding='utf-8') as f:
            rows.append(json.load(f))
    if not rows:
        print('无结果'); return

    euc_mre = rows[0]['euclid_relative_error']
    euc_mae = rows[0].get('euclid_mae', float('nan'))

    items = [{'label': '纯 3D 欧氏直线（无学习）', 'mre': euc_mre, 'mae': euc_mae,
              'rmse': float('nan'), 'kind': 'ref'}]
    items.append({'label': OURS['label'], 'mre': OURS['mre'], 'mae': OURS['mae'],
                  'rmse': OURS['rmse'], 'kind': 'ours'})
    for r in rows:
        items.append({'label': '%s · %s' % (NAME.get(r['arch'], r['arch']),
                                            MODE.get(r['out_mode'], r['out_mode'])),
                      'mre': r['test_relative_error'], 'mae': r['test_mae'],
                      'rmse': r['test_rmse'],
                      'kind': 'base' if r['out_mode'] == 'euclidean_residual' else 'native'})
    items.sort(key=lambda d: d['mre'])

    colors = {'ref': '#999999', 'ours': '#C44E52', 'base': '#4C72B0', 'native': '#DD8452'}
    labels = [d['label'] for d in items]
    y = np.arange(len(items))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    ax.barh(y, [d['mre'] * 100 for d in items], color=[colors[d['kind']] for d in items])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('测试集平均相对误差 MRE (%)  —— 越小越好')
    ax.set_title('EP_high：三篇论文方法 vs 我方方法（精确表面测地真值）')
    for i, d in enumerate(items):
        ax.text(d['mre'] * 100 + 0.8, i, '%.2f%%' % (d['mre'] * 100), va='center', fontsize=8)
    ax.set_xlim(0, max(d['mre'] for d in items) * 100 * 1.18)
    ax.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    mae_items = [d for d in items if np.isfinite(d['mae'])]
    y2 = np.arange(len(mae_items))
    ax2.barh(y2, [d['mae'] for d in mae_items], color=[colors[d['kind']] for d in mae_items])
    ax2.set_yticks(y2); ax2.set_yticklabels([d['label'] for d in mae_items], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('测试集 MAE（越小越好）')
    ax2.set_title('同一测试集上的 MAE 对比')
    for i, d in enumerate(mae_items):
        ax2.text(d['mae'] + 80, i, '%.0f' % d['mae'], va='center', fontsize=8)
    ax2.set_xlim(0, max(d['mae'] for d in mae_items) * 1.18)
    ax2.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    p1 = os.path.join(args.out_dir, 'C1_对比柱状图.png')
    fig.savefig(p1, dpi=150)
    print('[plot]', p1)

    # 收敛曲线
    if args.ours_log and os.path.exists(args.ours_log):
        re_ours = re.compile(r'\[surf_res1_h64_k3_100ep\] epoch=(\d+).*?val_rel=([0-9.eE+-]+)')
        ours = []
        with open(args.ours_log, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re_ours.search(line)
                if m:
                    ours.append((int(m.group(1)), float(m.group(2))))
        fig2, ax = plt.subplots(figsize=(11, 6))
        ours.sort()
        ax.plot([e for e, _ in ours], [v for _, v in ours], color=colors['ours'], lw=2,
                label=OURS['label'])
        for r in rows:
            if r['out_mode'] != 'euclidean_residual':
                continue
            h = r['history']
            ax.plot([x['epoch'] for x in h], [x['val_rel'] for x in h],
                    label='%s · 欧氏残差输出' % NAME.get(r['arch'], r['arch']), lw=1.5)
        ax.axhline(euc_mre, color='#999999', ls='--', lw=1.2, label='纯 3D 欧氏直线')
        ax.set_xlabel('训练轮次 epoch'); ax.set_ylabel('验证集 MRE')
        ax.set_title('EP_high 收敛曲线（验证集相对误差）')
        ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig2.tight_layout()
        p2 = os.path.join(args.out_dir, 'C2_收敛曲线.png')
        fig2.savefig(p2, dpi=150)
        print('[plot]', p2)


if __name__ == '__main__':
    main()
