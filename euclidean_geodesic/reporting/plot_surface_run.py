# -*- coding: utf-8 -*-
"""画表面测地标签实验的训练曲线（3+1 配置）。

用法: python plot_surface_run.py --log outputs/.../surface/surface_train.log --out_dir outputs/.../surface
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for fam in ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti']:
    try:
        from matplotlib import font_manager
        if any(fam.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            break
    except Exception:
        pass

RE = re.compile(r"\[(?P<cfg>[^\]]+)\] epoch=(?P<ep>\d+).*?val_mae=(?P<vmae>[0-9.eE+-]+)"
                r".*?val_rel=(?P<vrel>[0-9.eE+-]+).*?lr=(?P<lr>[0-9.eE+-]+)")


def parse(path):
    data = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = RE.search(line)
            if not m:
                continue
            cfg = m.group('cfg')
            data.setdefault(cfg, []).append((int(m.group('ep')), float(m.group('vmae')),
                                             float(m.group('vrel')), float(m.group('lr'))))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = parse(args.log)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for cfg, rows in data.items():
        rows.sort()
        ep = [r[0] for r in rows]
        axes[0].plot(ep, [r[2] for r in rows], marker='.', ms=3, label=cfg)
        axes[1].plot(ep, [r[1] for r in rows], marker='.', ms=3, label=cfg)
    axes[0].set_xlabel('训练轮次 epoch'); axes[0].set_ylabel('验证相对误差 val_rel')
    axes[0].set_title('表面测地标签：验证相对误差曲线'); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel('训练轮次 epoch'); axes[1].set_ylabel('验证 MAE')
    axes[1].set_title('表面测地标签：验证 MAE 曲线'); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, 'S1_表面标签_训练曲线.png')
    fig.savefig(p1, dpi=150)
    print('[plot]', p1)

    best = max(data.items(), key=lambda kv: len(kv[1])) if data else None
    if best:
        cfg, rows = best
        rows.sort()
        fig2, ax2 = plt.subplots(figsize=(11, 5.5))
        ax2.plot([r[0] for r in rows], [r[2] for r in rows], label='val_rel', color='#C44E52')
        ax3 = ax2.twinx()
        ax3.plot([r[0] for r in rows], [r[3] for r in rows], label='lr', color='#4C72B0', ls='--')
        ax2.set_xlabel('epoch'); ax2.set_ylabel('val_rel', color='#C44E52')
        ax3.set_ylabel('learning rate', color='#4C72B0')
        ax2.set_title('表面测地标签：%s 详细曲线（val_rel 与学习率）' % cfg)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        p2 = os.path.join(args.out_dir, 'S2_表面标签_最优配置详细曲线.png')
        fig2.savefig(p2, dpi=150)
        print('[plot]', p2)


if __name__ == '__main__':
    main()
