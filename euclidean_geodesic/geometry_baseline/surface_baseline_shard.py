# -*- coding: utf-8 -*-
"""表面真值几何基线（fork 隔离版，避开 Pool 崩溃）：同一 1000 对（seed 42）。

分片计算:
  python surface_baseline_shard.py --off_file dataset/EP_high/EP_high.off --out_prefix outputs/.../surface_baseline \
      --pairs 1000 --seed 42 --shard 0 --shards 4
汇总:
  python surface_baseline_shard.py --out_prefix ... --merge
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from baseline import compute_metrics, euclidean_2d, euclidean_3d  # noqa: E402
from build_highway import load_off  # noqa: E402
from surface_labels_shard import build_solver, compute_isolated  # noqa: E402


def sample_pairs(num_vertices, num_pairs, seed):
    import random as _r
    rng = _r.Random(seed)
    pairs = set()
    while len(pairs) < num_pairs:
        s = rng.randrange(num_vertices)
        t = rng.randrange(num_vertices)
        if s == t:
            continue
        if s > t:
            s, t = t, s
        pairs.add((s, t))
    return sorted(pairs)


def merge(out_prefix):
    recs = {}
    for p in sorted(glob.glob(out_prefix + '_pairs.shard*.csv')):
        with open(p, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                try:
                    recs[(int(r['s']), int(r['t']))] = (float(r['true_distance_surface']),
                                                        float(r['euclidean_2d']), float(r['euclidean_3d']))
                except Exception:
                    continue
    records = [{'s': s, 't': t, 'true_distance_surface': v[0], 'euclidean_2d': v[1], 'euclidean_3d': v[2]}
               for (s, t), v in recs.items() if math.isfinite(v[0])]
    records.sort(key=lambda r: r['true_distance_surface'])
    n = len(records)
    groups = {'overall': records, 'short_dist': records[:n // 3],
              'mid_dist': records[n // 3:2 * n // 3], 'long_dist': records[2 * n // 3:]}
    rows = []
    for scope, grp in groups.items():
        y = [r['true_distance_surface'] for r in grp]
        for method in ('euclidean_2d', 'euclidean_3d'):
            m = compute_metrics(y, [r[method] for r in grp])
            rows.append({'scope': scope, 'method': method, 'count': len(grp),
                         'mae': m['mae'], 'rmse': m['rmse'], 'relative_error': m['relative_error']})
    with open(out_prefix + '_summary.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scope', 'method', 'count', 'mae', 'rmse', 'relative_error'])
        w.writeheader()
        w.writerows(rows)
    with open(out_prefix + '_pairs.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['s', 't', 'true_distance_surface', 'euclidean_2d', 'euclidean_3d'])
        w.writeheader()
        w.writerows(records)
    print('[baseline] 汇总 %d 对（表面真值）' % n, flush=True)
    for r in rows:
        print("[baseline] %-11s %-13s count=%d mae=%.4f rmse=%.4f rel=%.6f" %
              (r['scope'], r['method'], r['count'], r['mae'], r['rmse'], r['relative_error']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off_file')
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--pairs', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()

    if args.merge:
        merge(args.out_prefix)
        return

    vertices, _ = load_off(args.off_file)
    pairs = sample_pairs(len(vertices), args.pairs, args.seed)
    mine = [p for i, p in enumerate(pairs) if i % args.shards == args.shard]
    out_path = '%s_pairs.shard%d.csv' % (args.out_prefix, args.shard)
    hdr = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    print('[baseline%d] 本分片 %d 对' % (args.shard, len(mine)), flush=True)
    solver = build_solver(args.off_file)
    t0 = time.time()
    crashes = 0
    with open(out_path, 'a', encoding='utf-8-sig', newline='', buffering=1) as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(['s', 't', 'true_distance_surface', 'euclidean_2d', 'euclidean_3d'])
        for k, (s, t) in enumerate(mine):
            d, crashed = compute_isolated(solver, s, t)
            if crashed:
                crashes += 1
                d = float('inf')
            w.writerow([s, t, '%.6f' % d if math.isfinite(d) else 'inf',
                        '%.6f' % euclidean_2d(vertices, s, t), '%.6f' % euclidean_3d(vertices, s, t)])
            f.flush()
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print('[baseline%d] %d/%d 用时%.0fs 速率%.2f对/s 崩溃%d' %
                      (args.shard, k + 1, len(mine), el, (k + 1) / max(el, 1e-9), crashes), flush=True)
    print('[baseline%d] 完成 %d 对（崩溃 %d）' % (args.shard, len(mine), crashes), flush=True)


if __name__ == '__main__':
    main()
