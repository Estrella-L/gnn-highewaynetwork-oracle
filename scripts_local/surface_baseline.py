# -*- coding: utf-8 -*-
"""用【表面测地真值】重算几何基线（3D/2D 直线），与旧（图折线）基线同口径可比。

同一 1000 对（seed 42，与旧基线一致），仅替换真值。
"""
import csv
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline import compute_metrics, euclidean_2d, euclidean_3d
from build_highway import load_off


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


_W = {}


def _init(off):
    import potpourri3d as pp3d
    V, F = load_off(off)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    _W['V'] = V
    _W['S'] = pp3d.EdgeFlipGeodesicSolver(V, F)


def _work(pairs):
    out = []
    for s, t in pairs:
        try:
            path = _W['S'].find_geodesic_path(int(s), int(t))
            d = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        except Exception:
            d = float('inf')
        out.append((int(s), int(t), d))
    return out


def main():
    off = sys.argv[1]
    out_prefix = sys.argv[2]
    n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 20

    vertices, _ = load_off(off)
    pairs = sample_pairs(len(vertices), n_pairs, seed)
    print(f'[surface-baseline] pairs={len(pairs)} workers={workers}', flush=True)

    import multiprocessing as mp
    ctx = mp.get_context('fork')
    tasks = [pairs[i:i + 50] for i in range(0, len(pairs), 50)]
    t0 = time.time()
    dist = {}
    with ctx.Pool(processes=workers, initializer=_init, initargs=(off,)) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            for s, t, d in res:
                dist[(s, t)] = d
    print(f'[surface-baseline] 表面测地计算完成 {time.time()-t0:.0f}s', flush=True)

    records = []
    for s, t in pairs:
        d = dist[(s, t)]
        if not math.isfinite(d):
            continue
        records.append({'s': s, 't': t, 'true_distance_surface': d,
                        'euclidean_2d': euclidean_2d(vertices, s, t),
                        'euclidean_3d': euclidean_3d(vertices, s, t)})
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
    for r in rows:
        print(f"[surface-baseline] {r['scope']:<11} {r['method']:<13} count={r['count']} "
              f"mae={r['mae']:.4f} rmse={r['rmse']:.4f} rel={r['relative_error']:.6f}", flush=True)


if __name__ == '__main__':
    main()
