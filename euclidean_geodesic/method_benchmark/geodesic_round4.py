# -*- coding: utf-8 -*-
"""第四轮：flip-out 精确测地在真实地形上的速度与一致性（vs 图折线、vs MMP）。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local")
from build_highway import load_off
from scripts_local.geodesic_method_benchmark import edges_from_faces
import potpourri3d as pp3d

EP_LOW = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_low/EP_low.off"
EP_HIGH = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_high/EP_high.off"


def bench(off_path, label, n_pairs=20, do_mmp=False, do_graph=True):
    print("=" * 72)
    V, F = load_off(off_path)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    print(f"[{label}] V={n} F={len(F)}")
    rng = np.random.default_rng(7)
    pairs = []
    seen = set()
    while len(pairs) < n_pairs:
        s, t = (int(x) for x in rng.integers(0, n, 2))
        if s != t and (s, t) not in seen:
            seen.add((s, t))
            pairs.append((s, t))

    # flip-out 精确
    t0 = time.time()
    ef = pp3d.EdgeFlipGeodesicSolver(V, F)
    t_setup = time.time() - t0
    ts, dists = [], []
    for s, t in pairs:
        t1 = time.time()
        path = ef.find_geodesic_path(s, t)
        seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        ts.append(time.time() - t1)
        dists.append(seg)
    ts = np.array(ts)
    print(f"  h) flip-out 精确: 构造{t_setup:.2f}s, 每对 平均{1000*ts.mean():.1f}ms "
          f"中位{1000*np.median(ts):.1f}ms 最大{1000*ts.max():.1f}ms")
    print(f"     距离统计: 平均{dists and np.mean(dists):.1f} 最小{min(dists):.1f} 最大{max(dists):.1f}")

    if do_graph:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        E, W = edges_from_faces(V, F)
        g = csr_matrix((W, (E[:, 0], E[:, 1])), shape=(n, n))
        srcs = sorted({s for s, _ in pairs})
        t0 = time.time()
        D = dijkstra(g, directed=False, indices=srcs)
        dt = time.time() - t0
        row = {s: i for i, s in enumerate(srcs)}
        errs = [(D[row[s]][t] - d) / d * 100 for (s, t), d in zip(pairs, dists)]
        print(f"  a) 图折线 Dijkstra: 每源 {dt/len(srcs):.2f}s -> 相对精确测地 平均{np.mean(errs):+.2f}% "
              f"中位{np.median(errs):+.2f}% 最大{max(errs):+.2f}%")

    if do_mmp:
        from pygeodesic import geodesic as pg
        geo = pg.PyGeodesicAlgorithmExact(V, F)
        n_mmp = min(3, len(pairs))
        t0 = time.time()
        for s, t in pairs[:n_mmp]:
            d, _ = geo.geodesicDistances([s], None)
            print(f"     MMP 校验: pair -> {d[t]:.3f} vs flip-out {dists[pairs[:n_mmp].index((s,t))]:.3f}")
        print(f"     MMP 用时 {(time.time()-t0)/n_mmp:.1f}s/源")
    return ts


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "low"
    if what in ("low", "both"):
        bench(EP_LOW, "EP_low(16.4万点)", n_pairs=20, do_mmp=True)
    if what in ("high", "both"):
        bench(EP_HIGH, "EP_high(139万点)", n_pairs=12, do_mmp=False)
