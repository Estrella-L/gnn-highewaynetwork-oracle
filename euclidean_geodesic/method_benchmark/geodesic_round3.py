# -*- coding: utf-8 -*-
"""第三轮：flip-out 精确测地 + heat 边际成本 + 与 MMP 精度对比。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local")
from build_highway import load_off
from scripts_local.geodesic_method_benchmark import icosphere, method_exact_mmp, method_heat

import potpourri3d as pp3d

print("potpourri3d 可用 API:", [a for a in dir(pp3d) if not a.startswith("_")])
print()


def sphere_truth(V, s, t):
    import math
    return math.acos(max(-1.0, min(1.0, float(np.dot(V[s], V[t])))))


def test_flip_sphere(subdiv=4, n_pairs=60, seed=0):
    V, F = icosphere(subdiv=subdiv)
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        s, t = rng.integers(0, len(V), 2)
        if s != t:
            pairs.append((int(s), int(t)))
    srcs = sorted({p[0] for p in pairs})
    idx = {s: i for i, s in enumerate(srcs)}
    pairs = [(idx[s], t) for s, t in pairs]
    print(f"[flip/sphere] V={len(V)} F={len(F)} sources={len(srcs)}")
    # flip-out geodesics
    t0 = time.time()
    try:
        flip = pp3d.EdgeFlipGeodesicsManager(V, F)
        D = np.zeros((len(srcs), len(V)))
        for i, s in enumerate(srcs):
            D[i] = flip.compute_geodesic_distance(s)
        dt = time.time() - t0
        errs = np.array([(D[r][t] - sphere_truth(V, srcs[r], t)) / sphere_truth(V, srcs[r], t) for r, t in pairs])
        print(f"  flip-out 精确: 平均{100*errs.mean():+.4f}% 中位{100*np.median(errs):+.4f}% "
              f"P95 {100*np.percentile(errs,95):+.4f}% 用时{dt:.2f}s ({dt/len(srcs):.3f}s/源)")
    except Exception as e:
        print("  flip-out 失败:", repr(e)[:120])


def marginal_cost(off_path, label, n_sources=3):
    V, F = load_off(off_path)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    rng = np.random.default_rng(1)
    srcs = [int(x) for x in rng.integers(0, n, n_sources)]
    print(f"[边际成本] {label}: V={n}")
    # heat
    try:
        t0 = time.time()
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        t_setup = time.time() - t0
        ts = []
        for s in srcs:
            t0 = time.time()
            solver.compute_distance(s)
            ts.append(time.time() - t0)
        print(f"  heat: 分解{t_setup:.1f}s + 每源 平均{np.mean(ts):.2f}s (明细 {[round(x,2) for x in ts]})")
    except Exception as e:
        print("  heat 失败:", repr(e)[:80])
    # flip-out
    try:
        t0 = time.time()
        flip = pp3d.EdgeFlipGeodesicsManager(V, F)
        t_setup = time.time() - t0
        ts = []
        for s in srcs:
            t0 = time.time()
            flip.compute_geodesic_distance(s)
            ts.append(time.time() - t0)
        print(f"  flip-out 精确: 构造{t_setup:.1f}s + 每源 平均{np.mean(ts):.2f}s (明细 {[round(x,2) for x in ts]})")
    except Exception as e:
        print("  flip-out 失败:", repr(e)[:100])
    # MMP
    try:
        from pygeodesic import geodesic as pg
        geo = pg.PyGeodesicAlgorithmExact(V, F)
        t0 = time.time()
        geo.geodesicDistances([srcs[0]], None)
        print(f"  MMP 精确: 每源 ~{time.time()-t0:.2f}s")
    except Exception as e:
        print("  MMP 失败:", repr(e)[:80])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sphere"
    if mode == "sphere":
        test_flip_sphere()
    elif mode == "ep_low":
        marginal_cost("/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_low/EP_low.off", "EP_low(16.4万)")
    elif mode == "ep_high":
        marginal_cost("/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_high/EP_high.off", "EP_high(139万)")
