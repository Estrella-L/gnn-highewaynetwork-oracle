# -*- coding: utf-8 -*-
"""第三轮 b：把 potpourri3d 所有测地算法在球面上做精度/速度基准。"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local")
from scripts_local.geodesic_method_benchmark import icosphere, method_exact_mmp, method_heat
import potpourri3d as pp3d

V, F = icosphere(subdiv=4)
n = len(V)
rng = np.random.default_rng(0)
pairs = []
while len(pairs) < 60:
    s, t = rng.integers(0, n, 2)
    if s != t:
        pairs.append((int(s), int(t)))
srcs = sorted({p[0] for p in pairs})
idx = {s: i for i, s in enumerate(srcs)}
pairs_idx = [(idx[s], t) for s, t in pairs]


def truth(s, t):
    return math.acos(max(-1.0, min(1.0, float(np.dot(V[s], V[t])))))


def report(name, errs, dt, extra=""):
    errs = np.asarray(errs)
    print(f"  {name:<32} 平均{100*errs.mean():+8.4f}%  中位{100*np.median(errs):+8.4f}%  "
          f"P95 {100*np.percentile(np.abs(errs),95):7.4f}%  用时{dt:7.2f}s {extra}")


print(f"[球面基准 subdiv=4] V={n} F={len(F)} 源={len(srcs)} 对={len(pairs)}")

# 1) heat
t0 = time.time()
solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
D = np.array([solver.compute_distance(s) for s in srcs])
dt = time.time() - t0
report("c) heat method", [(D[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dt)

# 2) fast marching
t0 = time.time()
fm = pp3d.MeshFastMarchingDistanceSolver(V, F)
D = np.array([fm.compute_distance(s) for s in srcs])
dt = time.time() - t0
report("f) fast marching", [(D[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dt)

# 3) marching triangles
try:
    t0 = time.time()
    mt = pp3d.MeshMarchingTrianglesSolver(V, F)
    D = np.array([mt.marching_triangles(s, np.inf) for s in srcs])
    dt = time.time() - t0
    if D.ndim == 1:
        D = D.reshape(1, -1)
    report("g) marching triangles", [(D[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dt)
except Exception as e:
    print("  g) marching triangles 失败:", repr(e)[:150])

# 4) flip-out exact paths (per pair)
try:
    t0 = time.time()
    ef = pp3d.EdgeFlipGeodesicSolver(V, F)
    d = []
    for r, t in pairs_idx:
        path = ef.find_geodesic_path(srcs[r], t)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        d.append(seg)
    dt = time.time() - t0
    errs = [(seg - truth(srcs[r], t)) / truth(srcs[r], t) for seg, (r, t) in zip(d, pairs_idx)]
    report("h) flip-out (exact, per-pair)", errs, dt, f"({dt/len(pairs_idx)*1000:.1f} ms/对)")

except Exception as e:
    print("  h) flip-out 失败:", repr(e)[:150])

# 5) MMP (exact per-source)
try:
    t0 = time.time()
    _, dt = method_exact_mmp(V, F, srcs)
    Dm, _ = method_exact_mmp(V, F, srcs)
    report("d) MMP (exact, per-source)", [(Dm[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dt)
except Exception as e:
    print("  d) MMP 失败:", repr(e)[:100])

# 6) 我们现在的做法：网格折线
from scripts_local.geodesic_method_benchmark import method_graph_dijkstra
t0 = time.time()
Dg, dtg = method_graph_dijkstra(V, F, srcs)
report("a) 网格折线 Dijkstra(现做法)", [(Dg[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dtg)
