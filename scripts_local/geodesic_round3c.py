# -*- coding: utf-8 -*-
"""第三轮 c：正确调用 fast marching + flip-out 精确路径，球面精度基准。"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local")
from scripts_local.geodesic_method_benchmark import icosphere, method_graph_dijkstra
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
    print(f"  {name:<34} 平均{100*errs.mean():+8.4f}%  中位{100*np.median(errs):+8.4f}%  "
          f"|P95| {100*np.percentile(np.abs(errs),95):7.4f}%  用时{dt:8.2f}s {extra}")


print(f"[球面基准] V={n} F={len(F)} 源={len(srcs)} 对={len(pairs_idx)}")

# fast marching（点源：curves=[[(v,[u,v])]], distances=[[0]]）
try:
    t0 = time.time()
    fm = pp3d.MeshFastMarchingDistanceSolver(V, F)
    D = np.zeros((len(srcs), n))
    for i, s in enumerate(srcs):
        D[i] = fm.compute_distance([[(s, [0.0, 0.0])]], [[0.0]], False).reshape(-1)
    dt = time.time() - t0
    report("f) fast marching(点源)", [(D[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dt)
except Exception as e:
    print("  f) fast marching 失败:", repr(e)[:200])

# flip-out 精确路径（逐对）
try:
    t0 = time.time()
    ef = pp3d.EdgeFlipGeodesicSolver(V, F)
    t_setup = time.time() - t0
    t0 = time.time()
    errs = []
    for r, t in pairs_idx:
        path = ef.find_geodesic_path(srcs[r], t)
        seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        errs.append((seg - truth(srcs[r], t)) / truth(srcs[r], t))
    dt = time.time() - t0
    report("h) flip-out 精确(逐对)", errs, dt, f"构造{t_setup:.2f}s, {dt/len(pairs_idx)*1000:.1f} ms/对")
except Exception as e:
    print("  h) flip-out 失败:", repr(e)[:200])

# 现做法
t0 = time.time()
Dg, dtg = method_graph_dijkstra(V, F, srcs)
report("a) 网格折线 Dijkstra(现做法)", [(Dg[r][t] - truth(srcs[r], t)) / truth(srcs[r], t) for r, t in pairs_idx], dtg)
