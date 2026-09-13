# -*- coding: utf-8 -*-
"""terrain 表面距离：各方案误差基准测试。

Part 1 解析验证（球面，真值 = 大圆距离）:
  a) 网格折线 Dijkstra（我们现在的做法）
  b) 细分网格 Dijkstra（1->4 细分 1/2 级）
  c) Heat Method (potpourri3d)
  d) 精确多面体测地 MMP (pygeodesic)
Part 2 真实地形互验（EP_low），以 d) 为参照。
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local"
sys.path.insert(0, ROOT)

RESULT = {}


# ---------------- 网格工具 ----------------
def icosphere(subdiv=4, radius=1.0):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    for _ in range(subdiv):
        cache = {}
        newF = []
        for a, b, c in F:
            def mid(i, j):
                key = (min(i, j), max(i, j))
                if key not in cache:
                    m = (V[i] + V[j]) / 2.0
                    V_list.append(m / np.linalg.norm(m) * radius)
                    cache[key] = len(V_list) - 1
                return cache[key]
            V_list = list(V)
            ab = mid(a, b); bc = mid(b, c); ca = mid(c, a)
            V = np.array(V_list, dtype=np.float64)
            newF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = np.array(newF, dtype=np.int64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    return V, F


def subdivide_midpoint(V, F, times=1):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    for _ in range(times):
        cache = {}
        vlist = [tuple(p) for p in V]
        newF = []
        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = (V[i] + V[j]) / 2.0
                vlist.append(tuple(m))
                cache[key] = len(vlist) - 1
            return cache[key]
        for a, b, c in F:
            ab = mid(a, b); bc = mid(b, c); ca = mid(c, a)
            newF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V = np.array(vlist, dtype=np.float64)
        F = np.array(newF, dtype=np.int64)
    return V, F


def edges_from_faces(V, F):
    E = set()
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            E.add((min(i, j), max(i, j)))
    E = np.array(sorted(E), dtype=np.int64)
    W = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return E, W


def dijkstra_multi(V, F, sources):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    E, W = edges_from_faces(V, F)
    n = len(V)
    g = csr_matrix((W, (E[:, 0], E[:, 1])), shape=(n, n))
    D = dijkstra(g, directed=False, indices=list(sources))
    return np.atleast_2d(D)


# ---------------- 方法实现 ----------------
def method_graph_dijkstra(V, F, sources):
    t0 = time.time()
    D = dijkstra_multi(V, F, sources)
    return D, time.time() - t0


def method_subdiv_dijkstra(V, F, sources, times=1):
    t0 = time.time()
    V2, F2 = subdivide_midpoint(V, F, times)
    D = dijkstra_multi(V2, F2, sources)
    return D, time.time() - t0, len(V2), len(F2)


def method_heat(V, F, sources):
    import potpourri3d as pp3d
    t0 = time.time()
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    D = np.zeros((len(sources), len(V)))
    for i, s in enumerate(sources):
        D[i] = solver.compute_distance(int(s))
    return D, time.time() - t0


def method_exact_mmp(V, F, sources):
    import pygeodesic.geodesic as pg
    t0 = time.time()
    geo = pg.PyGeodesicAlgorithmExact(V, F)
    D = np.zeros((len(sources), len(V)))
    for i, s in enumerate(sources):
        d, _ = geo.geodesicDistances([int(s)], None)
        D[i] = d
    return D, time.time() - t0


def stats(err):
    err = np.asarray(err, dtype=np.float64)
    return {
        "mean_rel_%": 100 * float(np.mean(err)),
        "median_rel_%": 100 * float(np.median(err)),
        "p95_rel_%": 100 * float(np.percentile(err, 95)),
        "max_rel_%": 100 * float(np.max(err)),
        "bias_%": 100 * float(np.mean(err)),  # 正=高估
    }


def rel_err(D, ref, pairs):
    out = []
    for srow, t in pairs:
        r = ref[srow][t]
        if r <= 0:
            continue
        out.append((D[srow][t] - r) / r)
    return np.array(out)


# ---------------- Part 1: 球面解析 ----------------
def run_sphere(subdiv=4, n_pairs=120, seed=0):
    print("=" * 70)
    print(f"[Part1] 球面解析验证 (icosphere subdiv={subdiv}, 真值=大圆距离)")
    V, F = icosphere(subdiv=subdiv)
    n = len(V)
    print(f"  顶点={n} 面={len(F)}")
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        s, t = rng.integers(0, n, 2)
        if s != t:
            pairs.append((int(s), int(t)))
    sources = sorted({p[0] for p in pairs})
    idx = {s: i for i, s in enumerate(sources)}
    pairs = [(idx[s], t) for s, t in pairs]

    def truth(k, t):
        v1, v2 = V[sources[k]], V[t]
        return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)))))

    ref = np.array([[truth(k, t) for t in range(n)] for k in range(len(sources))])

    results = {}
    # a) 图折线
    D, dt = method_graph_dijkstra(V, F, sources)
    results["a_图折线Dijkstra(现在做法)"] = (stats(rel_err(D, ref, pairs)), dt, {})
    # b) 细分
    for times in (1, 2):
        D2, dt2, n2, f2 = method_subdiv_dijkstra(V, F, sources, times=times)
        key = f"b_细分{times}级Dijkstra(顶点{n2})"
        results[key] = (stats(rel_err(D2, ref, pairs)), dt2, {"vertices": n2, "faces": f2})
    # c) heat
    try:
        D3, dt3 = method_heat(V, F, sources)
        results["c_HeatMethod(potpourri3d)"] = (stats(rel_err(D3, ref, pairs)), dt3, {})
    except Exception as e:
        results["c_HeatMethod(potpourri3d)"] = ({"error": repr(e)[:80]}, 0, {})
    # d) exact MMP
    try:
        D4, dt4 = method_exact_mmp(V, F, sources)
        results["d_精确MMP(pygeodesic)"] = (stats(rel_err(D4, ref, pairs)), dt4, {})
    except Exception as e:
        results["d_精确MMP(pygeodesic)"] = ({"error": repr(e)[:80]}, 0, {})

    for name, (st, dt, extra) in results.items():
        if "error" in st:
            print(f"  {name:<34} ERROR {st['error']}")
        else:
            print(f"  {name:<34} 平均{st['mean_rel_%']:+.3f}% 中位{st['median_rel_%']:+.3f}% "
                  f"P95 {st['p95_rel_%']:+.3f}% 最大{st['max_rel_%']:+.3f}% 用时{dt:.1f}s {extra}")
    return results


# ---------------- Part 2: 真实地形 ----------------
def run_terrain(off_path, n_sources=8, seed=0):
    print("=" * 70)
    print(f"[Part2] 真实地形互验: {os.path.basename(off_path)}")
    from build_highway import load_off
    vertices, faces = load_off(off_path)
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    print(f"  顶点={len(V)} 面={len(F)}")
    rng = np.random.default_rng(seed)
    sources = sorted(set(int(x) for x in rng.integers(0, len(V), n_sources)))
    targets = [int(x) for x in rng.integers(0, len(V), 40)]

    out = {}
    t0 = time.time()
    Dg, dtg = method_graph_dijkstra(V, F, sources)
    out["a_图折线Dijkstra"] = (Dg, dtg)
    print(f"  a) 图折线 Dijkstra: {dtg:.2f}s")

    try:
        Ds, dts, n2, f2 = method_subdiv_dijkstra(V, F, sources, times=1)
        out["b_细分1级"] = (Ds, dts)
        print(f"  b) 细分1级(顶点{n2}): {dts:.2f}s")
    except Exception as e:
        print("  b) 细分失败", repr(e)[:60])

    try:
        Dh, dth = method_heat(V, F, sources)
        out["c_HeatMethod"] = (Dh, dth)
        print(f"  c) Heat Method: {dth:.2f}s")
    except Exception as e:
        print("  c) heat 失败", repr(e)[:60])

    exact = None
    try:
        t0 = time.time()
        De, dte = method_exact_mmp(V, F, sources[:2])
        print(f"  d) 精确 MMP (前2个源): {dte:.2f}s -> 单源 {dte/2:.2f}s")
        exact = De
        out["d_精确MMP"] = (De, dte)
    except Exception as e:
        print("  d) MMP 失败", repr(e)[:80])

    if exact is not None:
        print("  ----- 与精确 MMP 的相对偏差 -----")
        for name, (D, dt) in out.items():
            if name.startswith("d_"):
                continue
            errs = []
            for k in range(exact.shape[0]):
                for t in targets:
                    r = exact[k][t]
                    if r > 0:
                        errs.append((D[k][t] - r) / r)
            errs = np.array(errs)
            print(f"    {name:<18} 平均{100*errs.mean():+.3f}% 中位{100*np.median(errs):+.3f}% "
                  f"P95 {100*np.percentile(errs,95):+.3f}% 最大{100*np.abs(errs).max():+.3f}%")
    return out


if __name__ == "__main__":
    r1 = run_sphere(subdiv=4, n_pairs=120)
    ep_low = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_low/EP_low.off"
    if os.path.exists(ep_low):
        r2 = run_terrain(ep_low, n_sources=8)
    print("=" * 70)
    print("DONE")
