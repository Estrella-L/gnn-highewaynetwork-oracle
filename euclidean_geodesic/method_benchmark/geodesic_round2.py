# -*- coding: utf-8 -*-
"""第二轮：诊断细分为何无改善 + EP_high 上的耗时/内存实测。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local")
from build_highway import load_off
from scripts_local.geodesic_method_benchmark import (
    edges_from_faces, method_graph_dijkstra, method_subdiv_dijkstra,
    method_heat, method_exact_mmp, dijkstra_multi,
)

EP_LOW = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_low/EP_low.off"
EP_HIGH = "/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/EP_high/EP_high.off"


def diag_subdiv(off_path, src=0, n_check=6):
    V, F = load_off(off_path)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    print(f"[诊断] {os.path.basename(off_path)}: V={len(V)} F={len(F)}")
    E0, W0 = edges_from_faces(V, F)
    print(f"  原始图: 边={len(E0)} 平均边长={W0.mean():.3f} 最大={W0.max():.3f}")
    D0 = dijkstra_multi(V, F, [src])[0]
    for times in (1, 2):
        V2, F2 = None, None
        import scripts_local.geodesic_method_benchmark as B
        V2, F2 = B.subdivide_midpoint(V, F, times)
        E1, W1 = edges_from_faces(V2, F2)
        print(f"  细分{times}级: 顶点={len(V2)} 边={len(E1)} 平均边长={W1.mean():.4f}")
        D1 = dijkstra_multi(V2, F2, [src])[0]
        diffs = [(D1[t] - D0[t]) / D0[t] * 100 for t in range(0, len(V), max(1, len(V)//n_check)) if D0[t] > 0]
        print(f"    -> 与原始图距离的相对变化: 最大 {max(diffs):+.4f}% 最小 {min(diffs):+.4f}% (采样{len(diffs)}个点)")


def time_methods(off_path, label, n_sources=1, do_mmp=True):
    V, F = load_off(off_path)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    print(f"[耗时] {label}: V={n} F={len(F)}")
    rng = np.random.default_rng(0)
    srcs = sorted(set(int(x) for x in rng.integers(0, n, n_sources)))
    t0 = time.time()
    method_graph_dijkstra(V, F, srcs[:1])
    t_a = time.time() - t0
    print(f"  a) 图折线 Dijkstra      : {t_a:8.2f}s / 源")
    t0 = time.time()
    try:
        method_heat(V, F, srcs[:1])
        t_c = time.time() - t0
        print(f"  c) Heat Method          : {t_c:8.2f}s / 源（含分解）")
    except Exception as e:
        print("  c) heat 失败:", repr(e)[:80])
    if do_mmp:
        t0 = time.time()
        try:
            method_exact_mmp(V, F, srcs[:1])
            t_d = time.time() - t0
            print(f"  d) 精确 MMP             : {t_d:8.2f}s / 源")
        except Exception as e:
            print("  d) MMP 失败:", repr(e)[:100])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if which == "diag":
        diag_subdiv(EP_LOW)
    elif which == "ep_low":
        time_methods(EP_LOW, "EP_low(16.4万点)", n_sources=1)
    elif which == "ep_high":
        time_methods(EP_HIGH, "EP_high(139万点)", n_sources=1)
