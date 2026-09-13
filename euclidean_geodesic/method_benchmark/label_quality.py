# -*- coding: utf-8 -*-
"""标签质量检查：我们对（网格图 Dijkstra）离真实曲面测地（精确 MMP）有多远？

对若干随机源点做一次 SSSD（精确 MMP），再对同样的源点做网格图 Dijkstra，
在同一批目标点上对比三种"距离"：

  A. 网格图最短路（我们现在的训练标签）
  B. 3D 欧氏直线距离（无学习基线）
  C. 三角曲面精确测地（pygeodesic exact MMP，GeGnn 的口径）<- reference

用法：
  PYTHONPATH=<workspace>/.pylibs python3 scripts_local/label_quality.py \
      --off_file /path/to/EP_low.off --num_sources 8 --num_dest 300
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from exact_geodesic import load_off, ExactGeodesic  # noqa: E402


def build_csr(V, F):
    n = len(V)
    rows, cols, vals = [], [], []
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            w = float(np.linalg.norm(V[u] - V[v]))
            rows.append(u); cols.append(v); vals.append(max(1e-9, w))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return m


def summarize(name, pred, ref):
    m = np.isfinite(pred) & np.isfinite(ref) & (ref > 1e-9)
    if m.sum() == 0:
        print("  %-24s (no valid pairs)" % name)
        return
    rel = np.abs(pred[m] - ref[m]) / ref[m]
    ratio = pred[m] / ref[m]
    print("  %-24s rel_err mean=%.6f median=%.6f | MAE=%.2f | ratio median=%.4f P05=%.4f P95=%.4f | >ref %.1f%%"
          % (name, rel.mean(), np.median(rel), np.abs(pred[m] - ref[m]).mean(),
             np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95),
             100.0 * (ratio > 1.0 + 1e-9).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_file", required=True)
    ap.add_argument("--num_sources", type=int, default=8)
    ap.add_argument("--num_dest", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "geo_cache"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eg = ExactGeodesic(args.off_file, args.cache_dir)
    V, F = eg.V, eg.F
    print("[label] mesh |V|=%d |F|=%d  build=%.2fs" % (len(V), len(F), eg.build_seconds))

    t0 = time.time()
    csr = build_csr(V, F)
    print("[label] graph csr nnz=%d (%.1fs)" % (csr.nnz, time.time() - t0))

    sources = rng.choice(len(V), size=min(args.num_sources, len(V)), replace=False)
    print("[label] %d sources x %d dests" % (len(sources), args.num_dest))

    t0 = time.time()
    graph_rows = sp_dijkstra(csr, directed=False, indices=sources)
    print("[label] graph Dijkstra done in %.1fs" % (time.time() - t0))

    g_all, e_all, u_all = [], [], []
    for k, s in enumerate(sources):
        t0 = time.time()
        d_exact = eg.row(int(s))
        print("    src=%d exact SSSD %.1fs" % (s, time.time() - t0), flush=True)
        dests = rng.choice(len(V), size=args.num_dest, replace=False)
        g_all.append(graph_rows[k][dests])
        e_all.append(d_exact[dests].astype(np.float64))
        u_all.append(np.linalg.norm(V[dests] - V[s], axis=1))
    g = np.concatenate(g_all); e = np.concatenate(e_all); u = np.concatenate(u_all)

    print("[label] === reference = 精确 MMP 曲面测地 ===")
    summarize("graph-Dijkstra (our label)", g, e)
    summarize("euclidean_3d", u, e)
    print("[label] 绝对量级：exact mean=%.2f  graph mean=%.2f  euclid mean=%.2f" % (e.mean(), g.mean(), u.mean()))
    print("[label] graph 比 exact 长的比例 = %.2f%%" % (100.0 * (g > e + 1e-6).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
