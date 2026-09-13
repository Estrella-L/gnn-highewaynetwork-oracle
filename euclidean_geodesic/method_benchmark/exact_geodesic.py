# -*- coding: utf-8 -*-
"""用精确测地算法（MMP, pygeodesic）在我们自己的地形网格上算真值距离。

背景
----
三篇 baseline（GeGnn / NeuroGF / LiteGE）的真值都是**三角曲面上的精确测地距离**：
  - GeGnn  : pygeodesic 的 PyGeodesicAlgorithmExact（精确 MMP）
  - NeuroGF: DGG-VTP / fDGG (eps=1e-7)
  - LiteGE : Fast-DGG (accuracy control 0.1-0.3%)
而我们现在的标签是「网格图 Dijkstra 最短路」（边权 = 3D 边长），只是折线近似。
学长说的"有计算方式就行"，就是把这套**距离计算方式**直接搬到我们的图上。

做法
----
一次 SSSD（单源 -> 全图）就能得到该源点到**所有顶点**的精确测地距离，
所以：

  * 成本 = 不同源点数 x 单次 SSSD 时间（不是点对数！）
  * 每个源点的距离行按 .npy 缓存，之后任意 (s,t) 只要源点算过就零成本查表

实测速度（本机 M 系列 CPU）：
  small_terrain (1221 顶点)   : 0.02 s / 源
  EP_low        (164238 顶点) : 14.0 s / 源   -> 1000 源约 4 小时，可后台跑

用法
----
  # (A) 给已有的一批点对（例如某个 run 的 test_pairs.csv）打精确测地标签
  PYTHONPATH=<workspace>/.pylibs python3 scripts_local/exact_geodesic.py \
      --off_file /path/to/small_terrain.off \
      --pairs_file outputs/results/<run>_test_pairs.csv \
      --out_file outputs/geo_results/<tag>_exact_geo.csv

  # (B) GeGnn 风格重新采样点对（num_sources x num_each_dest），适合大图上摊薄成本
  PYTHONPATH=<workspace>/.pylibs python3 scripts_local/exact_geodesic.py \
      --off_file /path/to/EP_low.off --gen --num_sources 600 --num_each_dest 100 \
      --out_file outputs/geo_results/EP_low_exact60k.csv

输出 CSV 列：s,t,graph_dist,exact_geo,euclid_3d
"""
import argparse
import hashlib
import heapq
import math
import os
import random
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

try:
    import pygeodesic.geodesic as pg
except ImportError:
    sys.stderr.write(
        "缺少 pygeodesic。安装方式（装到工作区内，避免系统目录权限问题）：\n"
        "  PIP_CACHE_DIR=<workspace>/.pipcache python3 -m pip install --no-cache-dir "
        "--target <workspace>/.pylibs pygeodesic\n"
        "然后设 PYTHONPATH=<workspace>/.pylibs 再运行本脚本。\n")
    raise


# ---------------------------------------------------------------------------
# .off 读取 + 网格图（与 build_highway.load_off / build_mesh_graph 同口径）
# ---------------------------------------------------------------------------
def load_off(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tok = f.read().split()
    i = 0
    if tok[0].upper().startswith("OFF"):
        i = 1
    nv = int(tok[i]); nf = int(tok[i + 1]); i += 3
    V = np.asarray(tok[i:i + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i += 3 * nv
    F = np.empty((nf, 3), dtype=np.int32)
    for k in range(nf):
        m = int(tok[i]); i += 1
        F[k, 0] = int(tok[i]); F[k, 1] = int(tok[i + 1]); F[k, 2] = int(tok[i + 2])
        i += m
    return np.ascontiguousarray(V), np.ascontiguousarray(F)


def mesh_fingerprint(path):
    h = hashlib.md5()
    st = os.stat(path)
    h.update(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:12]


def build_graph_adj(V, F):
    """网格图邻接（无向、边权 = 3D 欧氏边长），与主流程 build_mesh_graph 一致。"""
    n = len(V)
    nbr = [set() for _ in range(n)]
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr[u].add(v); nbr[v].add(u)
    adj = [[] for _ in range(n)]
    for u in range(n):
        for v in nbr[u]:
            w = float(np.linalg.norm(V[u] - V[v]))
            adj[u].append((v, max(1e-9, w)))
    return adj


def dijkstra_row(adj, src):
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ---------------------------------------------------------------------------
# 精确测地：按源点缓存
# ---------------------------------------------------------------------------
class ExactGeodesic:
    def __init__(self, off_path, cache_dir):
        self.off_path = off_path
        self.V, self.F = load_off(off_path)
        self.n = len(self.V)
        self.fp = mesh_fingerprint(off_path)
        self.cache_dir = os.path.join(cache_dir, "geo_" + self.fp)
        os.makedirs(self.cache_dir, exist_ok=True)
        t0 = time.time()
        self.geo = pg.PyGeodesicAlgorithmExact(self.V, self.F)
        self.build_seconds = time.time() - t0
        self._row_cache = {}
        self.n_computed = 0
        self.total_seconds = 0.0

    def _cache_path(self, src):
        return os.path.join(self.cache_dir, "src_%d.npy" % src)

    def row(self, src):
        """返回源点到全图所有顶点的精确测地距离 [N]（numpy float32），带磁盘/内存缓存。"""
        if src in self._row_cache:
            return self._row_cache[src]
        p = self._cache_path(src)
        if os.path.exists(p):
            d = np.load(p)
        else:
            t0 = time.time()
            d, _best = self.geo.geodesicDistances(np.array([src], dtype=np.int32), None)
            dt = time.time() - t0
            self.n_computed += 1
            self.total_seconds += dt
            d = np.asarray(d, dtype=np.float32)
            np.save(p, d)
        self._row_cache[src] = d
        return d

    def pair_dist(self, s, t):
        d = self.row(s)
        v = float(d[t])
        return v if np.isfinite(v) else float("inf")


def euclid3d(V, s, t):
    return float(np.linalg.norm(V[s] - V[t]))


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="用精确 MMP 测地在我们地形上算真值")
    p.add_argument("--off_file", required=True)
    p.add_argument("--pairs_file", default="", help="已有 (s,t,true_distance) CSV；与 --gen 二选一")
    p.add_argument("--gen", action="store_true", help="按 num_sources x num_each_dest 重新采样点对")
    p.add_argument("--num_sources", type=int, default=200)
    p.add_argument("--num_each_dest", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_file", required=True)
    p.add_argument("--cache_dir", default=os.path.join(PROJECT_ROOT, "outputs", "geo_cache"))
    p.add_argument("--limit_sources", type=int, default=0, help="只算前 N 个不同源点（用于分批跑）")
    p.add_argument("--max_minutes", type=float, default=0.0, help="超过该分钟数就安全停下（便于分批续跑）")
    p.add_argument("--with_graph_label", action="store_true",
                   help="gen 模式下也顺手算网格图 Dijkstra 标签（大图上极慢，默认关闭；诊断用）")
    p.add_argument("--out_root", default=os.path.join(PROJECT_ROOT, "outputs", "geo_results"))
    return p


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                pairs.append((int(parts[0]), int(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return pairs


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    if args.gen:
        # 先建图只是为了拿到顶点数
        V, _F = load_off(args.off_file)
        n = len(V)
        rng = random.Random(args.seed)
        # 关键：**不要把 (s,t) 规范化成 s<t**！否则源点集合会爆炸（每个 min(s,t) 都变成新源点），
        # 一次 SSSD 覆盖全图的优势就没了。这里保留"采样时的源点"，
        # 一个源点跑一次 SSSD 就能给它的所有目标点出距离。
        pair_set = set()
        for _ in range(args.num_sources):
            s = rng.randrange(n)
            for _ in range(args.num_each_dest):
                t = rng.randrange(n)
                if t == s:
                    continue
                pair_set.add((s, t))
        pairs = [(a, b, float("nan")) for a, b in sorted(pair_set)]
        print("[exact] generated %d unique pairs (%d sources x %d dests)"
              % (len(pairs), args.num_sources, args.num_each_dest))
    else:
        if not args.pairs_file:
            print("[exact] 需要 --pairs_file 或 --gen")
            return 1
        pairs = read_pairs(args.pairs_file)
        print("[exact] loaded %d pairs from %s" % (len(pairs), args.pairs_file))

    eg = ExactGeodesic(args.off_file, args.cache_dir)
    print("[exact] mesh |V|=%d |F|=%d build=%.2fs cache=%s"
          % (eg.n, len(eg.F), eg.build_seconds, eg.cache_dir))

    # 选择"从哪个端点跑 SSSD"：贪心覆盖，尽量让一次 SSSD 服务尽可能多的点对
    # （曲面测地距离是对称的，d(s,t)=d(t,s)，所以从任一端点算都可以）
    from collections import defaultdict
    popularity = defaultdict(int)
    for a, b, _ in pairs:
        popularity[a] += 1
        popularity[b] += 1
    by_src = {}
    assigned = 0
    for a, b, _ in pairs:
        if a in by_src:
            by_src[a].append(b)
        elif b in by_src:
            by_src[b].append(a)
        else:
            # 选"人气更高"的端点当源点，提高后续复用概率
            src, dst = (a, b) if popularity[a] >= popularity[b] else (b, a)
            by_src.setdefault(src, []).append(dst)
            assigned += 1
    sources = sorted(by_src.keys())
    print("[exact] greedy source selection: %d SSSD runs for %d pairs (naive would be %d)"
          % (len(sources), len(pairs), len({a for a, _, _ in pairs})))
    if args.limit_sources and len(sources) > args.limit_sources:
        sources = sources[:args.limit_sources]
        print("[exact] limited to first %d sources" % len(sources))

    # 已有缓存的行不需要重算，先估算剩余源点
    todo = [s for s in sources if not os.path.exists(eg._cache_path(s))]
    print("[exact] %d distinct sources, %d already cached, %d to compute"
          % (len(sources), len(sources) - len(todo), len(todo)))
    per_src_guess = 14.0 if eg.n > 100000 else 0.02
    if todo:
        print("[exact] %d 个源点待算；粗估 %.1f 分钟（首次实测后会打印真实速率与 ETA）"
              % (len(todo), len(todo) * per_src_guess / 60.0))

    t_start = time.time()
    rows = []
    done = 0
    for s in sources:
        if args.max_minutes and (time.time() - t_start) / 60.0 > args.max_minutes:
            print("[exact] 达到 --max_minutes %.1f，安全停下（已写 %d 行）" % (args.max_minutes, len(rows)))
            break
        d = eg.row(s)
        for t in by_src[s]:
            gd = float(d[t])
            rows.append((s, t, float("nan"), gd if np.isfinite(gd) else float("nan"), euclid3d(eg.V, s, t)))
        done += 1
        if done % 25 == 0 or done == len(sources):
            el = time.time() - t_start
            rate = el / max(1, done)
            print("[exact] %d/%d sources  elapsed=%.1fs  %.2fs/src  eta=%.1fmin"
                  % (done, len(sources), el, rate, (len(sources) - done) * rate / 60.0), flush=True)

    # 补上图最短路标签（只看 pairs_file 里给的真值；gen 模式下现算，小图才划算）
    graph_vals = {}
    if not args.gen:
        for a, b, g in pairs:
            graph_vals[(a, b)] = g
    elif args.with_graph_label:
        need = sorted({a for a, b, _ in rows})
        adj = build_graph_adj(eg.V, eg.F)
        for a in need:
            dg = dijkstra_row(adj, a)
            for s, t, _, _, _ in rows:
                if s == a:
                    graph_vals[(s, t)] = float(dg[t])
    else:
        print("[exact] (gen 模式默认不算网格图 Dijkstra 标签；需要时加 --with_graph_label)")

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("s,t,graph_dist,exact_geo,euclid_3d\n")
        for s, t, _, eg_d, eu in rows:
            g = graph_vals.get((s, t), float("nan"))
            f.write("%d,%d,%s,%s,%.6f\n" % (s, t, ("%.6f" % g) if np.isfinite(g) else "", eg_d, eu))
    print("[exact] wrote %d rows -> %s" % (len(rows), args.out_file))

    # ---- 诊断：图最短路标签 vs 精确测地；欧氏 vs 精确测地 ----
    gv = np.array([graph_vals.get((s, t), np.nan) for s, t, _, _, _ in rows], dtype=np.float64)
    ev = np.array([r[3] for r in rows], dtype=np.float64)
    uv = np.array([r[4] for r in rows], dtype=np.float64)
    ok = np.isfinite(gv) & np.isfinite(ev) & (ev > 1e-9)

    def stat(name, pred, ref):
        m = np.isfinite(pred) & np.isfinite(ref) & (ref > 1e-9)
        if m.sum() == 0:
            return
        rel = np.abs(pred[m] - ref[m]) / ref[m]
        ratio = pred[m] / ref[m]
        print("[exact] %-22s rel_err mean=%.6f  median=%.6f  MAE=%.2f  |  ratio median=%.4f P05=%.4f P95=%.4f"
              % (name, rel.mean(), np.median(rel), np.abs(pred[m] - ref[m]).mean(),
                 np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95)))

    print("[exact] === 诊断（reference = 精确 MMP 曲面测地距离）===")
    stat("graph-Dijkstra label", gv, ev)
    stat("euclidean_3d", uv, ev)
    if ok.sum():
        print("[exact] 图最短路 比 精确测地 短/长的比例：短 %.2f%%  长 %.2f%%  相等 %.2f%%"
              % (100.0 * (gv[ok] < ev[ok] - 1e-6).mean(),
                 100.0 * (gv[ok] > ev[ok] + 1e-6).mean(),
                 100.0 * (np.abs(gv[ok] - ev[ok]) <= 1e-6).mean()))
    print("[exact] total %.1fs, computed %d new SSSD rows" % (time.time() - t_start, eg.n_computed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
