# -*- coding: utf-8 -*-
"""用 flip-out 精确测地算法重算点对距离标签（真实 terrain 表面距离）。

用法:
  python surface_distance_labels.py --off_file X.off --pairs_csv in.csv --out_csv out.csv --workers 8
说明:
  - 输入 pairs_csv 至少包含 s,t 两列（可含旧的 graph distance 列作对照）
  - 输出列: s,t,true_distance,graph_distance(若输入有)
  - 每对用 potpourri3d.EdgeFlipGeodesicSolver.find_geodesic_path 求精确表面测地距离
  - 支持断点续跑：若 out_csv 已存在，跳过已完成的 pair
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_W_V = None
_W_F = None
_W_SOLVER = None


def _init(off_path):
    global _W_V, _W_F, _W_SOLVER
    import potpourri3d as pp3d
    npz = off_path + ".npz"
    if os.path.exists(npz):
        z = np.load(npz)
        _W_V = np.asarray(z["V"], dtype=np.float64)
        _W_F = np.asarray(z["F"], dtype=np.int64)
    else:
        from build_highway import load_off
        V, F = load_off(off_path)
        _W_V = np.asarray(V, dtype=np.float64)
        _W_F = np.asarray(F, dtype=np.int64)
    _W_SOLVER = pp3d.EdgeFlipGeodesicSolver(_W_V, _W_F)


def _work(pairs):
    out = []
    for item in pairs:
        s, t = item[0], item[1]
        try:
            path = _W_SOLVER.find_geodesic_path(int(s), int(t))
            d = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        except Exception:
            d = float("inf")
        out.append((int(s), int(t), d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_file", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 对（调试用）")
    args = ap.parse_args()

    rows = []
    with open(args.pairs_csv, encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        has_graph = "distance" in (rdr.fieldnames or [])
        for row in rdr:
            rows.append((int(row["s"]), int(row["t"]), float(row["distance"]) if has_graph else None))
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done.add((int(row["s"]), int(row["t"])))
        print(f"[surface] 断点续跑：已完成 {len(done)} 对")
    todo = [r for r in rows if (r[0], r[1]) not in done]
    print(f"[surface] 总对={len(rows)} 待算={len(todo)} 网格={os.path.basename(args.off_file)}", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    tasks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    t0 = time.time()
    written = 0
    mode = "a" if done else "w"
    with open(args.out_csv, mode, encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["s", "t", "true_distance", "graph_distance"])
        with ctx.Pool(processes=args.workers, initializer=_init, initargs=(args.off_file,)) as pool:
            for res in pool.imap_unordered(_work, tasks, chunksize=1):
                gm = {(r[0], r[1]): r[2] for r in res}
                for s, t, g in res:
                    key = (s, t)
                    orig = next((x[2] for x in rows if x[0] == s and x[1] == t), None)
                    w.writerow([s, t, f"{g:.6f}", "" if orig is None else f"{orig:.6f}"])
                f.flush()
                written += len(res)
                if written % (args.chunk * 10) == 0 or written == len(todo):
                    el = time.time() - t0
                    rate = written / max(1e-9, el)
                    print(f"[surface] {written}/{len(todo)} 用时{el:.0f}s 速率{rate:.1f}对/s "
                          f"预计剩余{(len(todo)-written)/max(1e-9, rate)/60:.1f}分钟", flush=True)
    print(f"[surface] 完成 -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
