# -*- coding: utf-8 -*-
"""用热方法（MeshHeatMethodDistanceSolver）回填 flip-out 崩溃的少数点对。

用法（分片并行）:
  python surface_poison_fill.py --off_file X.off --main_csv main.csv --shard 0 --shards 6
合并:
  python surface_poison_fill.py --main_csv main.csv --merge
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_heat(off_path):
    import potpourri3d as pp3d
    npz = off_path + ".npz"
    if os.path.exists(npz):
        z = np.load(npz)
        V = np.asarray(z["V"], dtype=np.float64)
        F = np.asarray(z["F"], dtype=np.int64)
    else:
        from build_highway import load_off
        V, F = load_off(off_path)
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
    return pp3d.MeshHeatMethodDistanceSolver(V, F)


def is_bad(v):
    return (v is None) or (not math.isfinite(v))


def read_main(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                s, t = int(r["s"]), int(r["t"])
            except Exception:
                continue
            try:
                d = float(r["true_distance"])
            except Exception:
                d = float("nan")
            rows.append([s, t, d, r.get("graph_distance", ""), r.get("label_method", "unknown")])
    return rows


def merge(main_csv):
    fill = {}
    for p in sorted(glob.glob(main_csv + ".poisonfill*")):
        with open(p, "rb") as source:
            data = source.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n")
            with open(p, "wb") as target:
                target.write(data[:cut + 1] if cut >= 0 else b"")
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    fill[(int(r["s"]), int(r["t"]))] = float(r["d"])
                except Exception:
                    continue
    rows = read_main(main_csv)
    filled = 0
    for row in rows:
        if is_bad(row[2]) and (row[0], row[1]) in fill:
            row[2] = fill[(row[0], row[1])]
            row[4] = "heat"
            filled += 1
    tmp = main_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "t", "true_distance", "graph_distance", "label_method"])
        for s, t, d, graph_d, method in rows:
            w.writerow([s, t, "%.6f" % d if math.isfinite(d) else "inf", graph_d, method])
    os.replace(tmp, main_csv)
    bad_left = len([1 for r in rows if is_bad(r[2])])
    for p in glob.glob(main_csv + ".poisonfill*"):
        try:
            os.remove(p)
        except OSError:
            pass
    print("[poison-merge] 回填 %d 对，仍缺 %d 对，总行数 %d" % (filled, bad_left, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_file")
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge(args.main_csv)
        return

    rows = read_main(args.main_csv)
    bad = [(i, s, t) for i, (s, t, d, _, _) in enumerate(rows) if is_bad(d)]
    mine = [(s, t) for i, s, t in bad if i % args.shards == args.shard]
    out_path = args.main_csv + ".poisonfill%d" % args.shard
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    done.add((int(r["s"]), int(r["t"])))
                except Exception:
                    continue
    todo = [p for p in mine if p not in done]
    print("[fill%d] 需回填 %d 对（本分片 %d，待算 %d）" % (args.shard, len(bad), len(mine), len(todo)), flush=True)
    if not todo:
        return
    solver = build_heat(args.off_file)
    new_file = not os.path.exists(out_path)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8-sig", newline="", buffering=1) as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["s", "t", "d"])
        for k, (s, t) in enumerate(todo):
            try:
                d = float(solver.compute_distance(int(s))[int(t)])
            except Exception:
                d = float("nan")
            w.writerow([int(s), int(t), "%.6f" % d])
            f.flush()
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print("[fill%d] %d/%d 用时%.0fs 速率%.2f对/s" % (args.shard, k + 1, len(todo), el, (k + 1) / max(el, 1e-9)), flush=True)
    print("[fill%d] 完成 %d 对" % (args.shard, len(todo)), flush=True)


if __name__ == "__main__":
    main()
