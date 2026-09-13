# -*- coding: utf-8 -*-
"""把 surface_*.csv.shard* 合并进主 CSV（去重、保留已完成的旧行）。"""
import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    args = ap.parse_args()

    rows = {}
    if os.path.exists(args.main):
        with open(args.main, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    rows[(int(r["s"]), int(r["t"]))] = float(r["true_distance"])
                except Exception:
                    continue
    before = len(rows)
    shards = sorted(glob.glob(args.main + ".shard*"))
    for sp in shards:
        data = open(sp, "rb").read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n")
            open(sp, "wb").write(data[:cut + 1] if cut >= 0 else b"")
        with open(sp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    rows[(int(r["s"]), int(r["t"]))] = float(r["true_distance"])
                except Exception:
                    continue
    tmp = args.main + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "t", "true_distance", "graph_distance"])
        for (s, t), d in rows.items():
            w.writerow([s, t, "%.6f" % d, ""])
    os.replace(tmp, args.main)
    for sp in shards:
        try:
            os.remove(sp)
        except OSError:
            pass
    print("[merge] shards=%d 合并后 %d 对（原有 %d）" % (len(shards), len(rows), before))


if __name__ == "__main__":
    main()
