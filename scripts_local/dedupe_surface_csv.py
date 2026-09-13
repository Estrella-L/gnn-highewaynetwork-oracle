# -*- coding: utf-8 -*-
"""对表面标签 CSV 去重（保留每个 (s,t) 的第一条），并统计 inf 数量。"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    args = ap.parse_args()

    seen = {}
    dup = 0
    inf = 0
    with open(args.main, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["s"]), int(r["t"]))
            except Exception:
                continue
            if key in seen:
                dup += 1
                continue
            val = r.get("true_distance", "")
            if val.strip().lower() in ("inf", "-inf", "nan", ""):
                inf += 1
            seen[key] = val
    tmp = args.main + ".dedup.tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "t", "true_distance", "graph_distance"])
        for (s, t), v in seen.items():
            w.writerow([s, t, v if v else "inf", ""])
    os.replace(tmp, args.main)
    print("[dedupe] 唯一 %d 对，去掉重复 %d 行，inf %d 个" % (len(seen), dup, inf))


if __name__ == "__main__":
    main()
