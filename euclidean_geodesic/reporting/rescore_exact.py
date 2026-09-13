# -*- coding: utf-8 -*-
"""把模型预测 + 精确测地真值合起来重新打分。

回答三个问题：
  1. 模型对着**原来的图最短路标签**误差多少（= 我们之前报的数字）
  2. 模型对着**真实曲面测地**误差多少（= 诚实的数字）
  3. 同一个测试集上，无学习的 3D 欧氏基线对着**真实曲面测地**误差多少

用法：
  python3 scripts_local/rescore_exact.py \
      --labels outputs/geo_results/<tag>_exact.csv \
      --preds  outputs/geo_results/<tag>_preds.csv \
      [--name M0_full]
"""
import argparse
import csv
import math


def read_csv(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows[(int(d["s"]), int(d["t"]))] = d
    return rows


def metrics(pred, ref):
    n = len(ref)
    ae = [abs(p - r) for p, r in zip(pred, ref)]
    rel = [a / (r + 1e-9) for a, r in zip(ae, ref)]
    return {
        "n": n,
        "rel_mean": sum(rel) / n,
        "rel_median": sorted(rel)[n // 2],
        "mae": sum(ae) / n,
        "rmse": math.sqrt(sum(a * a for a in ae) / n),
        "bias": sum(p - r for p, r in zip(pred, ref)) / n,
    }


def fmt(m):
    return "rel_mean=%.6f  rel_med=%.6f  MAE=%.2f  RMSE=%.2f  bias=%+.2f" % (
        m["rel_mean"], m["rel_median"], m["mae"], m["rmse"], m["bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", default="model")
    args = ap.parse_args()

    L = read_csv(args.labels)
    P = read_csv(args.preds)
    keys = [k for k in L.keys() if k in P]
    graph, exact, eucl, pred = [], [], [], []
    for k in keys:
        d = L[k]
        try:
            g = float(d["graph_dist"]) if d.get("graph_dist") not in (None, "") else float("nan")
            e = float(d["exact_geo"]); u = float(d["euclid_3d"]); p = float(P[k]["pred"])
        except (ValueError, KeyError):
            continue
        if not (math.isfinite(e) and math.isfinite(u) and math.isfinite(p)):
            continue
        graph.append(g if math.isfinite(g) else float("nan"))
        exact.append(e); eucl.append(u); pred.append(p)

    n = len(exact)
    print("=== %s  (%d pairs) ===" % (args.name, n))
    print("  [参考=图最短路标签]  %-10s %s" % (args.name, fmt(metrics(pred, graph))))
    print("  [参考=精确曲面测地]  %-10s %s" % (args.name, fmt(metrics(pred, exact))))
    print("  [参考=精确曲面测地]  %-10s %s" % ("euclidean_3d", fmt(metrics(eucl, exact))))
    print("  [参考=图最短路标签]  %-10s %s" % ("euclidean_3d", fmt(metrics(eucl, graph))))
    print("  -- 标签本身 --")
    print("  [参考=精确曲面测地]  %-10s %s" % ("graph label", fmt(metrics(graph, exact))))
    print("  -- 关键对比 --")
    m_pred_exact = metrics(pred, exact)["rel_mean"]
    m_eucl_exact = metrics(eucl, exact)["rel_mean"]
    print("  模型 vs 精确测地 = %.4f%%   |  纯欧氏 vs 精确测地 = %.4f%%   |  模型是欧氏的 %.2f 倍"
          % (m_pred_exact * 100, m_eucl_exact * 100, m_pred_exact / max(1e-12, m_eucl_exact)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
