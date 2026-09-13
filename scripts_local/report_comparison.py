# -*- coding: utf-8 -*-
"""把「我方方法 + 三篇 baseline」在**同一套精确测地标签**下的结果汇总成对比表。

读取若干 *_results.jsonl，合并后按方法类别排序，输出：
  - 各方法的 test 相对误差 / MAE / RMSE（均值 ± 标准差，跨种子）
  - 无学习参照（3D 欧氏直线）在同一测试集上的误差，作为"及格线"
  - 相对我方完整方法的倍数

用法：
  python3 scripts_local/report_comparison.py \
      --jsonl ../消融实验/results/exact_small_results.jsonl \
      --jsonl ../消融实验/results/exact_base_results.jsonl \
      --labels outputs/geo_results/small_terrain_exact_trainval.csv \
      --out ../消融实验/对比结果_exact_small.md \
      --title "small_terrain.off（精确曲面测地标签）"
"""
import argparse
import csv
import json
import math
from collections import defaultdict

ORDER = [
    ("M0_full", "**我方方法（完整）**", "四叉树分区 + highway + Inner/Inter GNN + Fusion + 3D 欧氏残差"),
    ("A1_no_euclid", "我方：去掉欧氏残差", "保留分区/highway/双 GNN，直接回归距离"),
    ("A2_no_part_hw", "我方：去掉分区+highway（单 GNN）", "单 GNN + 欧氏残差"),
    ("A3_no_both", "我方：两者都去掉", "单 GNN 直接回归"),
    ("B_gegnn", "**GeGnn（原样输出）**", "整图 GeoConv(max聚合, 相对位置+边长) -> 256维嵌入 -> MLP((e_i-e_j)^2)"),
    ("B_neurogf", "**NeuroGF（原样输出）**", "逐点 lifting FC(3->64->128->256) -> |e_s-e_t| -> MLP"),
    ("B_litege", "**LiteGE（原样输出）**", "CoordMLP + UDF-PCA 描述子 -> (e_s-e_t) -> MLP -> abs()"),
    ("B_gegnn_euc", "GeGnn + 欧氏残差输出", "网络同 GeGnn，输出参数化换成我方 d_3D*(1+0.5tanh(f))"),
    ("B_neurogf_euc", "NeuroGF + 欧氏残差输出", "网络同 NeuroGF，输出参数化换成我方"),
    ("B_litege_euc", "LiteGE + 欧氏残差输出", "网络同 LiteGE，输出参数化换成我方"),
]


def load_jsonl(paths):
    recs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            print("[cmp] 缺少文件（跳过）：" + p)
    return [r for r in recs if r.get("status") == "ok"]


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = 0.0 if len(vals) == 1 else math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, s


def euclid_reference(labels_path):
    """从标签 CSV 直接算无学习 3D 欧氏距离的误差（同一测试集口径由调用方保证）。"""
    if not labels_path:
        return None
    import numpy as np
    rel, mae = [], []
    with open(labels_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                e = float(row["exact_geo"]); u = float(row["euclid_3d"])
            except (ValueError, KeyError):
                continue
            if e > 1e-9:
                rel.append(abs(u - e) / e)
                mae.append(abs(u - e))
    if not rel:
        return None
    return sum(rel) / len(rel), sum(mae) / len(mae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--settings", default="")
    args = ap.parse_args()

    recs = load_jsonl(args.jsonl)
    by_run = defaultdict(list)
    for r in recs:
        by_run[r["run_id"]].append(r)
    seeds = sorted({r["seed"] for r in recs})

    base_rel, _ = mean_std([r.get("test_relative_error") for r in by_run.get("M0_full", [])])
    euc = euclid_reference(args.labels)

    L = []
    L.append("# 与三篇 baseline 的对比（同一套精确曲面测地标签）")
    L.append("")
    if args.title:
        L.append("**数据集**：" + args.title)
    L.append("**种子**：" + str(seeds))
    if args.settings:
        L.append("**设置**：" + args.settings)
    L.append("")
    L.append("> 真值 = pygeodesic 精确 MMP 曲面测地距离（与 GeGnn 同一套计算方式）。")
    L.append("> 所有方法共用同一 train/val/test 划分与同一测试集；指标为平均相对误差 MRE。")
    L.append("")
    L.append("| 方法 | 说明 | test 相对误差 | MAE | RMSE | 相对我方完整方法 |")
    L.append("|---|---|---|---|---|---|")
    for rid, label, desc in ORDER:
        rs = by_run.get(rid, [])
        if not rs:
            continue
        m, s = mean_std([r.get("test_relative_error") for r in rs])
        mae, _ = mean_std([r.get("test_mae") for r in rs])
        rmse, _ = mean_std([r.get("test_rmse") for r in rs])
        rel_txt = "-" if m is None else ("%.6f" % m if s in (None, 0.0) else "%.6f ± %.6f" % (m, s))
        ratio = "-" if (m is None or not base_rel) else "%.2f×" % (m / base_rel)
        L.append("| " + label + " | " + desc + " | " + rel_txt + " | " +
                 ("-" if mae is None else "%.2f" % mae) + " | " +
                 ("-" if rmse is None else "%.2f" % rmse) + " | " + ratio + " |")
    if euc:
        L.append("| **纯 3D 欧氏直线（无学习）** | d = ||v_s - v_t||，不训练 | **%.6f** | %.2f | - | %s |"
                 % (euc[0], euc[1], "-" if not base_rel else "%.2f×" % (euc[0] / base_rel)))
    L.append("")
    L.append("## 结论")
    L.append("")
    if euc and base_rel:
        L.append("- 无学习参照（3D 直线）MRE = **%.4f%%**；我方完整方法 = **%.4f%%**。" % (euc[0] * 100, base_rel * 100))
    best = None
    for rid, label, _ in ORDER:
        rs = by_run.get(rid, [])
        if not rs:
            continue
        m, _ = mean_std([r.get("test_relative_error") for r in rs])
        if m is not None and (best is None or m < best[1]):
            best = (label, m)
    if best:
        L.append("- 当前最优的学习方法：**%s**（MRE = %.6f）。" % (best[0], best[1]))
    L.append("")

    text = "\n".join(L)
    print(text)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n[cmp] 已写入 " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
