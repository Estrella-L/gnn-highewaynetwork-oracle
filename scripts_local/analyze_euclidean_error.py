# -*- coding: utf-8 -*-
"""实验 4：量化「欧氏距离」本身的误差（不做任何学习）。

回答的问题：3D 直线距离作为最短路(测地)距离的锚点，到底差多少？
如果它已经足够准，残差学习的价值就有限；如果它系统性低估，就能解释为什么
"以欧氏距离为中心 + 有界乘性修正" 远好于 "直接回归距离"。

在同一批 test 点对上计算：
  euclidean_3d / euclidean_2d / mean_constant 的 mae / rmse / relative_error / 偏差；
  真实比值 r = d_geo / d_euclid 的分位数（r > 1 表示测地距离长于直线距离）；
  按同分区/跨分区、短/中/长距离分组；
  以及"残差修正量" (d_geo - d_euclid)/d_geo 的分布，
  即主方法的 0.5*tanh 有界修正需要覆盖多大范围（含落在可表达区间之外的比例）。

用法：
    python3 scripts_local/analyze_euclidean_error.py \
        --off_file <terrain.off> \
        --test_pairs_file outputs/results/<run>_test_pairs.csv \
        --out_prefix /path/to/消融实验/results/<tag>_euclid
"""
import argparse
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from build_highway import load_off, build_graph_and_partition  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="欧氏距离误差量化（实验4）")
    p.add_argument("--off_file", required=True)
    p.add_argument("--test_pairs_file", required=True)
    p.add_argument("--out_prefix", default="", help="输出文件前缀（不含扩展名）")
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--capacity", type=int, default=512)
    p.add_argument("--uniform", action="store_true")
    return p


def load_pairs(path):
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


def metrics(true, pred):
    n = len(true)
    ae = [abs(p - t) for p, t in zip(pred, true)]
    return {
        "mae": sum(ae) / n,
        "rmse": math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / n),
        "relative_error": sum(a / (t + 1e-9) for a, t in zip(ae, true)) / n,
        "mean_bias": sum(p - t for p, t in zip(pred, true)) / n,
        "mean_bias_ratio": sum((p - t) / (t + 1e-9) for p, t in zip(pred, true)) / n,
    }


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main():
    args = build_parser().parse_args()
    off_path = args.off_file
    vertices, _ = load_off(off_path)
    pairs = load_pairs(args.test_pairs_file)
    if not pairs:
        print("[euclid] 测试集为空")
        return 1
    graph_info, coords, leaf_of, num_leaves = build_graph_and_partition(
        off_path, max_depth=args.max_depth, capacity=args.capacity, adaptive=not args.uniform
    )

    y_true = [d for (_, _, d) in pairs]
    mean_const = sum(y_true) / len(y_true)
    e3, e2, ratios, corrections = [], [], [], []
    for s, t, d in pairs:
        (xs, ys, zs), (xt, yt, zt) = vertices[s], vertices[t]
        d3 = math.sqrt((xs - xt) ** 2 + (ys - yt) ** 2 + (zs - zt) ** 2)
        d2 = math.hypot(xs - xt, ys - yt)
        e3.append(d3)
        e2.append(d2)
        if d3 > 1e-9:
            ratios.append(d / d3)
            corrections.append((d - d3) / d)

    preds = {"euclidean_3d": e3, "euclidean_2d": e2, "mean_constant": [mean_const] * len(pairs)}
    L = []
    L.append("# 欧氏距离误差量化（实验 4）")
    L.append("")
    L.append("- 地形：@@BT@@" + os.path.basename(off_path) + "@@BT@@（|V|=" + str(len(vertices)) +
             "，四叉树 d" + str(args.max_depth) + "/c" + str(args.capacity) + "，叶子 " + str(num_leaves) + "）")
    L.append("- 测试点对：@@BT@@" + os.path.basename(args.test_pairs_file) + "@@BT@@（" + str(len(pairs)) +
             " 对，真值 = 网格图最短路 / 测地处近似）")
    L.append("")
    L.append("## 1. 无学习基线的整体误差")
    L.append("")
    L.append("| 基线 | rel.err | MAE | RMSE | 平均偏差 | 平均相对偏差 |")
    L.append("|---|---|---|---|---|---|")
    for name in ["euclidean_3d", "euclidean_2d", "mean_constant"]:
        m = metrics(y_true, preds[name])
        L.append("| " + name + " | {:.6f} | {:.2f} | {:.2f} | {:+.2f} | {:+.4f} |".format(
            m["relative_error"], m["mae"], m["rmse"], m["mean_bias"], m["mean_bias_ratio"]))

    L.append("")
    L.append("## 2. 测地距离 / 3D 直线距离 的比值分布 r = d_geo / d_euclid")
    L.append("")
    L.append("- 均值 {:.4f}，中位数 {:.4f}".format(sum(ratios) / len(ratios), pct(ratios, 0.5)))
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        L.append("- P{:02d} = {:.4f}".format(int(q * 100), pct(ratios, q)))
    over = sum(1 for r in ratios if r > 1.0) / len(ratios)
    L.append("- 比值 > 1（测地长于直线，即 3D 直线必然低估）的比例：{:.2f}%".format(over * 100))

    L.append("")
    L.append("## 3. 残差修正量 (d_geo - d_euclid)/d_geo 的分布")
    L.append("")
    L.append("主方法的欧氏残差使用 @@BT@@d_hat = d_3D * (1 + 0.5*tanh(f))@@BT@@，可表达区间为 **[0.5, 1.5] x d_3D**，"
             "即修正量落在 [-0.5, +0.5]。")
    L.append("")
    for q in [0.01, 0.05, 0.5, 0.95, 0.99]:
        L.append("- P{:02d} = {:+.4f}".format(int(q * 100), pct(corrections, q)))
    oob = sum(1 for c in corrections if c > 0.5 or c < -0.5) / len(corrections)
    L.append("- 落在 [-0.5, 0.5] 之外（主方法参数化无法精确表达）的比例：{:.2f}%".format(oob * 100))

    L.append("")
    L.append("## 4. 分组统计（3D 欧氏基线 vs 真实测地距离）")
    L.append("")
    groups = {"same_leaf": [], "cross_leaf": [], "short_dist": [], "mid_dist": [], "long_dist": []}
    sorted_d = sorted(y_true)
    q1, q2 = sorted_d[len(sorted_d) // 3], sorted_d[(2 * len(sorted_d)) // 3]
    for idx, (s, t, d) in enumerate(pairs):
        groups["same_leaf" if leaf_of.get(s) == leaf_of.get(t) else "cross_leaf"].append(idx)
        groups["short_dist" if d <= q1 else ("mid_dist" if d <= q2 else "long_dist")].append(idx)
    L.append("| 分组 | 对数 | 3D 欧氏 rel.err | 3D 欧氏 MAE | 比值中位数 |")
    L.append("|---|---|---|---|---|")
    for name, idxs in groups.items():
        if not idxs:
            continue
        m = metrics([y_true[i] for i in idxs], [e3[i] for i in idxs])
        r_sub = [ratios[i] for i in idxs if i < len(ratios)]
        L.append("| " + name + " | " + str(len(idxs)) + " | {:.6f} | {:.2f} | {:.4f} |".format(
            m["relative_error"], m["mae"], pct(r_sub, 0.5)))

    text = "\n".join(L).replace("@@BT@@", chr(96))
    print(text)
    if args.out_prefix:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
        with open(args.out_prefix + ".md", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("\n[euclid] 已写入 " + args.out_prefix + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
