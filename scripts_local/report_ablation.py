# -*- coding: utf-8 -*-
"""把消融实验的 results.jsonl 汇总成可直接贴进论文/组会 PPT 的 Markdown 报告。

输出内容：
  1. 总体效果 + 逐项解剖的指标表（均值 ± 标准差，跨种子）
  2. 相对 M0_full 的退化幅度（绝对 pp、相对百分比、放大倍数）
  3. 4 项解剖实验的逐条结论（自动生成文字）
  4. 分组（同分区 / 跨分区、短 / 中 / 长）明细

用法：
    python3 scripts_local/report_ablation.py \
        --jsonl /path/to/消融实验/results/local_small_results.jsonl \
        --out   /path/to/消融实验/消融实验结果_small_terrain.md \
        --title "small_terrain.off（1221 顶点）"
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

MAIN_ORDER = ["M0_full", "A1_no_euclid", "A2_no_part_hw", "A3_no_both"]
EXTRA_ORDER = ["A1_hwfeat", "A2_deep", "A3_deep", "A1_rel", "A3_rel"]
BASELINE_ORDER = ["A4_euclidean_2d", "A4_euclidean_3d", "A4_mean_constant", "A4_highway_decomp"]

CONFIG_NOTES = {
    "M0_full": ("总体方法（完整）", "四叉树地形分区 + highway 网络 + Inner/Inter GNN + Fusion MLP + 3D 欧氏距离残差"),
    "A1_no_euclid": ("解剖 1：去掉欧氏距离残差", "保留地形分区 / highway / Inter-GNN / Inner-GNN，直接回归 terrain 表面 geodesic distance"),
    "A2_no_part_hw": ("解剖 2：去掉地形分区 + highway 网络", "只用一个全图 GNN 编码 s/t 后直接预测，但保留 3D 欧氏距离残差参数化"),
    "A3_no_both": ("解剖 3：同时去掉 1 和 2 的内容", "单 GNN 直接回归 geodesic distance，既无分区/highway 也无欧氏残差"),
    "A1_hwfeat": ("防守型对照：解剖 1 但开启 highway 分解距离特征",
                  "直接回归的模型仍拿到 access+highway+access 三段分解距离特征，排除「模型完全没有距离输入」的解释"),
    "A2_deep": ("防守型对照：解剖 2 但单 GNN 加深到 6 层", "排除「消融分支容量/深度不够」的解释"),
    "A3_deep": ("防守型对照：解剖 3 但单 GNN 加深到 6 层", "排除「消融分支容量/深度不够」的解释"),
    "A1_rel": ("补充：解剖 1 + relative 损失", "公平性对照，排除「损失函数没选好」的解释"),
    "A3_rel": ("补充：解剖 3 + relative 损失", "公平性对照"),
    "A4_euclidean_3d": ("解剖 4：纯 3D 欧氏距离（无学习）", "d = ||v_s - v_t||_2 直接当预测值"),
    "A4_euclidean_2d": ("解剖 4：纯 2D 平面欧氏距离（无学习）", "忽略高程 z 的 (x,y) 直线距离"),
    "A4_mean_constant": ("最强常数基线", "永远预测 test 真值均值"),
    "A4_highway_decomp": ("highway 分解距离基线（无学习）", "access + highway + access 三段分解"),
}


def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0, 1
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), len(vals)


def fmt_pm(m, s, digits=6):
    if m is None:
        return "-"
    if s is None or s == 0:
        return "{:.{d}f}".format(m, d=digits)
    return "{:.{d}f} ± {:.{d}f}".format(m, s, d=digits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--settings", default="", help="实验设置的一句话描述，直接写进报告")
    args = p.parse_args()

    recs = [r for r in load_jsonl(args.jsonl) if r.get("status") == "ok"]
    by_run = defaultdict(list)
    for r in recs:
        by_run[r["run_id"]].append(r)

    seeds = sorted({r["seed"] for r in recs})
    L = []
    L.append("# 消融实验报告（Ablation Study）")
    L.append("")
    if args.title:
        L.append("**数据集**：" + args.title)
    L.append("**种子**：" + str(seeds) + "（表中为均值 ± 标准差）")
    if args.settings:
        L.append("**设置**：" + args.settings)
    L.append("")
    L.append("> 消融逻辑：先给出总体方法的完整效果，再逐项拿掉方法中的技术组件，"
             "用**完全相同的采样、划分、测试集与训练预算**测出每个组件各自贡献了多少精度。")
    L.append("")

    def get(run_id, key):
        return mean_std([r.get(key) for r in by_run.get(run_id, [])])

    base_rel, _, _ = get("M0_full", "test_relative_error")
    euclid_rel, _, _ = get("A4_euclidean_3d", "test_relative_error")

    L.append("## 1. 总体效果与逐项解剖")
    L.append("")
    L.append("| 配置 | 说明 | test 相对误差 | test MAE | test RMSE | 相对 M0 退化 | vs 纯欧氏基线 |")
    L.append("|---|---|---|---|---|---|---|")
    order = [r for r in MAIN_ORDER if r in by_run] + [r for r in EXTRA_ORDER if r in by_run] \
        + [r for r in BASELINE_ORDER if r in by_run]
    for rid in order:
        label, note = CONFIG_NOTES.get(rid, (rid, ""))
        m_rel, s_rel, _ = get(rid, "test_relative_error")
        m_mae, _, _ = get(rid, "test_mae")
        m_rmse, _, _ = get(rid, "test_rmse")
        deg = "-"
        if base_rel and m_rel is not None:
            if rid == "M0_full":
                deg = "基准"
            else:
                ratio = m_rel / base_rel
                deg = "+{:.1f}%（{:.2f}×）".format(100 * (ratio - 1), ratio)
        vs_e = "-"
        if euclid_rel and m_rel is not None:
            vs_e = "{:.2f}×".format(m_rel / euclid_rel)
            if rid.startswith("A4_"):
                vs_e = "基准"
        L.append("| @@B@@" + rid + "@@B@@ | " + label + " | " + fmt_pm(m_rel, s_rel) + " | " +
                 ("-" if m_mae is None else "{:.2f}".format(m_mae)) + " | " +
                 ("-" if m_rmse is None else "{:.2f}".format(m_rmse)) + " | " + deg + " | " + vs_e + " |")
    L.append("")

    L.append("## 2. 分组明细（同分区 / 跨分区、短 / 中 / 长距离）")
    L.append("")
    L.append("| 配置 | same-leaf rel | cross-leaf rel | short rel | mid rel | long rel |")
    L.append("|---|---|---|---|---|---|")
    for rid in order:
        cells = []
        for g in ["group_same_leaf_relative_error", "group_cross_leaf_relative_error",
                  "group_short_dist_relative_error", "group_mid_dist_relative_error",
                  "group_long_dist_relative_error"]:
            m, s, _ = get(rid, g)
            cells.append(fmt_pm(m, s))
        L.append("| @@B@@" + rid + "@@B@@ | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. 逐条结论（由数据自动生成，供人工核对措辞）")
    L.append("")
    if base_rel:
        def delta(rid):
            m, _, _ = get(rid, "test_relative_error")
            if m is None:
                return None
            return m, m / base_rel, 100 * (m / base_rel - 1)

        d1 = delta("A1_no_euclid")
        d2 = delta("A2_no_part_hw")
        d3 = delta("A3_no_both")
        d4 = delta("A4_euclidean_3d")
        L.append("- **总体效果**：完整方法 test 相对误差 = {:.6f}（{:.4f}%），MAE/RMSE 见上表。".format(
            base_rel, base_rel * 100))
        if d1:
            L.append("- **解剖 1（去掉欧氏距离残差）**：相对误差 {:.6f}，是完整方法的 {:.2f} 倍"
                     "（恶化 {:.1f}%）。说明欧氏残差是精度的主要来源。".format(d1[0], d1[1], d1[2]))
        if d2:
            L.append("- **解剖 2（去掉地形分区 + highway 网络）**：相对误差 {:.6f}，是完整方法的 {:.2f} 倍"
                     "（恶化 {:.1f}%）。说明分区 + highway 分层结构的增益。".format(d2[0], d2[1], d2[2]))
        if d3:
            L.append("- **解剖 3（两者都去掉）**：相对误差 {:.6f}，是完整方法的 {:.2f} 倍"
                     "（恶化 {:.1f}%）。".format(d3[0], d3[1], d3[2]))
        if d1 and d2 and d3:
            interaction = d3[1] - d1[1] * d2[1]
            L.append("- **组件交互**：两者同时去掉的退化倍数 {:.2f}×，而单独退化倍数的乘积为 {:.2f}×，".format(
                d3[1], d1[1] * d2[1]) + ("存在正向协同（1+1>2）" if interaction > 0 else "存在冗余（1+1<2）") + "。")
        if d4:
            L.append("- **解剖 4（纯 3D 欧氏距离，无学习）**：相对误差 {:.6f}（{:.2f}%）。"
                     "这就是「残差锚点」本身的误差水平——完整方法把它压到 {:.4f}%，"
                     "相当于把欧氏距离的误差降低了 {:.2f}%。".format(
                         d4[0], d4[0] * 100, base_rel * 100, (1 - base_rel / d4[0]) * 100))
        if euclid_rel:
            worse = []
            for rid in ["A1_no_euclid", "A1_hwfeat", "A2_no_part_hw", "A3_no_both", "A2_deep", "A3_deep",
                        "A1_rel", "A3_rel"]:
                mm, _, _ = get(rid, "test_relative_error")
                if mm is not None and mm > euclid_rel:
                    worse.append(rid)
            tail = ("以下配置**比它更差**：" + "、".join(worse) +
                    "——说明拿掉这些组件后，模型甚至不如直接量一条直线。") if worse else "所有学习配置都优于该基线。"
            L.append("- **关键判据**：纯 3D 欧氏距离（不做任何学习）的相对误差是 {:.6f}（{:.2f}%）。".format(
                euclid_rel, euclid_rel * 100) + tail)
        for extra in EXTRA_ORDER:
            if extra in by_run:
                de = delta(extra)
                if de:
                    L.append("- **补充对照 " + extra + "**：相对误差 {:.6f}（{:.2f}× M0）。".format(de[0], de[1]))
    L.append("")
    L.append("## 4. 如何复现")
    L.append("")
    L.append("@@B3@@bash")
    L.append("# 本地（CPU）")
    L.append("python3 scripts_local/run_ablation.py --off_file <terrain.off> --tag <tag> \\")
    L.append("    --seeds 42 43 44 --runs all --max_depth 3 --capacity 128 \\")
    L.append("    --distance_samples 10000 --num_epoch 80 --device cpu")
    L.append("")
    L.append("# 云端（EP_low 全量，需 AutoDL 实例开机）")
    L.append("python3 cloud_tools/ablation_launch.py --launch --tag ep_low \\")
    L.append("    --runs A1_no_euclid,A2_no_part_hw,A3_no_both --samples 100000 --epochs 120")
    L.append("")
    L.append("# 汇总成报告")
    L.append("python3 scripts_local/report_ablation.py --jsonl <tag>_results.jsonl --out report.md")
    L.append("@@B3@@")
    L.append("")

    text = "\n".join(L).replace("@@B3@@", chr(96) * 3).replace("@@B@@", chr(96))
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n[report] 已写入 " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
