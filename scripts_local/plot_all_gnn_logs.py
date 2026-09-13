# -*- coding: utf-8 -*-
"""为所有 GNN 训练日志画图（延续 Codex 的 plot_gnn_experiment_logs.py 思路，扩展三种日志格式）。

输出：<out_dir>/figures/*.png + figures_index.md + experiment_summary_all.csv
用法：python plot_all_gnn_logs.py --out_dir ~/Desktop/GNN_实验图表
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体（找不到就用英文） ----
CJK = None
for f in ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti", "SimHei"]:
    try:
        from matplotlib import font_manager
        if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
            CJK = f
            break
    except Exception:
        pass
if CJK:
    plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
T = (lambda zh, en: zh) if CJK else (lambda zh, en: en)

EPOCH_RE = re.compile(
    r"(?:\[(?P<cfg>[^\]]+)\]\s+)?epoch=(?P<epoch>\d+)\s+"
    r"train_mae=(?P<train_mae>[0-9.eE+-]+)\s+val_mae=(?P<val_mae>[0-9.eE+-]+)\s+"
    r"(?:train_rmse=(?P<train_rmse>[0-9.eE+-]+)\s+val_rmse=(?P<val_rmse>[0-9.eE+-]+)\s+)?"
    r"(?:train_rel=(?P<train_rel>[0-9.eE+-]+)\s+val_rel=(?P<val_rel>[0-9.eE+-]+)\s+)?"
    r"lr=(?P<lr>[0-9.eE+-]+)\s+time=(?P<t>[0-9.eE+-]+)s"
)
TEST_OLD_RE = re.compile(r"test_mae=([0-9.eE+-]+),\s*test_rmse=([0-9.eE+-]+),\s*test_relative_error=([0-9.eE+-]+)")


def parse_log(path, cfg_fallback=None):
    text = open(path, encoding="utf-8", errors="ignore").read()
    rows = []
    for m in EPOCH_RE.finditer(text):
        d = {k: (float(v) if k not in ("epoch", "cfg") and v is not None else v) for k, v in m.groupdict().items()}
        d["epoch"] = int(m.group("epoch"))
        d["cfg"] = m.group("cfg") if m.group("cfg") and m.group("cfg") != "distance" else (cfg_fallback or os.path.basename(path))
        rows.append(d)
    test = None
    tm = TEST_OLD_RE.search(text)
    if tm:
        test = {"test_mae": float(tm.group(1)), "test_rmse": float(tm.group(2)), "test_rel": float(tm.group(3))}
    return rows, test


def plot_multi_curves(series, key, title, out, ylabel=None, top_legend=16, figsize=(12, 7)):
    plt.figure(figsize=figsize)
    for name, rows in series:
        xs = [r["epoch"] for r in rows]
        ys = [r[key] for r in rows if r.get(key) is not None]
        if len(ys) != len(xs) or not ys:
            continue
        plt.plot(xs, ys, marker="o", ms=2.5, lw=1.4, label=name)
    plt.xlabel(T("训练轮次 epoch", "epoch"))
    plt.ylabel(ylabel or key)
    plt.title(title)
    plt.grid(alpha=0.3)
    if len(series) > 1:
        plt.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.expanduser("~/Desktop/GNN_实验图表"))
    ap.add_argument("--workspace", default=os.path.expanduser("~/Documents/科研小组/gnn-euclidean-local"))
    ap.add_argument("--sandisk", default="/Volumes/闪迪硬盘/科研小组/GNN_地形实验")
    args = ap.parse_args()
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    made = []

    def save(fig_name):
        made.append(fig_name)

    # ================= EP_high =================
    ep_high_log = os.path.join(args.sandisk, "EP_high_实验日志", "pro6000_runner.log")
    ep_high_jsonl = os.path.join(args.sandisk, "EP_high_实验日志", "ep_high_d3c32_grid.jsonl")
    ep_high_base = os.path.join(args.sandisk, "EP_high_实验日志", "EP_high_geometry_baseline_seed42_n1000_summary.csv")

    if os.path.exists(ep_high_log):
        rows, _ = parse_log(ep_high_log)
        by_cfg = defaultdict(list)
        for r in rows:
            by_cfg[r["cfg"]].append(r)
        series = sorted(by_cfg.items(), key=lambda kv: -len(kv[1]))
        f = os.path.join(fig_dir, "01_EP_high_13配置_val_rel曲线.png")
        plot_multi_curves(series, "val_rel", T("EP_high：13 个配置的验证相对误差曲线", "EP_high: val relative error of 13 configs"), f)
        save("01_EP_high_13配置_val_rel曲线.png")
        f = os.path.join(fig_dir, "01b_EP_high_13配置_val_mae曲线.png")
        plot_multi_curves(series, "val_mae", T("EP_high：13 个配置的验证 MAE 曲线", "EP_high: val MAE of 13 configs"), f)
        save("01b_EP_high_13配置_val_mae曲线.png")
        # 最优配置详图
        best_key = None
        for k in by_cfg:
            if "100ep" in k:
                best_key = k
        if best_key is None and series:
            best_key = series[0][0]
        rows_b = by_cfg[best_key]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        xs = [r["epoch"] for r in rows_b]
        axes[0].plot(xs, [r["train_rel"] for r in rows_b], label=T("训练", "train"))
        axes[0].plot(xs, [r["val_rel"] for r in rows_b], label=T("验证", "val"))
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel(T("相对误差", "relative error"))
        axes[0].set_title(T("最优配置 100 轮：相对误差", "best config (100ep): relative error")); axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(xs, [r["train_mae"] for r in rows_b], label=T("训练", "train"))
        axes[1].plot(xs, [r["val_mae"] for r in rows_b], label=T("验证", "val"))
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE")
        axes[1].set_title(T("最优配置 100 轮：MAE", "best config (100ep): MAE")); axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "02_EP_high_最优100轮_详细曲线.png"), dpi=160); plt.close()
        save("02_EP_high_最优100轮_详细曲线.png")

    grid = []
    if os.path.exists(ep_high_jsonl):
        for line in open(ep_high_jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                grid.append(json.loads(line))
        grid = [g for g in grid if g.get("test_relative_error") is not None]
        grid.sort(key=lambda g: g["test_relative_error"])
        names = [g["name"] for g in grid]
        rels = [g["test_relative_error"] for g in grid]
        maes = [g["test_mae"] for g in grid]
        plt.figure(figsize=(11, 6))
        bars = plt.barh(range(len(names)), rels, color=["#2ecc71" if i == 0 else "#5dade2" for i in range(len(names))])
        plt.yticks(range(len(names)), names, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel(T("测试集相对误差 test_relative_error", "test_relative_error"))
        plt.title(T("EP_high：13 个配置的测试集相对误差排名", "EP_high: test relative error ranking (13 configs)"))
        for i, (b, v, m) in enumerate(zip(bars, rels, maes)):
            plt.text(b.get_width() + 0.0006, b.get_y() + b.get_height() / 2, f"{v:.4f} (MAE {m:.0f})", va="center", fontsize=8)
        plt.grid(axis="x", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "03_EP_high_配置排名_test_rel柱状.png"), dpi=160); plt.close()
        save("03_EP_high_配置排名_test_rel柱状.png")

        top = grid[0]
        groups = [("cross_leaf", T("跨叶", "cross-leaf")), ("long_dist", T("长距离", "long")),
                  ("mid_dist", T("中距离", "mid")), ("short_dist", T("短距离", "short")),
                  ("same_leaf", T("同叶", "same-leaf"))]
        vals = [top.get(f"group_{g}_relative_error", float("nan")) for g, _ in groups]
        cnts = [top.get(f"group_{g}_count", 0) for g, _ in groups]
        plt.figure(figsize=(8, 4.6))
        b = plt.bar([zh for _, zh in groups], vals, color="#5dade2")
        for bb, v, c in zip(b, vals, cnts):
            plt.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.0015, f"{v:.4f}\nn={c}", ha="center", fontsize=8)
        plt.ylabel(T("相对误差", "relative error"))
        plt.title(T("EP_high 最优配置：分组相对误差", "EP_high best config: grouped relative error"))
        plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "04_EP_high_最优分组误差.png"), dpi=160); plt.close()
        save("04_EP_high_最优分组误差.png")

    if os.path.exists(ep_high_base):
        base = {}
        with open(ep_high_base, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["scope"] == "overall":
                    base[row["method"]] = float(row["relative_error"])
        labels = [T("3D 直线基线", "euclidean_3d"), T("2D 直线基线", "euclidean_2d"), T("GNN 最优(100轮)", "GNN best(100ep)")]
        vals = [base.get("euclidean_3d", 0), base.get("euclidean_2d", 0), grid[0]["test_relative_error"] if grid else 0]
        plt.figure(figsize=(7.5, 4.6))
        b = plt.bar(labels, vals, color=["#e74c3c", "#f39c12", "#2ecc71"])
        for bb, v in zip(b, vals):
            plt.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.004, f"{v:.4f}", ha="center", fontsize=10)
        plt.ylabel(T("相对误差", "relative error"))
        plt.title(T("EP_high：无学习基线 vs GNN 模型", "EP_high: no-learning baselines vs GNN"))
        plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "05_EP_high_基线对比.png"), dpi=160); plt.close()
        save("05_EP_high_基线对比.png")

    # ================= EP_low =================
    ep_low_dir = os.path.join(args.sandisk, "EP_low_实验日志")
    csv_path = os.path.join(ep_low_dir, "EP_low_euclidean_residual_50000samples.csv")
    if os.path.exists(csv_path):
        eps, tr_rel, va_rel, tr_mae, va_mae = [], [], [], [], []
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("record_type") == "epoch" and row.get("epoch"):
                    try:
                        eps.append(int(row["epoch"]))
                        tr_rel.append(float(row["train_relative_error"]))
                        va_rel.append(float(row["val_relative_error"]))
                        tr_mae.append(float(row["train_mae"]))
                        va_mae.append(float(row["val_mae"]))
                    except Exception:
                        pass
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].plot(eps, tr_rel, label=T("训练", "train")); axes[0].plot(eps, va_rel, label=T("验证", "val"))
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel(T("相对误差", "relative error"))
        axes[0].set_title(T("EP_low 欧式残差版：相对误差", "EP_low euclidean_residual: relative error")); axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(eps, tr_mae, label=T("训练", "train")); axes[1].plot(eps, va_mae, label=T("验证", "val"))
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE")
        axes[1].set_title(T("EP_low 欧式残差版：MAE", "EP_low euclidean_residual: MAE")); axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "06_EP_low_欧式残差_训练曲线.png"), dpi=160); plt.close()
        save("06_EP_low_欧式残差_训练曲线.png")

    exp_logs = sorted(glob.glob(os.path.join(ep_low_dir, "EP_low_exp*.log")))
    exp_logs = [p for p in exp_logs if "(1)" not in p]
    series, finals = [], []
    for p in exp_logs:
        name = os.path.basename(p).replace("EP_low_", "").replace(".log", "")
        rows, test = parse_log(p, cfg_fallback=name)
        if not rows:
            continue
        for r in rows:
            r["cfg"] = name
        series.append((name, rows))
        if test:
            finals.append((name, test["test_rel"], test["test_mae"]))
    if series:
        has_rel = any(r.get("val_rel") is not None for _, rs in series for r in rs)
        key = "val_rel" if has_rel else "val_mae"
        f = os.path.join(fig_dir, "07_EP_low_历史实验_曲线.png")
        plot_multi_curves(series, key, T("EP_low 历史实验：验证曲线对比", "EP_low historical runs: validation curves"), f)
        save("07_EP_low_历史实验_曲线.png")
    if finals:
        finals.sort(key=lambda x: x[1])
        plt.figure(figsize=(11, 5))
        b = plt.barh([n for n, _, _ in finals], [v for _, v, _ in finals], color="#5dade2")
        plt.gca().invert_yaxis(); plt.xlabel(T("测试集相对误差", "test_relative_error"))
        plt.title(T("EP_low 历史实验：最终测试相对误差", "EP_low historical runs: final test relative error"))
        for bb, (_, v, m) in zip(b, finals):
            plt.text(bb.get_width() + 0.005, bb.get_y() + bb.get_height() / 2, f"{v:.4f} (MAE {m:.0f})", va="center", fontsize=8)
        plt.grid(axis="x", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "08_EP_low_历史实验_test_rel柱状.png"), dpi=160); plt.close()
        save("08_EP_low_历史实验_test_rel柱状.png")

    # ================= sample_terrain 调参 =================
    tune_logs = sorted(glob.glob(os.path.join(args.workspace, "logs", "euclidean_tune", "*.log")))
    series2, finals2 = [], []
    for p in tune_logs:
        name = os.path.basename(p).replace(".log", "")
        rows, test = parse_log(p, cfg_fallback=name)
        if not rows:
            continue
        for r in rows:
            r["cfg"] = name
        series2.append((name, rows))
        if test:
            finals2.append((name, test["test_rel"], test["test_mae"]))
    if finals2:
        finals2.sort(key=lambda x: x[1])
        top10 = finals2[:10]
        sel = {n for n, _, _ in top10}
        s_top = [s for s in series2 if s[0] in sel]
        f = os.path.join(fig_dir, "09_sample_terrain_调参Top10曲线.png")
        plot_multi_curves(s_top, "val_rel", T("sample_terrain 调参：Top-10 配置验证相对误差", "sample_terrain tuning: Top-10 val relative error"), f)
        save("09_sample_terrain_调参Top10曲线.png")
        plt.figure(figsize=(11, 5))
        b = plt.barh([n for n, _, _ in top10], [v for _, v, _ in top10], color="#af7ac5")
        plt.gca().invert_yaxis(); plt.xlabel(T("测试集相对误差", "test_relative_error"))
        plt.title(T("sample_terrain 调参：Top-10 最终测试相对误差", "sample_terrain tuning: Top-10 final test relative error"))
        for bb, (_, v, m) in zip(b, top10):
            plt.text(bb.get_width() + 0.0002, bb.get_y() + bb.get_height() / 2, f"{v:.5f}", va="center", fontsize=8)
        plt.grid(axis="x", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "10_sample_terrain_调参排名柱状.png"), dpi=160); plt.close()
        save("10_sample_terrain_调参排名柱状.png")

    # ================= 三数据集对比 =================
    labels, vals = [], []
    if finals2:
        labels.append(T("sample_terrain(793点)", "sample_terrain(793)")); vals.append(finals2[0][1])
    if os.path.exists(csv_path):
        best_low = None
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("record_type") == "test" and row.get("group") == "all":
                    best_low = float(row["test_relative_error"])
        if best_low:
            labels.append(T("EP_low(16.4万点)", "EP_low(164k)")); vals.append(best_low)
    if grid:
        labels.append(T("EP_high(139万点)", "EP_high(1.39M)")); vals.append(grid[0]["test_relative_error"])
    if labels:
        plt.figure(figsize=(7.5, 4.6))
        b = plt.bar(labels, vals, color=["#af7ac5", "#5dade2", "#2ecc71"][:len(labels)])
        for bb, v in zip(b, vals):
            plt.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + max(vals) * 0.02, f"{v:.4f}", ha="center", fontsize=10)
        plt.ylabel(T("测试集相对误差", "test_relative_error"))
        plt.title(T("三个数据集：欧式残差版最优结果对比", "three datasets: best euclidean_residual results"))
        plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "11_三数据集最优对比.png"), dpi=160); plt.close()
        save("11_三数据集最优对比.png")

    # ================= 汇总 CSV + 索引 =================
    summary_path = os.path.join(args.out_dir, "experiment_summary_all.csv")
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "run", "test_relative_error", "test_mae", "log_file"])
        for n, r, m in finals2:
            w.writerow(["sample_terrain", n, r, m, os.path.join(args.workspace, "logs", "euclidean_tune", n + ".log")])
        for n, r, m in finals:
            w.writerow(["EP_low", n, r, m, os.path.join(ep_low_dir, "EP_low_" + n + ".log")])
        for g in grid:
            w.writerow(["EP_high", g["name"], g["test_relative_error"], g["test_mae"], ep_high_log])
    idx = os.path.join(args.out_dir, "figures_index.md")
    with open(idx, "w", encoding="utf-8") as f:
        f.write("# 实验图表索引\n\n")
        for i, n in enumerate(made, 1):
            f.write(f"{i}. ![{n}](figures/{n})\n\n")
    print("figures:", len(made))
    for n in made:
        print("  -", n)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
