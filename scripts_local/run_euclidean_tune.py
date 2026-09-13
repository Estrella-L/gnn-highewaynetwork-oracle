# -*- coding: utf-8 -*-
"""
run_euclidean_tune.py — DSH 本地全参数网格实验（euclidean_residual 主线，教授建议：欧氏距离基线 + GNN 修正）
覆盖维度：
  A. highway 特征开关完整对照（feat=ON × loss × sample × k × size）
  B. 补齐既有 pure_euclidean 网格缺失的 10 个（distance_gap × log_l1/huber × k3/k5）
  C. prediction_mode=direct 对照（无先验纯 GNN）
  D. 结构消融：inner_mode=ego / depth2-cap64 / depth4-cap16 / transit_k=8 / transit_k=16
  E. 数据规模：samples=1200 / 2400
  F. 更大模型：hidden=128 / out=64
所有配置 seed=42、12 epoch（快速筛选），输出 logs/euclidean_tune/。
"""
import itertools
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs", "euclidean_tune")
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

BASE_CMD = [
    sys.executable, "-u", "main.py",
    "--off_file", "sample_terrain.off",
    "--file_folder", "data",
    "--max_depth", "3",
    "--capacity", "32",
    "--in_feat", "16",
    "--dropout_ratio", "0.1",
    "--learning_rate", "0.001",
    "--lr_scheduler", "plateau",
    "--lr_patience", "3",
    "--lr_factor", "0.5",
    "--min_lr", "1e-6",
    "--num_epoch", "12",
    "--batch_size", "16",
    "--train_percent", "0.8",
    "--early_stop_patience", "6",
    "--distance_samples", "600",
    "--selection_metric", "relative_error",
    "--inner_mode", "partition",
    "--prediction_mode", "euclidean_residual",
    "--device", "cpu",
    "--seed", "42",
    "--cache_dir", "outputs/cache/euclidean_tune",
]


def parse_log(text):
    result = {}
    patterns = {
        "best_epoch": r"best_epoch=(\d+)",
        "best_val_relative_error": r"best_val_relative_error=([0-9.]+)",
        "test_mae": r"test_mae=([0-9.]+)",
        "test_rmse": r"test_rmse=([0-9.]+)",
        "test_relative_error": r"test_relative_error=([0-9.]+)",
        "train_seconds": r"train=([0-9.]+)s",
    }
    for key, pattern in patterns.items():
        m = re.findall(pattern, text)
        if m:
            v = m[-1]
            result[key] = int(v) if key == "best_epoch" else float(v)
    for group in ["same_leaf", "cross_leaf", "short_dist", "mid_dist", "long_dist"]:
        m = re.findall(rf"\[distance\]\[group\] {group} count=(\d+) .*?relative_error=([0-9.]+)", text)
        if m:
            result[f"{group}_count"] = int(m[-1][0])
            result[f"{group}_relative_error"] = float(m[-1][1])
    result["early_stopped"] = "early stop" in text
    result["cache_hit"] = "[cache] 命中" in text
    return result


def run_one(cfg, idx, total):
    log_name = (
        f"{idx:03d}_{cfg['tag']}.log"
    )
    log_path = os.path.join(LOG_DIR, log_name)
    cmd = BASE_CMD + [
        "--sample_strategy", cfg["sample_strategy"],
        "--loss_type", cfg["loss_type"],
        "--highway_k", str(cfg["highway_k"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--out_dim", str(cfg["out_dim"]),
    ]
    if cfg.get("num_epoch"):
        cmd += ["--num_epoch", str(cfg["num_epoch"])]
    if cfg.get("distance_samples"):
        cmd += ["--distance_samples", str(cfg["distance_samples"])]
    if cfg.get("inner_mode"):
        cmd += ["--inner_mode", cfg["inner_mode"]]
    if cfg.get("prediction_mode"):
        cmd += ["--prediction_mode", cfg["prediction_mode"]]
    if cfg.get("max_depth"):
        cmd += ["--max_depth", str(cfg["max_depth"])]
    if cfg.get("capacity"):
        cmd += ["--capacity", str(cfg["capacity"])]
    if cfg.get("transit_k"):
        cmd += ["--transit_k", str(cfg["transit_k"])]
    if not cfg.get("use_highway_feature"):
        cmd += ["--disable_highway_distance_feature"]
    t0 = time.time()
    print(f"[{idx}/{total}] {log_name}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    text = open(log_path, encoding="utf-8", errors="ignore").read()
    row = {
        "idx": idx, "log": log_name, "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - t0, 3),
        "tag": cfg["tag"],
        "prediction_mode": cfg.get("prediction_mode", "euclidean_residual"),
        "sample_strategy": cfg["sample_strategy"],
        "loss_type": cfg["loss_type"],
        "highway_k": cfg["highway_k"],
        "size": cfg["size"],
        "hidden_dim": cfg["hidden_dim"], "out_dim": cfg["out_dim"],
        "use_highway_feature": bool(cfg.get("use_highway_feature")),
        "inner_mode": cfg.get("inner_mode", "partition"),
        "max_depth": cfg.get("max_depth", 3), "capacity": cfg.get("capacity", 32),
        "transit_k": cfg.get("transit_k", 0),
        "distance_samples": cfg.get("distance_samples", 600),
        "num_epoch": cfg.get("num_epoch", 12),
    }
    row.update(parse_log(text))
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def build_grid():
    grid = []
    idx = 0
    def add(tag, **kw):
        nonlocal idx
        idx += 1
        kw["tag"] = tag
        kw.setdefault("size", "small" if (kw["hidden_dim"], kw["out_dim"]) == (32, 16) else "medium")
        grid.append(kw)

    sizes = [("small", 32, 16), ("medium", 64, 32)]
    # A. highway 特征开启的完整网格（euclidean_residual, feat=ON）
    for loss, sample, k, (sname, h, o) in itertools.product(
            ["l1", "log_l1", "relative", "huber"],
            ["random", "oracle_mix", "distance_gap"],
            [1, 3, 5], sizes):
        add(f"hw_{sample}_{loss}_k{k}_{sname}",
            sample_strategy=sample, loss_type=loss, highway_k=k,
            hidden_dim=h, out_dim=o, use_highway_feature=True)
    # B. 补齐 pure_euclidean 缺失：distance_gap × log_l1/huber × k3/k5（feat=OFF）
    for loss, k, (sname, h, o) in itertools.product(
            ["log_l1", "huber"], [3, 5], sizes):
        add(f"nohw_gap_{loss}_k{k}_{sname}",
            sample_strategy="distance_gap", loss_type=loss, highway_k=k,
            hidden_dim=h, out_dim=o, use_highway_feature=False)
    # C. direct 对照（无先验）
    for loss in ["relative", "log_l1", "huber"]:
        add(f"direct_{loss}_k3_small",
            sample_strategy="random", loss_type=loss, highway_k=3,
            hidden_dim=32, out_dim=16, use_highway_feature=False,
            prediction_mode="direct")
    # D. 结构消融（random/relative/k3/small/feat=OFF 基线）
    add("ego_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False, inner_mode="ego")
    add("d2c64_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False,
        max_depth=2, capacity=64)
    add("d4c16_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False,
        max_depth=4, capacity=16)
    add("tk8_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False, transit_k=8)
    add("tk16_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False, transit_k=16)
    # E. 数据规模（random/relative/k3/small/feat=OFF）
    add("n1200_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False, distance_samples=1200)
    add("n2400_relative_k3_small", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=32, out_dim=16, use_highway_feature=False, distance_samples=2400)
    # F. 更大模型（random/relative/k3/feat=OFF）
    add("big128_relative_k3", sample_strategy="random", loss_type="relative",
        highway_k=3, hidden_dim=128, out_dim=64, use_highway_feature=False)
    return grid


def main():
    if os.path.exists(SUMMARY_PATH):
        os.remove(SUMMARY_PATH)
    grid = build_grid()
    print(f"total configs: {len(grid)}", flush=True)
    rows = []
    for i, cfg in enumerate(grid, 1):
        rows.append(run_one(cfg, i, len(grid)))
    valid = [r for r in rows if r.get("test_relative_error") is not None]
    valid.sort(key=lambda r: r["test_relative_error"])
    print("\n=== Top 15 by test_relative_error ===")
    for r in valid[:15]:
        print(
            f"{r['test_relative_error']:.5f} | {r['tag']:<34} | feat={int(r['use_highway_feature'])} "
            f"best_ep={r.get('best_epoch')} mae={r.get('test_mae', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
