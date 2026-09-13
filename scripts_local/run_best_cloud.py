# -*- coding: utf-8 -*-
"""
run_best_cloud.py — 云平台(AutoDL)完整训练脚本
最终配置由本地全参数网格实验（run_euclidean_tune.py + verify_24ep.py）筛出。

结论（DSH 2026-08-28）：
  教授要求：不依赖高速网络（已有工作），以 3D 欧氏距离为基线 + GNN 修正。
  配置：euclidean_residual + --disable_highway_distance_feature（关闭高速）。
  本地 sample_terrain.off(793v) 上 test_relative_error ≈ 0.01675（12 & 24 epoch 一致，best_epoch=9）。

  最优超参：
    loss_type = huber        sample_strategy = random
    highway_k = 3            hidden_dim = 32, out_dim = 16（small）
    inner_mode = partition   max_depth = 3, capacity = 32（自适应四叉树）
    selection_metric = relative_error

用法（在 AutoDL 上）：
  python run_best_cloud.py --off_file dataset/EP_low/EP_low.off --num_epoch 150
环境：CUDA GPU、torch、torch-geometric、scipy、numpy
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 最终推荐配置（由本地网格实验确定）----
BEST = {
    "prediction_mode": "euclidean_residual",   # 教授建议：欧氏距离基线 + GNN 修正
    "use_highway_feature": False,              # 关闭高速（不依赖已有工作）
    "sample_strategy": "random",
    "loss_type": "huber",
    "highway_k": 3,
    "hidden_dim": 32,
    "out_dim": 16,
    "inner_mode": "partition",
    "max_depth": 3,
    "capacity": 32,
    "transit_k": 0,
    "distance_samples": 50000,                 # EP_low 大图样本上限
}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--off_file", default="dataset/EP_low/EP_low.off", help="云平台地形 .off 路径")
    p.add_argument("--file_folder", default=".", help="off_file 相对基准目录")
    p.add_argument("--num_epoch", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_feat", type=int, default=64, help="大图特征维度（本地小图用16，云端EP_low建议64）")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--out_dim", type=int, default=16)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p


def main():
    args = build_parser().parse_args()
    cmd = [
        sys.executable, "-u", "main.py",
        "--off_file", args.off_file,
        "--file_folder", args.file_folder,
        "--max_depth", str(BEST["max_depth"]),
        "--capacity", str(BEST["capacity"]),
        "--in_feat", str(args.in_feat),
        "--hidden_dim", str(args.hidden_dim),
        "--out_dim", str(args.out_dim),
        "--dropout_ratio", "0.1",
        "--learning_rate", "0.001",
        "--lr_scheduler", "plateau",
        "--lr_patience", str(args.lr_patience),
        "--lr_factor", "0.5",
        "--min_lr", "1e-6",
        "--num_epoch", str(args.num_epoch),
        "--batch_size", str(args.batch_size),
        "--train_percent", "0.8",
        "--early_stop_patience", str(args.early_stop_patience),
        "--distance_samples", str(BEST["distance_samples"]),
        "--sample_strategy", BEST["sample_strategy"],
        "--highway_k", str(BEST["highway_k"]),
        "--inner_mode", BEST["inner_mode"],
        "--loss_type", BEST["loss_type"],
        "--selection_metric", "relative_error",
        "--prediction_mode", BEST["prediction_mode"],
        "--device", args.device,
        "--seed", str(args.seed),
        "--cache_dir", "outputs/cache/cloud_best",
        "--disable_highway_distance_feature",   # 关闭高速
    ]
    if BEST["transit_k"]:
        cmd += ["--transit_k", str(BEST["transit_k"])]

    log_path = os.path.join(ROOT, "logs", f"cloud_best_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(">>> 运行命令：")
    print(" ".join(cmd))
    print(f">>> 日志写入：{log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f">>> 完成，returncode={proc.returncode}")


if __name__ == "__main__":
    main()
