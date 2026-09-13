import itertools
import json
import os
import re
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs", "euclidean_grid")
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)


BASE_CMD = [
    sys.executable,
    "-u",
    "main.py",
    "--off_file",
    "sample_terrain.off",
    "--file_folder",
    "data",
    "--max_depth",
    "3",
    "--capacity",
    "32",
    "--in_feat",
    "16",
    "--dropout_ratio",
    "0.1",
    "--learning_rate",
    "0.001",
    "--lr_scheduler",
    "plateau",
    "--lr_patience",
    "3",
    "--lr_factor",
    "0.5",
    "--min_lr",
    "1e-6",
    "--num_epoch",
    "12",
    "--batch_size",
    "16",
    "--train_percent",
    "0.8",
    "--early_stop_patience",
    "6",
    "--distance_samples",
    "600",
    "--selection_metric",
    "relative_error",
    "--inner_mode",
    "partition",
    "--prediction_mode",
    "euclidean_residual",
    "--device",
    "cpu",
    "--seed",
    "42",
    "--cache_dir",
    "outputs/cache/euclidean_grid",
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
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            result[key] = int(value) if key == "best_epoch" else float(value)
    for group in ["same_leaf", "cross_leaf", "short_dist", "mid_dist", "long_dist"]:
        m = re.findall(
            rf"\[distance\]\[group\] {group} count=(\d+) .*?relative_error=([0-9.]+)",
            text,
        )
        if m:
            result[f"{group}_count"] = int(m[-1][0])
            result[f"{group}_relative_error"] = float(m[-1][1])
    result["early_stopped"] = "early stop" in text
    result["cache_hit"] = "[cache] 命中" in text
    return result


def run_one(config, idx, total):
    size_name, hidden_dim, out_dim = config["size"]
    log_name = (
        f"{idx:03d}_{config['sample_strategy']}_{config['loss_type']}_"
        f"k{config['highway_k']}_{size_name}_pure_euclidean.log"
    )
    log_path = os.path.join(LOG_DIR, log_name)
    cmd = BASE_CMD + [
        "--sample_strategy",
        config["sample_strategy"],
        "--loss_type",
        config["loss_type"],
        "--highway_k",
        str(config["highway_k"]),
        "--hidden_dim",
        str(hidden_dim),
        "--out_dim",
        str(out_dim),
        "--disable_highway_distance_feature",
    ]
    t0 = time.time()
    print(f"[{idx}/{total}] {log_name}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    text = open(log_path, encoding="utf-8", errors="ignore").read()
    row = {
        "idx": idx,
        "log": log_name,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - t0, 3),
        "sample_strategy": config["sample_strategy"],
        "loss_type": config["loss_type"],
        "highway_k": config["highway_k"],
        "size": size_name,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "highway_distance_feature": False,
    }
    row.update(parse_log(text))
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main():
    if os.path.exists(SUMMARY_PATH):
        os.remove(SUMMARY_PATH)

    grid = [
        {
            "sample_strategy": sample_strategy,
            "loss_type": loss_type,
            "highway_k": highway_k,
            "size": size,
        }
        for sample_strategy, loss_type, highway_k, size in itertools.product(
            ["random", "oracle_mix", "distance_gap"],
            ["relative", "log_l1", "huber"],
            [1, 3, 5],
            [("small", 32, 16), ("medium", 64, 32)],
        )
    ]
    rows = []
    for idx, config in enumerate(grid, 1):
        rows.append(run_one(config, idx, len(grid)))
    rows.sort(key=lambda r: r.get("test_relative_error", float("inf")))
    print("\nTop 10 by test_relative_error")
    for row in rows[:10]:
        print(
            f"{row['test_relative_error']:.6f} | {row['sample_strategy']} "
            f"{row['loss_type']} k={row['highway_k']} {row['size']} "
            f"best={row.get('best_val_relative_error')} log={row['log']}"
        )


if __name__ == "__main__":
    main()
