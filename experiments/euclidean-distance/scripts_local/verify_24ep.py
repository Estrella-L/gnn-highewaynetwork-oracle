# -*- coding: utf-8 -*-
"""验证 feat=OFF + euclidean_residual 候选配置在更长训练(24 epoch)下的表现。"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs", "euclidean_verify")
SUMMARY = os.path.join(LOG_DIR, "summary.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

BASE = [
    sys.executable, "-u", "main.py", "--off_file", "sample_terrain.off", "--file_folder", "data",
    "--in_feat", "16", "--dropout_ratio", "0.1", "--learning_rate", "0.001",
    "--lr_scheduler", "plateau", "--lr_patience", "4", "--lr_factor", "0.5", "--min_lr", "1e-6",
    "--num_epoch", "24", "--batch_size", "16", "--train_percent", "0.8",
    "--early_stop_patience", "8", "--distance_samples", "600",
    "--selection_metric", "relative_error", "--inner_mode", "partition",
    "--prediction_mode", "euclidean_residual", "--device", "cpu", "--seed", "42",
    "--disable_highway_distance_feature", "--cache_dir", "outputs/cache/euclidean_verify",
]

# 候选：feat=OFF + euclidean_residual + random + partition + d3c32 + n600
CANDS = [
    ("v_huber_k3_small",  "huber", 3, 32, 16),
    ("v_huber_k3_medium", "huber", 3, 64, 32),
    ("v_relative_k3_small","relative", 3, 32, 16),
    ("v_relative_k3_medium","relative", 3, 64, 32),
    ("v_logl1_k3_small",  "log_l1", 3, 32, 16),
    ("v_logl1_k3_medium", "log_l1", 3, 64, 32),
    ("v_huber_k5_small",  "huber", 5, 32, 16),
]

if os.path.exists(SUMMARY): os.remove(SUMMARY)
rows = []
for i, (tag, loss, k, h, o) in enumerate(CANDS, 1):
    log_path = os.path.join(LOG_DIR, f"{i:02d}_{tag}.log")
    cmd = BASE + ["--sample_strategy", "random", "--loss_type", loss,
                  "--highway_k", str(k), "--hidden_dim", str(h), "--out_dim", str(o)]
    t0 = time.time()
    print(f"[{i}/{len(CANDS)}] {tag}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    text = open(log_path, encoding="utf-8", errors="ignore").read()
    rel = re.findall(r"test_relative_error=([0-9.]+)", text)
    mae = re.findall(r"test_mae=([0-9.]+)", text)
    be = re.findall(r"best_epoch=([0-9]+)", text)
    row = {"tag": tag, "loss": loss, "k": k, "size": "small" if o==16 else "medium",
           "rel": float(rel[-1]) if rel else None, "mae": float(mae[-1]) if mae else None,
           "best_epoch": int(be[-1]) if be else None, "early": "early stop" in text,
           "elapsed": round(time.time()-t0, 1)}
    rows.append(row)
    print(f"  rel={row['rel']:.5f} mae={row['mae']:.2f} be={row['best_epoch']} early={row['early']}", flush=True)
    with open(SUMMARY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

rows.sort(key=lambda r: r["rel"])
print("\n=== 24 epoch 验证结果（按 rel 排序）===")
for r in rows:
    print(f"{r['rel']:.5f} | {r['tag']:<22} loss={r['loss']:<8} k{r['k']} {r['size']:<6} | be={r['best_epoch']} early={r['early']}")
