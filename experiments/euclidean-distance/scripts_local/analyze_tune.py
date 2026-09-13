# -*- coding: utf-8 -*-
"""分析 run_euclidean_tune 的 summary.jsonl：Top 配置、维度聚合、feat on/off 对比。"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "logs", "euclidean_tune", "summary.jsonl")


def load():
    rows = []
    with open(SUMMARY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if r.get("test_relative_error") is not None]


def main():
    rows = load()
    print(f"有效运行: {len(rows)}")
    rows.sort(key=lambda r: r["test_relative_error"])
    print("\n=== Top 20 ===")
    for r in rows[:20]:
        print(
            f"{r['test_relative_error']:.5f} | {r['tag']:<34} | feat={int(r['use_highway_feature'])} "
            f"mode={r['prediction_mode']:<17} best_ep={r.get('best_epoch'):>2} mae={r.get('test_mae', 0):7.2f}"
        )
    print("\n=== 按维度聚合（mean/min）===")
    for dim in ["loss_type", "sample_strategy", "highway_k", "size", "use_highway_feature"]:
        agg = defaultdict(list)
        for r in rows:
            agg[str(r[dim])].append(r["test_relative_error"])
        parts = []
        for k, v in sorted(agg.items()):
            parts.append(f"{k}: n={len(v):2d} mean={sum(v)/len(v):.5f} min={min(v):.5f}")
        print(f"{dim}: " + " | ".join(parts))
    # 仅 euclidean_residual + partition + d3c32 + n600 + 12ep 的严格可比子集
    print("\n=== 严格可比子集（euclidean_residual/partition/d3c32/n600/12ep）===")
    strict = [r for r in rows if r["prediction_mode"] == "euclidean_residual"
              and r["inner_mode"] == "partition" and r["max_depth"] == 3
              and r["capacity"] == 32 and r["distance_samples"] == 600]
    print(f"子集大小: {len(strict)}")
    # feat ON vs OFF 同配置配对对比
    print("\n=== feat ON vs OFF 配对对比（同 loss/sample/k/size）===")
    by_key = defaultdict(dict)
    for r in strict:
        key = (r["loss_type"], r["sample_strategy"], r["highway_k"], r["size"])
        by_key[key][bool(r["use_highway_feature"])] = r["test_relative_error"]
    n_on_win = n_off_win = n_tie = 0
    for key, d in sorted(by_key.items()):
        if True in d and False in d:
            on, off = d[True], d[False]
            mark = "ON更好" if on < off else ("OFF更好" if off < on else "=")
            if on < off: n_on_win += 1
            elif off < on: n_off_win += 1
            else: n_tie += 1
            print(f"  {key[0]:>8} {key[1]:>12} k{key[2]} {key[3]:<6} ON={on:.5f} OFF={off:.5f}  {mark}")
    print(f"  配对汇总: ON更好={n_on_win}, OFF更好={n_off_win}, 平={n_tie}")
    # 推荐：最好 5 个配置（考虑稳定性：best_epoch 不过早、无 early stop 过急）
    print("\n=== 推荐配置（Top 5，best_epoch>=4 且 non-early-stopped 优先）===")
    stable = sorted(rows, key=lambda r: (r["test_relative_error"], -(r.get("best_epoch") or 0)))
    for r in stable[:5]:
        print(
            f"{r['test_relative_error']:.5f} | {r['tag']:<34} | mode={r['prediction_mode']} "
            f"loss={r['loss_type']} sample={r['sample_strategy']} k={r['highway_k']} "
            f"size={r['size']} feat={int(r['use_highway_feature'])} best_ep={r.get('best_epoch')} "
            f"early={r.get('early_stopped')}"
        )


if __name__ == "__main__":
    main()
