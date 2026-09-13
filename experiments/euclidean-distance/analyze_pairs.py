# 分析点对的跨分区/同分区比例：用缓存的分区表 + 采样的 test 点对，给出精确证据。
"""
analyze_pairs.py
================

回答"采样点对里有多少是跨分区/同分区"——用**实测**代替估算。

输入：
  --partition_csv : build_pipeline_inputs_cached 导出的 <key>_partition.csv（含每点 leaf_id）
  --pairs_csv     : （可选）main.py 导出的 <run>_test_pairs.csv（s,t,true_distance）

输出：
  1. 叶子大小分布（个数/最小/最大/均值），以及 Σpᵢ²；
  2. 全图**所有**唯一对中的同分区比例（精确）：Σ C(nᵢ,2) / C(N,2)；
  3. 若给了 pairs_csv：实际采样点对中同分区/跨分区的**实测**比例。

用法：
  python analyze_pairs.py \
    --partition_csv outputs/cache/EP_low_d2_c512_ada_f64_16f4cacf_partition.csv \
    --pairs_csv outputs/results/EP_low_distance_2026-07-09_10-49-01_test_pairs.csv
"""

import argparse
from collections import Counter


def load_leaf_of(partition_csv):
    """读取 partition CSV(node,leaf_id,is_highway,x,y) → {node: leaf_id}。"""
    leaf_of = {}
    with open(partition_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                node = int(parts[0]); leaf = int(parts[1])
            except (ValueError, IndexError):
                continue  # 跳过表头
            leaf_of[node] = leaf
    return leaf_of


def main():
    p = argparse.ArgumentParser(description="分析点对的跨分区/同分区比例（实测）")
    p.add_argument("--partition_csv", required=True, help="<key>_partition.csv 路径")
    p.add_argument("--pairs_csv", default="", help="（可选）<run>_test_pairs.csv 路径")
    args = p.parse_args()

    leaf_of = load_leaf_of(args.partition_csv)
    N = len(leaf_of)
    sizes = Counter(leaf_of.values())
    leaf_sizes = list(sizes.values())
    n_leaves = len(leaf_sizes)

    total_pairs = N * (N - 1) // 2
    intra_pairs = sum(s * (s - 1) // 2 for s in leaf_sizes)
    p2 = sum((s / N) ** 2 for s in leaf_sizes)  # Σpᵢ²

    print(f"[analyze] 节点数 N={N}, 叶子数={n_leaves}")
    print(f"[analyze] 叶子大小: min={min(leaf_sizes)}, max={max(leaf_sizes)}, "
          f"mean={N / n_leaves:.1f}")
    print(f"[analyze] Σpᵢ² = {p2:.4f}  (叶子等大时应为 1/{n_leaves} = {1/n_leaves:.4f})")
    print(f"[analyze] 全图所有对中同分区比例(精确) = intra/total = "
          f"{intra_pairs}/{total_pairs} = {intra_pairs / total_pairs:.4%}")
    print(f"[analyze] 全图所有对中跨分区比例(精确) = {1 - intra_pairs / total_pairs:.4%}")

    if args.pairs_csv:
        n_pair = n_intra = n_missing = 0
        with open(args.pairs_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    s = int(parts[0]); t = int(parts[1])
                except (ValueError, IndexError):
                    continue  # 表头
                if s not in leaf_of or t not in leaf_of:
                    n_missing += 1
                    continue
                n_pair += 1
                if leaf_of[s] == leaf_of[t]:
                    n_intra += 1
        if n_pair:
            print(f"[analyze] === 采样点对实测（{args.pairs_csv}）===")
            print(f"[analyze] 有效对数={n_pair}（缺失leaf的={n_missing}）")
            print(f"[analyze] 采样中同分区(intra) = {n_intra}/{n_pair} = {n_intra / n_pair:.4%}")
            print(f"[analyze] 采样中跨分区(cross) = {n_pair - n_intra}/{n_pair} = "
                  f"{1 - n_intra / n_pair:.4%}")


if __name__ == "__main__":
    main()
