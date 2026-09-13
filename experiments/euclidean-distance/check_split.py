# 8:1:1 划分验证：检查 train/val/test 点对的比例、对级泄漏、节点/边重叠。
"""
check_split.py
==============

验证 `main.py` 导出的 train/val/test 点对划分是否可信（对应实验计划的"第 5 步：8:1:1 节点/边重叠"）。

输入：同一次训练导出的三份 CSV（`outputs/results/<run>_{train,val,test}_pairs.csv`）。
只需给 `--run_prefix`（即 `<run>` 或其完整前缀路径），脚本自动拼三份文件名。

检查项：
  1. 划分比例：train:val:test 是否 ≈ 8:1:1。
  2. **对级泄漏（关键）**：同一个 (s,t) 是否跨集出现——必须为 0，否则测试指标虚高。
  3. 节点重叠：各集合出现的端点集合、以及 test 端点中有多少也出现在 train。
  4. 边（无序对）重叠：把 (s,t) 视作一条查询"边"，统计跨集的重复边数（= 对级泄漏，另一角度）。

> 说明：本项目是**单图直推式**——train/val/test 共享同一张图，所以**节点重叠高是正常且预期的**，
> 不算泄漏；真正的泄漏判据是**同一对 (s,t) 跨集出现**（第 2 项须为 0）。

用法：
  python check_split.py --run_prefix outputs/results/EP_low_distance_2026-07-09_10-49-01
  # 或分别指定：
  python check_split.py --train a_train_pairs.csv --val a_val_pairs.csv --test a_test_pairs.csv
"""

import argparse
import os


def load_pairs(path):
    """读取 <split>_pairs.csv(s,t,true_distance) → 归一化无序对集合 + 端点节点集合。"""
    pairs = set()
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                s = int(parts[0]); t = int(parts[1])
            except (ValueError, IndexError):
                continue  # 表头
            key = (s, t) if s <= t else (t, s)  # 无序化
            pairs.add(key)
            nodes.add(s); nodes.add(t)
    return pairs, nodes


def main():
    p = argparse.ArgumentParser(description="8:1:1 划分验证（比例/泄漏/节点边重叠）")
    p.add_argument("--run_prefix", default="", help="<run> 前缀，自动拼 _{train,val,test}_pairs.csv")
    p.add_argument("--train", default="", help="train pairs CSV（覆盖 run_prefix）")
    p.add_argument("--val", default="", help="val pairs CSV")
    p.add_argument("--test", default="", help="test pairs CSV")
    args = p.parse_args()

    if args.run_prefix:
        train_p = args.train or f"{args.run_prefix}_train_pairs.csv"
        val_p = args.val or f"{args.run_prefix}_val_pairs.csv"
        test_p = args.test or f"{args.run_prefix}_test_pairs.csv"
    else:
        train_p, val_p, test_p = args.train, args.val, args.test

    for name, path in [("train", train_p), ("val", val_p), ("test", test_p)]:
        if not path or not os.path.exists(path):
            print(f"[check] 缺少 {name} 文件: {path or '(未指定)'}")
            print("[check] 提示：需 main.py 导出三份 *_pairs.csv（train/val/test 都要）。退出。")
            return

    train_pairs, train_nodes = load_pairs(train_p)
    val_pairs, val_nodes = load_pairs(val_p)
    test_pairs, test_nodes = load_pairs(test_p)

    n_tr, n_va, n_te = len(train_pairs), len(val_pairs), len(test_pairs)
    total = n_tr + n_va + n_te
    print(f"[check] 划分大小: train={n_tr}, val={n_va}, test={n_te}, total={total}")
    if total:
        print(f"[check] 划分比例: {n_tr/total:.3f} : {n_va/total:.3f} : {n_te/total:.3f} "
              f"(目标 0.800 : 0.100 : 0.100)")

    # —— 对级泄漏（关键，必须全为 0）——
    tr_va = train_pairs & val_pairs
    tr_te = train_pairs & test_pairs
    va_te = val_pairs & test_pairs
    print("[check] === 对级泄漏（同一 (s,t) 跨集出现，必须为 0）===")
    print(f"[check]   train∩val = {len(tr_va)}")
    print(f"[check]   train∩test = {len(tr_te)}")
    print(f"[check]   val∩test   = {len(va_te)}")
    leak = len(tr_va) + len(tr_te) + len(va_te)
    if leak == 0:
        print("[check]   ✅ 无对级泄漏（train/val/test 的查询对互不重复）")
    else:
        print(f"[check]   ❌ 检测到 {leak} 处对级泄漏！测试指标不可信，需排查采样/切分。")

    # 唯一性自检（同一集合内不应有重复对）
    dup_note = []
    for name, path, pairs in [("train", train_p, train_pairs), ("val", val_p, val_pairs), ("test", test_p, test_pairs)]:
        raw = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                try:
                    int(parts[0]); int(parts[1]); raw += 1
                except (ValueError, IndexError):
                    pass
        if raw != len(pairs):
            dup_note.append(f"{name}(原始{raw}行 vs 唯一{len(pairs)}对)")
    if dup_note:
        print(f"[check]   ⚠️ 集合内有重复对: {', '.join(dup_note)}")

    # —— 节点/边重叠（直推式：高重叠是预期，仅作背景信息）——
    print("[check] === 节点重叠（单图直推式下高重叠属正常，非泄漏）===")
    print(f"[check]   唯一端点数: train={len(train_nodes)}, val={len(val_nodes)}, test={len(test_nodes)}")
    if test_nodes:
        inter = len(test_nodes & train_nodes)
        print(f"[check]   test 端点中也出现在 train 的比例 = {inter}/{len(test_nodes)} "
              f"= {inter/len(test_nodes):.2%}")
    print("[check] 结论：泄漏判据看'对级泄漏'(须为0)；节点重叠高是同图直推式的正常现象。")


if __name__ == "__main__":
    main()
