# 非学习基线：在导出的 test 点对上算常数/直线距离基线，与模型结果对比。
"""
baseline.py
===========

非学习（non-AI）基线，用于回答 README Q4：「GNN 到底比不学习的简单方法好多少？」

只在 **main.py 导出的 test 点对文件** 上评估（s,t,true_distance），
与模型评估的那批点对 **绝对一致**，不依赖种子复现。

三个朴素基线（弱 -> 强）：
  1. mean_constant —— 永远预测 test 真值均值（最优常数预测）。检验模型是否真的学到东西。
  2. euclidean_2d  —— 忽略高度的 (x,y) 平面直线距离。地形起伏下严重低估。
  3. euclidean_3d  —— s、t 两点 3D 直线距离。仍系统性低估真实图最短路。

模型若 rel_err 低于这些基线，即说明它在朴素方法之上确有价值。

用法（--test_pairs_file 必填）：
    python baseline.py --off_file generated/terrain_grid_20x20_400v.off \
        --test_pairs_file outputs/results/<run_name>_test_pairs.csv
"""

import argparse
import math
import os

from build_highway import load_off, build_pipeline_inputs_cached


def build_parser():
    p = argparse.ArgumentParser(description="距离回归的非学习基线（同口径，读 test 点对文件）")
    p.add_argument("--off_file", type=str, required=True, help="输入 .off 三角网格路径")
    p.add_argument("--file_folder", type=str, default="data", help="相对 --off_file 路径的基准目录（按项目根解析）")
    p.add_argument("--test_pairs_file", type=str, required=True,
                   help="main.py 导出的 test 点对 CSV(s,t,true_distance)，作为唯一的 test 集来源")
    p.add_argument("--out_file", type=str, default="", help="可选：将结果写入该路径")
    p.add_argument("--max_depth", type=int, default=3, help="highway baseline quadtree max depth")
    p.add_argument("--capacity", type=int, default=32, help="highway baseline quadtree capacity")
    p.add_argument("--uniform", action="store_true", help="use uniform quadtree for highway baseline")
    p.add_argument("--in_feat", type=int, default=64, help="feature dim used for highway cache key")
    p.add_argument("--highway_k", type=int, default=3, help="nearest entrances per endpoint for highway decomposition baseline; <=0 uses all")
    p.add_argument("--transit_k", type=int, default=0, help="same transit sparsification used by training; 0=full")
    p.add_argument("--cache_dir", type=str, default="outputs/cache", help="cache dir for derived highway context")
    return p


def load_test_pairs(path):
    """读取 main.py 导出的 test 点对 CSV: 表头 s,t,true_distance。返回 (s, t, true) 列表。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                s = int(parts[0]); t = int(parts[1]); d = float(parts[2])
            except ValueError:
                continue  # 跳过表头等非数据行
            pairs.append((s, t, d))
    return pairs


def compute_metrics(y_true, y_pred, eps=1e-9):
    """纯 Python 计算 mae / rmse / relative_error，与 model.compute_distance_metrics 同口径。"""
    n = len(y_true)
    abs_err = [abs(p - t) for p, t in zip(y_pred, y_true)]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n)
    rel = sum(ae / (t + eps) for ae, t in zip(abs_err, y_true)) / n
    return {"mae": mae, "rmse": rmse, "relative_error": rel}


def euclidean_3d(vertices, s, t):
    xs, ys, zs = vertices[s]
    xt, yt, zt = vertices[t]
    return math.sqrt((xs - xt) ** 2 + (ys - yt) ** 2 + (zs - zt) ** 2)


def euclidean_2d(vertices, s, t):
    """忽略高度 z，只用 (x,y) 平面直线距离。地形起伏下比 3D 直线更严重地低估。"""
    xs, ys, _ = vertices[s]
    xt, yt, _ = vertices[t]
    return math.sqrt((xs - xt) ** 2 + (ys - yt) ** 2)




def _nearest_indices(row, k):
    finite = [(float(d), idx) for idx, d in enumerate(row) if d != float("inf")]
    finite.sort(key=lambda x: x[0])
    if not finite:
        return []
    if k is not None and k > 0:
        finite = finite[:k]
    return finite


def highway_decomposition_preds(off_path, args, pairs, project_root):
    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(project_root, args.cache_dir)
    _, _, _, _, context = build_pipeline_inputs_cached(
        off_path=off_path,
        max_depth=args.max_depth,
        capacity=args.capacity,
        adaptive=not args.uniform,
        weighted=True,
        feature_dim=args.in_feat,
        device="cpu",
        cache_dir=cache_dir,
        transit_k=args.transit_k,
    )
    access = context["access_dist"]
    highway = context["highway_pair_dist"]
    preds = []
    for s, t, _ in pairs:
        s_entries = _nearest_indices(access[s], args.highway_k)
        t_entries = _nearest_indices(access[t], args.highway_k)
        best = float("inf")
        for ds, i in s_entries:
            for dt, j in t_entries:
                dh = float(highway[i][j])
                if dh != float("inf"):
                    best = min(best, ds + dh + dt)
        preds.append(best if best != float("inf") else 0.0)
    return preds

def main():
    args = build_parser().parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_folder = args.file_folder if os.path.isabs(args.file_folder) else os.path.join(project_root, args.file_folder)
    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(base_folder, args.off_file)

    vertices, _ = load_off(off_path)
    pairs = load_test_pairs(args.test_pairs_file)
    print(f"[baseline] off={off_path} |V|={len(vertices)}")
    print(f"[baseline] test pairs <- file: {args.test_pairs_file} ({len(pairs)} pairs)")

    if not pairs:
        print("[baseline] 测试集为空，退出。")
        return

    y_true = [d for (_, _, d) in pairs]
    mean_const = sum(y_true) / len(y_true)  # 最优常数预测（最强常数基线，对模型更保守）

    preds = {
        "mean_constant": [mean_const] * len(pairs),
        "euclidean_2d": [euclidean_2d(vertices, s, t) for (s, t, _) in pairs],
        "euclidean_3d": [euclidean_3d(vertices, s, t) for (s, t, _) in pairs],
        "highway_decomp": highway_decomposition_preds(off_path, args, pairs, project_root),
    }

    order = ["mean_constant", "euclidean_2d", "euclidean_3d", "highway_decomp"]
    results = {}
    for name in order:
        m = compute_metrics(y_true, preds[name])
        bias = sum(preds[name]) / len(pairs) - mean_const  # >0 平均高估, <0 平均低估
        results[name] = (m, bias)

    print("[baseline] === test-set non-learning baselines (弱 -> 强) ===")
    for name in order:
        m, bias = results[name]
        print(f"  {name:<14} mae={m['mae']:.6f}  rmse={m['rmse']:.6f}  "
              f"rel_err={m['relative_error']:.6f}  mean_bias={bias:+.4f}")
    print("[baseline] 模型的 test_relative_error 低于以上基线，即证明其在朴素方法之上的价值。")

    if args.out_file:
        out_file = args.out_file if os.path.isabs(args.out_file) else os.path.join(project_root, args.out_file)
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("baseline mae rmse relative_error mean_bias\n")
            for name in order:
                m, bias = results[name]
                f.write(f"{name} {m['mae']} {m['rmse']} {m['relative_error']} {bias}\n")
        print(f"[baseline] 结果已写入 {out_file}")


if __name__ == "__main__":
    main()
