# 生成规则网格地形 .off（带真实起伏，适配本项目的距离分解算法）。
"""
generate_terrain.py
===================

生成规则网格地形 `.off`，专门适配本项目的「本地 + 高速 + 本地」距离分解算法。

设计目标（为什么这样生成）：
  1. 真实起伏 z：边权是相邻顶点的 3D 欧氏距离，翻山要 sqrt(dx^2+dz^2)，
     因此「图最短路 != 平面直线」——让直线距离基线失效，问题变得非平凡。
  2. 山脊 / 障碍造成绕行：最短路被迫绕到低矮垭口，这正是高速分解能发挥作用、
     且「强行走高速入口」会系统性高估的地方，给可学习纠偏留出空间。
  3. 规则网格 + 空间局部性：四叉树分区天然合理，高速骨架结构清晰。

输出为标准 OFF 三角网格，可直接被 build_highway.load_off() 读取：
    OFF
    nV nF 0
    x y z          # nV 行
    3 a b c        # nF 行（每个网格单元切成两个三角形）

用法示例：
    # 一次生成 100 / 400 / 1600 三种规模（与云端命名一致）
    python generate_terrain.py --grid 10 20 40

    # 平地对照组（消融：z=0，此时图最短路 ~ 平面直线，直线基线会很强）
    python generate_terrain.py --grid 20 --mode flat --out_suffix _flat
"""

import argparse
import math
import os

try:
    import numpy as np
except ImportError as e:  # 仅依赖 numpy
    raise SystemExit("generate_terrain.py 需要 numpy，请先 pip install numpy") from e


def height_field(n, spacing, mode, relief, seed):
    """
    生成 n x n 网格的高度场 Z。

    Args:
        n (int): 每边顶点数（总顶点 = n*n）。
        spacing (float): 相邻网格点的水平间距。
        mode (str): flat / smooth / mountains / ridges / mixed。
        relief (float): 起伏幅度，按网格水平范围的比例给定（0.4 = 山高约为地图边长的 40%）。
        seed (int): 随机种子，保证可复现。

    Returns:
        (X, Y, Z): 各为 [n, n] 的 numpy 数组。
    """
    extent = max(1e-9, (n - 1) * spacing)
    amp = relief * extent
    rng = np.random.RandomState(seed)  # 兼容老版本 numpy

    coords1d = np.arange(n) * spacing
    X, Y = np.meshgrid(coords1d, coords1d, indexing="ij")
    Z = np.zeros((n, n), dtype=float)

    if mode == "flat":
        return X, Y, Z

    if mode in ("smooth", "mixed"):
        # 低频正弦起伏：大尺度平滑山形
        kx = rng.uniform(1.0, 2.0)
        ky = rng.uniform(1.0, 2.0)
        phx = rng.uniform(0, 2 * math.pi)
        phy = rng.uniform(0, 2 * math.pi)
        Z += 0.6 * amp * np.sin(2 * math.pi * kx * X / extent + phx) \
            * np.cos(2 * math.pi * ky * Y / extent + phy)

    if mode in ("mountains", "mixed"):
        # 若干高斯山包/凹谷（正负号混合）
        num_bumps = max(3, n // 8)
        for _ in range(num_bumps):
            cx = rng.uniform(0, extent)
            cy = rng.uniform(0, extent)
            sigma = rng.uniform(0.10, 0.22) * extent
            sign = 1.0 if rng.random_sample() < 0.6 else -1.0  # 偏向山包
            Z += sign * amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    if mode in ("ridges", "mixed"):
        # 几道带缺口的山脊：高而窄的墙，迫使最短路绕到垭口（不切断图，只抬高翻越代价）
        num_ridges = 1 if mode == "mixed" else 2
        for _ in range(num_ridges):
            horizontal = rng.random_sample() < 0.5
            pos = rng.uniform(0.3, 0.7) * extent          # 山脊所在位置
            width = rng.uniform(0.03, 0.06) * extent       # 山脊厚度
            gap_center = rng.uniform(0.2, 0.8) * extent     # 垭口中心
            gap_half = rng.uniform(0.08, 0.14) * extent     # 垭口半宽
            axis = Y if horizontal else X      # 沿哪个轴形成墙
            along = X if horizontal else Y     # 垭口开在哪个方向
            wall = 1.2 * amp * np.exp(-((axis - pos) ** 2) / (2 * width ** 2))
            gate = 1.0 - np.exp(-((along - gap_center) ** 2) / (2 * gap_half ** 2))
            Z += wall * gate  # 垭口处 gate->0，墙被压低，形成通道

    return X, Y, Z


def write_off(path, X, Y, Z):
    """把高度场写成 OFF 三角网格。每个单元 (i,j) 切成两个三角形。"""
    n = X.shape[0]
    n_v = n * n
    n_f = 2 * (n - 1) * (n - 1)

    def vid(i, j):
        return i * n + j

    lines = ["OFF", f"{n_v} {n_f} 0"]
    for i in range(n):
        for j in range(n):
            lines.append(f"{X[i, j]:.6f} {Y[i, j]:.6f} {Z[i, j]:.6f}")
    for i in range(n - 1):
        for j in range(n - 1):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            # 两个三角形，保持一致绕向
            lines.append(f"3 {a} {b} {c}")
            lines.append(f"3 {a} {c} {d}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n_v, n_f


def terrain_stats(Z):
    """打印一些便于判断「是否适配算法」的统计量。"""
    z = Z.flatten()
    return {
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_range": float(z.max() - z.min()),
        "z_std": float(z.std()),
    }


def build_parser():
    p = argparse.ArgumentParser(description="生成适配距离分解算法的网格地形 .off")
    p.add_argument("--grid", type=int, nargs="+", default=[10, 20, 40],
                   help="每边顶点数列表；如 --grid 10 20 40 生成 100/400/1600 点三张图")
    p.add_argument("--mode", type=str, default="mixed",
                   choices=["flat", "smooth", "mountains", "ridges", "mixed"],
                   help="高度场类型；mixed=起伏山丘+带缺口山脊（默认，最适配算法）")
    p.add_argument("--relief", type=float, default=0.4,
                   help="起伏幅度（占地图水平范围的比例）；越大地形越陡、问题越非平凡")
    p.add_argument("--spacing", type=float, default=1.0, help="相邻网格点水平间距")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--out_dir", type=str, default="data/generated", help="输出文件夹（相对路径按项目根解析）")
    p.add_argument("--out_suffix", type=str, default="", help="文件名后缀（便于区分模式，如 _flat）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(project_root, args.out_dir)
    print(f"[gen] mode={args.mode} relief={args.relief} spacing={args.spacing} seed={args.seed}")
    for n in args.grid:
        if n < 2:
            print(f"[gen] 跳过 grid={n}（至少需要 2）")
            continue
        X, Y, Z = height_field(n, args.spacing, args.mode, args.relief, args.seed)
        fname = f"terrain_grid_{n}x{n}_{n * n}v{args.out_suffix}.off"
        path = os.path.join(out_dir, fname)
        n_v, n_f = write_off(path, X, Y, Z)
        st = terrain_stats(Z)
        print(
            f"[gen] {path}  |V|={n_v} faces={n_f}  "
            f"z_range={st['z_range']:.3f} z_std={st['z_std']:.3f}"
        )
    print("[gen] done.")
