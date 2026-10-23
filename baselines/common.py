# -*- coding: utf-8 -*-
"""EP_high 基线对比：数据加载 / 图构建 / 切分 / 指标。

口径与主实验完全一致：
  - 图 = 三角网格无向去重边，边权 = 3D 欧氏边长
  - 标签 = 精确表面测地距离（flip-out），来自 outputs/cache_ep_high/surface/surface_*.csv
  - 切分 = 80/10/10，random.seed(42)，按 CSV 行序（与 preprocess.split_distance_dataset 一致）
  - 指标 = MRE(mean |pred-true|/true)、MAE、RMSE
"""
import csv
import math
import random

import numpy as np


def load_mesh(off_path):
    from build_highway import load_off
    V, F = load_off(off_path)
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def build_graph(V, F):
    """无向去重边 + 3D 欧氏边长。"""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_len = np.linalg.norm(V[src] - V[dst], axis=1).astype(np.float32)
    return edge_index, edge_len


def compute_normals(V, F):
    """逐顶点法向（面法向按顶点累加后归一化）。"""
    n = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    nrm = np.linalg.norm(n, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return (n / nrm).astype(np.float32)


def load_labels(csv_path, col='true_distance'):
    """读精确测地标签 CSV；跳过 inf/非有限值。"""
    rows = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                d = float(r[col])
            except Exception:
                continue
            if not math.isfinite(d):
                continue
            rows.append((int(r['s']), int(r['t']), d))
    return rows


def split_samples(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """与 preprocess.split_distance_dataset 逐行等价（同 seed、同输入顺序）。"""
    data = list(samples)
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + val_ratio))
    return data[:a], data[a:b], data[b:]


def metrics(y_true, y_pred, eps=1e-9):
    """与 baseline.compute_metrics / model.compute_distance_metrics 同口径。"""
    n = len(y_true)
    ae = [abs(p - t) for p, t in zip(y_pred, y_true)]
    mae = sum(ae) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n)
    rel = sum(a / (t + eps) for a, t in zip(ae, y_true)) / n
    return {'mae': mae, 'rmse': rmse, 'relative_error': rel}


def euclidean_pairs(V, pairs):
    """3D 直线距离（纯欧氏参照）。"""
    s = np.array([p[0] for p in pairs], dtype=np.int64)
    t = np.array([p[1] for p in pairs], dtype=np.int64)
    return np.linalg.norm(V[s] - V[t], axis=1)
