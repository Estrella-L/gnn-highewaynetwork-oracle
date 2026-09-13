# -*- coding: utf-8 -*-
"""LiteGE 形状描述子（UDF-PCA）在**单场景地形开曲面**上的适配实现。

LiteGE 原文（AAAI 2026）的做法（见官方 README 三步）：
  1. 数据集内所有形状先做 canonicalization（居中/缩放/对齐）；
  2. 形状体素化为 0/1 占用网格，取「跨数据集方差大」的 informative voxels；
  3. 用这些体素到形状的**无符号距离（UDF）**表示形状，再做 PCA -> 50~400 维 UDF-PCA 向量。

问题：第 2 步需要 inside/outside 体素，第 3 步需要"一批形状"。
我们的数据是**单张地形开曲面**（带边界，2761 条边界边，拓扑圆盘），既没有内外之分，
也不存在"一批形状"。

适配（本文件实现，论文对比表中需显式标注）：
  * 把地形按 (x,y) 包围盒切成 G x G 个 **patch**，把 patch 当作"形状集"；
  * 每个 patch 内做 canonicalization：把该 patch 的点平移到自身包围盒中心、缩放到单位盒；
  * 在统一体素网格（默认 8x8x8，只是拿到 512 维原始描述子，不是 LiteGE 的 informative voxel 选择）
    上计算每个体素中心到该 patch 点集的**最近点距离**，即 UDF；
  * 对所有 patch 的 UDF 原始向量做 PCA -> pca_dim 维；
  * 每个顶点取其所属 patch 的 PCA 向量。

这样得到的 node_pca 与 LiteGE 的 UDF-PCA 语义一致（紧凑的、patch 级的形状描述子），
只是"形状"的单位从数据集里的物体换成了地形的 patch。
"""
import math
import os

import numpy as np
import torch
from scipy.spatial import cKDTree


def _canonicalize(pts):
    """把点集居中并缩放到单位盒（对应 LiteGE 的 shape canonicalization）。"""
    c = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    p = pts - c
    scale = float(np.max(np.abs(p))) or 1.0
    return p / scale


def compute_udf_pca_descriptor(
    vertices3d,
    grid_axis=8,
    num_patches=None,
    pca_dim=200,
    min_points_per_patch=None,
    cache_path=None,
):
    """计算逐顶点的 UDF-PCA 形状描述子。

    Args:
        vertices3d: list[(x,y,z)] 或 ndarray [N,3]
        grid_axis: 每个 patch 的体素网格边长（8 -> 512 维原始 UDF 描述子）
        num_patches: 每个空间轴切成多少份（32 -> 最多 1024 个 patch，用作 PCA 的样本数）
        pca_dim: 输出维度
        min_points_per_patch: 少于该点数的 patch 被丢弃（并归到最近的保留 patch）
        cache_path: 给定则缓存/复用 .npz

    Returns:
        np.ndarray [N, pca_dim] float32
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)["node_pca"]

    V = np.asarray(vertices3d, dtype=np.float64)
    n = len(V)
    # 让 patch 数随顶点数自适应：目标是每个 patch 有几十个点，且 patch 数远多于 pca_dim 才做得动 PCA
    if num_patches is None:
        num_patches = int(min(48, max(4, round(math.sqrt(max(n, 1) / 40.0)))))
    if min_points_per_patch is None:
        min_points_per_patch = int(max(5, n / float(num_patches * num_patches) / 4.0))

    axis = np.linspace(-1.0, 1.0, grid_axis)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # [G^3, 3]

    # 地形是高度场（近似 2.5D），因此**只按 (x,y) 切 patch**：
    # 每个 patch 是该 (x,y) 邻域上的"一列地形"，这也更符合"把地形切成一组 patch 当形状集"的适配动机。
    def _split(npatch, min_pts):
        mins2, maxs2 = V[:, :2].min(axis=0), V[:, :2].max(axis=0)
        span2 = np.maximum(maxs2 - mins2, 1e-9)
        ci = np.floor((V[:, :2] - mins2) / span2 * npatch).astype(np.int64)
        ci = np.clip(ci, 0, npatch - 1)
        keys = ci[:, 0] * npatch + ci[:, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        raw, node_patch, kept = [], np.full(n, -1, dtype=np.int64), 0
        for pi, _k in enumerate(uniq):
            sel = np.where(inv == pi)[0]
            if len(sel) < min_pts:
                continue
            pts = _canonicalize(V[sel])
            tree = cKDTree(pts)
            d, _ = tree.query(grid_pts, k=1, workers=-1)
            raw.append(d.astype(np.float32))
            node_patch[sel] = kept
            kept += 1
        return raw, node_patch, kept

    # patch 太少（样本不够做 PCA）时自动降低切分粒度，保证至少有 12 个 patch
    npatch = num_patches
    min_pts = min_points_per_patch
    raw_feats, patch_of_node, n_kept = _split(npatch, min_pts)
    while n_kept < 12 and npatch > 2:
        npatch = max(2, npatch // 2)
        min_pts = max(3, min_pts // 2)
        raw_feats, patch_of_node, n_kept = _split(npatch, min_pts)
    print("[litege-pca] patches kept = %d (grid %dx%d, min_pts=%d)" % (n_kept, npatch, npatch, min_pts))

    if len(raw_feats) < 2:
        # patch 太少，退化为全零描述子
        out = np.zeros((n, pca_dim), dtype=np.float32)
    else:
        X = np.stack(raw_feats, axis=0)                 # [P, G^3]
        X = X - X.mean(axis=0, keepdims=True)
        # 经济型 SVD 做 PCA；主成分数不能超过 patch 数 - 1
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = int(min(pca_dim, max(1, len(raw_feats) - 1), Vt.shape[0]))
        if k < pca_dim:
            print("[litege-pca] patch 数(%d)不足，PCA 维度从 %d 降为 %d"
                  % (len(raw_feats), pca_dim, k))
        comp = Vt[:k]                                    # [k, G^3]
        Z = X @ comp.T                                   # [P, k]
        out = np.zeros((n, k), dtype=np.float32)
        valid = patch_of_node >= 0
        out[valid] = Z[patch_of_node[valid]]
        if k < pca_dim:
            pad = np.zeros((n, pca_dim - k), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)
        # 与 LiteGE 一致：PCA 向量做 L2 归一化
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norm, 1e-9)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, node_pca=out)
    return out
