# .off 地形 → 全局图 + 四叉树分区 + 高速骨干网络（对齐 EAR-Oracle）。
"""
.off 地形网格 → 全局图 + 四叉树分区 + 高速骨干网络。

本模块是 EAR-Oracle (SIGMOD'2023, weighted_distance_oracle) 中 quad.h 四叉树分区 + 边界点高速
方案的图层面实现，输入直接采用 EAR-Oracle 的 **.off 2-manifold 三角网格**：

  EAR-Oracle (C++/CGAL)                  →  本实现 (纯 Python)
  -----------------------------------------------------------------------
  .off 三角网格 (x y z + 面)             →  load_off + build_mesh_graph
  Quad::quadTree 递归四分 (NW/NE/SW/SE)  →  build_quadtree (真正的递归四叉树)
  盒子边界顶点 boundary_points_id        →  有跨叶子邻边的顶点 = 高速(边界)节点
  盒内 boundary→point 的 Dijkstra 预处理 →  盒内 boundary 节点间的全图最短路 (transit 边)
  WSPD spanner 高速网络                  →  高速节点 + (原始跨界边 ∪ 盒内 transit 边)

与之前 grid 版本的区别：这是**真正的四叉树**——每个节点递归四分为 NW/NE/SW/SE 子节点，
支持两种叶子策略：
  - 自适应 (默认)：节点内点数 > capacity 且深度 < max_depth 时才继续四分（叶子大小不均，真正的 quadtree）。
  - 均匀 (--uniform)：一律四分到 max_depth（4^max_depth 个等大叶子，对应 EAR-Oracle 的非自适应模式）。

边权使用顶点间的 3D 欧氏距离（图最短路即测地距离的折线近似；EAR-Oracle 用更精确的 Snell 距离，
这里在抽象图层面用折线距离近似，作为 GNN 的回归监督已足够）。

CLI:
  python build_highway.py --off_file sample_terrain.off --max_depth 3 --capacity 32 --out_prefix terrain
输出 <prefix>_partition.csv（每节点的叶子编号 + 坐标）。
"""
import argparse
import heapq
import math
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# .off 读取（与 EAR-Oracle 的输入格式一致）
# ---------------------------------------------------------------------------
def load_off(off_path):
    """
    读取 .off 三角网格。鲁棒地处理 OFF 头与计数行的不同排版。

    Returns:
        (vertices, faces):
          vertices: list[(x, y, z)]
          faces:    list[list[int]]  顶点索引（一般为三角形）
    """
    with open(off_path, "r", encoding="utf-8", errors="ignore") as f:
        tokens = f.read().split()
    if not tokens:
        raise ValueError(f"empty .off file: {off_path}")
    idx = 0
    if tokens[0].upper().startswith("OFF"):
        idx = 1  # 跳过 'OFF'/'COFF'/'NOFF' 头
    n_v = int(tokens[idx]); n_f = int(tokens[idx + 1]); idx += 3  # 跳过 nV nF nE
    vertices = []
    for _ in range(n_v):
        x = float(tokens[idx]); y = float(tokens[idx + 1]); z = float(tokens[idx + 2])
        idx += 3
        vertices.append((x, y, z))
    faces = []
    for _ in range(n_f):
        k = int(tokens[idx]); idx += 1
        face = [int(tokens[idx + j]) for j in range(k)]
        idx += k
        faces.append(face)
    return vertices, faces


# ---------------------------------------------------------------------------
# 网格 → 全局图（graph_info 结构，与 preprocess 其余部分兼容）
# graph_info = [ids, labels, degree, [edge_u, edge_v], edge_w, neighbors, label_dict]
# ---------------------------------------------------------------------------
def build_mesh_graph(vertices, faces):
    """由三角网格构造无向加权图：节点=顶点，边=网格边，权=3D 欧氏距离。"""
    n = len(vertices)
    nbr = [set() for _ in range(n)]
    for face in faces:
        m = len(face)
        for i in range(m):
            a = face[i]
            b = face[(i + 1) % m]
            if a != b and 0 <= a < n and 0 <= b < n:
                nbr[a].add(b)
                nbr[b].add(a)
    neighbors = [sorted(nbr[i]) for i in range(n)]
    edge_u, edge_v, edge_w = [], [], []
    for u in range(n):
        xu, yu, zu = vertices[u]
        for v in neighbors[u]:
            xv, yv, zv = vertices[v]
            w = math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2 + (zu - zv) ** 2)
            edge_u.append(u)
            edge_v.append(v)
            edge_w.append(w)
    degree = [len(neighbors[i]) for i in range(n)]
    ids = list(range(n))
    labels = [0] * n
    label_dict = defaultdict(list)
    for i in range(n):
        label_dict[0].append(i)
    coords = {i: (vertices[i][0], vertices[i][1]) for i in range(n)}  # 用 (x,y) 做分区/特征
    graph_info = [ids, labels, degree, [edge_u, edge_v], edge_w, neighbors, label_dict]
    return graph_info, coords


def _adj_from_graph_info(graph_info, weighted=True):
    n = len(graph_info[0])
    eu, ev = graph_info[3][0], graph_info[3][1]
    ew = graph_info[4]
    adj = [[] for _ in range(n)]
    for i in range(len(eu)):
        w = float(ew[i]) if weighted else 1.0
        adj[eu[i]].append((ev[i], max(1e-9, w)))
    return adj


def _dijkstra(adj, src):
    dist = [float("inf")] * len(adj)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _try_import_scipy():
    """惰性探测 scipy；不可用则返回 (None, None)，调用方回退纯 Python Dijkstra。"""
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra as sp_dijkstra
        return sp, sp_dijkstra
    except Exception:
        return None, None


def iter_source_distances(n, edge_u, edge_v, edge_w, sources, weighted=True, chunk=256):
    """
    依次产出 (source, dist)：dist 为长度 n 的距离序列（scipy 下是 np.ndarray，回退下是 list）。

    - 有 scipy：用**分块**的 C 版多源 Dijkstra（快 1~2 个数量级）；每块只算 [chunk, n]，
      内存峰值 O(chunk×n)，不会退回 K×N 稠密矩阵。
    - 无 scipy：回退纯 Python heapq，逐源计算（结果逐位一致）。

    两分支都给**精确最短路**，数值等价（无向图，directed=False）。
    """
    import numpy as np

    srcs = list(sources)
    sp, sp_dijkstra = _try_import_scipy()
    if sp is not None:
        w = np.asarray(edge_w, dtype=np.float64) if weighted else np.ones(len(edge_u), dtype=np.float64)
        w = np.maximum(1e-9, w)
        csr = sp.csr_matrix(
            (w, (np.asarray(edge_u, dtype=np.int64), np.asarray(edge_v, dtype=np.int64))),
            shape=(n, n),
        )
        for i in range(0, len(srcs), chunk):
            batch = srcs[i:i + chunk]
            dmat = sp_dijkstra(csr, directed=False, indices=batch)  # [len(batch), n] float64
            for j, s in enumerate(batch):
                yield s, dmat[j]
    else:
        adj = [[] for _ in range(n)]
        for idx in range(len(edge_u)):
            u, v = edge_u[idx], edge_v[idx]
            if 0 <= u < n and 0 <= v < n:
                ww = float(edge_w[idx]) if weighted else 1.0
                adj[u].append((v, max(1e-9, ww)))
        for s in srcs:
            yield s, _dijkstra(adj, s)


# ---------------------------------------------------------------------------
# 真正的递归四叉树（对应 EAR-Oracle 的 Quad::quadTree）
# ---------------------------------------------------------------------------
class _QuadNode:
    __slots__ = ("x_min", "y_min", "x_max", "y_max", "children", "points", "leaf_id")

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max
        self.children = None  # [SW, SE, NW, NE] 或 None
        self.points = []
        self.leaf_id = -1


def build_quadtree(coords, max_depth=3, capacity=32, adaptive=True):
    """
    构造递归四叉树并把每个节点分到叶子盒。

    Args:
        adaptive: True 时，仅当盒内点数 > capacity 且深度 < max_depth 才四分（真正的自适应四叉树）；
                  False 时一律四分到 max_depth（均匀 4^max_depth 个叶子，对应 EAR-Oracle 非自适应模式）。
    Returns:
        (leaf_of, num_leaves, root):
          leaf_of: dict[node_id -> leaf_id]
    """
    ids = list(coords.keys())
    xs = [coords[i][0] for i in ids]
    ys = [coords[i][1] for i in ids]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # 轻微扩边，保证最大坐标点也落在盒内
    pad_x = max(1e-9, (x_max - x_min) * 1e-6)
    pad_y = max(1e-9, (y_max - y_min) * 1e-6)
    root = _QuadNode(x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)
    root.points = ids

    leaves = []

    def subdivide(node, depth):
        if adaptive:
            should_split = depth < max_depth and len(node.points) > capacity
        else:
            should_split = depth < max_depth
        if not should_split or len(node.points) == 0:
            leaves.append(node)
            return
        px = 0.5 * (node.x_min + node.x_max)
        py = 0.5 * (node.y_min + node.y_max)
        # SW / SE / NW / NE
        kids = [
            _QuadNode(node.x_min, node.y_min, px, py),
            _QuadNode(px, node.y_min, node.x_max, py),
            _QuadNode(node.x_min, py, px, node.y_max),
            _QuadNode(px, py, node.x_max, node.y_max),
        ]
        for pid in node.points:
            x, y = coords[pid]
            ix = 1 if x >= px else 0
            iy = 1 if y >= py else 0
            kids[iy * 2 + ix].points.append(pid)
        node.children = kids
        node.points = []
        for ch in kids:
            subdivide(ch, depth + 1)

    subdivide(root, 0)

    leaf_of = {}
    occupied = 0
    for lid, leaf in enumerate(leaves):
        leaf.leaf_id = lid
        if leaf.points:
            occupied += 1
        for pid in leaf.points:
            leaf_of[pid] = lid
    return leaf_of, len(leaves), occupied


# ---------------------------------------------------------------------------
# 边界(高速)节点 + 高速边
# ---------------------------------------------------------------------------
def find_boundary_nodes(graph_info, leaf_of):
    """有跨叶子邻边的节点 = 高速(边界)节点（对应 EAR-Oracle 的 boundary_points_id）。"""
    neighbors = graph_info[5]
    boundary = set()
    for u in range(len(graph_info[0])):
        lu = leaf_of.get(u)
        for v in neighbors[u]:
            if leaf_of.get(v) != lu:
                boundary.add(u)
                break
    return boundary


def build_highway_edges(adj, boundary, leaf_of, dist_from_boundary):
    """高速节点之间：原始跨界边 ∪ 盒内 boundary-to-boundary 全图最短路 transit 边。"""
    edge_w = {}

    def add(u, v, w):
        if u == v:
            return
        key = (u, v)
        if key not in edge_w or w < edge_w[key]:
            edge_w[key] = w

    for u in boundary:
        for v, w in adj[u]:
            if v in boundary:
                add(u, v, w)
                add(v, u, w)

    cell_members = defaultdict(list)
    for u in boundary:
        cell_members[leaf_of[u]].append(u)
    for members in cell_members.values():
        for a in members:
            da = dist_from_boundary[a]
            for b in members:
                if a != b and da[b] != float("inf"):
                    add(a, b, da[b])
    return edge_w


# ---------------------------------------------------------------------------
# 端到端：.off → graph + 四叉树分区 + 高速上下文（供训练/推理使用）
# ---------------------------------------------------------------------------
def build_pipeline_inputs(
    off_path,
    max_depth=3,
    capacity=32,
    adaptive=True,
    weighted=True,
    feature_dim=64,
    device="cpu",
):
    """从 .off 直接产出训练/推理所需的全部对象。

    Returns:
        (graph_info, coords, leaf_of, num_leaves, highway_context)
    """
    vertices, faces = load_off(off_path)
    graph_info, coords = build_mesh_graph(vertices, faces)
    leaf_of, num_leaves, occupied = build_quadtree(coords, max_depth, capacity, adaptive)
    adj = _adj_from_graph_info(graph_info, weighted=weighted)

    boundary = find_boundary_nodes(graph_info, leaf_of)
    if not boundary:
        boundary = set(range(len(graph_info[0])))
    boundary_sorted = sorted(boundary)

    # 内存优化：用 float32 numpy 存距离（省 ~8x），并**流式**跑边界 Dijkstra
    # （每个高速点算完立刻填列 + 就地生成盒内 transit 边后丢弃，峰值从 K×N 降到 1×N）。
    import numpy as np

    n_full = len(graph_info[0])
    K = len(boundary_sorted)
    g2l = {g: i for i, g in enumerate(boundary_sorted)}

    # 高速节点按叶子盒分组
    cell_members = defaultdict(list)
    for u in boundary_sorted:
        cell_members[leaf_of[u]].append(u)

    edge_w = {}

    def _add_edge(u, v, w):
        if u == v:
            return
        key = (u, v)
        if key not in edge_w or w < edge_w[key]:
            edge_w[key] = w

    # (1) 原始跨界边：两端都是高速节点的原图边
    for u in boundary_sorted:
        for v, w in adj[u]:
            if v in boundary:
                _add_edge(u, v, w)
                _add_edge(v, u, w)

    # (2) 流式：每个高速点一次全图 Dijkstra → 填 access_dist 列 + 盒内 transit 边，随即丢弃
    #     底层用 iter_source_distances（有 scipy 走 C 分块，否则回退 Python），保持 O(chunk×N) 内存。
    eu_g, ev_g, ew_g = graph_info[3][0], graph_info[3][1], graph_info[4]
    access_dist = np.full((n_full, K), np.inf, dtype=np.float32)
    for g, dl in iter_source_distances(n_full, eu_g, ev_g, ew_g, boundary_sorted, weighted=weighted):
        local = g2l[g]
        access_dist[:, local] = np.asarray(dl, dtype=np.float32)
        for b in cell_members[leaf_of[g]]:
            db = dl[b]
            if b != g and db != float("inf"):
                _add_edge(g, b, float(db))

    # 高速图内部 local 邻接（收集边+权）+ 两两最短路（float32 存储）
    local_u, local_v, local_w = [], [], []
    for (u, v), w in edge_w.items():
        if u in g2l and v in g2l:
            local_u.append(g2l[u])
            local_v.append(g2l[v])
            local_w.append(w)
    highway_pair_dist = np.full((K, K), np.inf, dtype=np.float32)
    for s, dl in iter_source_distances(K, local_u, local_v, local_w, range(K), weighted=weighted):
        highway_pair_dist[s, :] = np.asarray(dl, dtype=np.float32)

    # 张量化与特征（需要 torch，延迟导入，便于在无 torch 环境下单测纯图部分）
    from preprocess import build_highway_context

    context = build_highway_context(
        graph_info=graph_info,
        coords=coords,
        highway_global_ids=boundary_sorted,
        local_edges=(local_u, local_v),
        access_dist=access_dist,
        highway_pair_dist=highway_pair_dist,
        leaf_of=leaf_of,
        feature_dim=feature_dim,
        device=device,
    )
    context["num_leaves"] = num_leaves
    context["num_leaves_occupied"] = occupied
    return graph_info, coords, leaf_of, num_leaves, context


def _file_fingerprint(path):
    """对 .off 文件内容取短哈希，纳入缓存键，避免同名但内容不同的图命中过期缓存。"""
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _pipeline_cache_key(off_path, max_depth, capacity, adaptive, feature_dim):
    """缓存键：.off 文件名 + 内容指纹 + 四叉树参数 + 特征维度，任一不同即视为不同缓存。"""
    base = os.path.splitext(os.path.basename(off_path))[0]
    mode = "ada" if adaptive else "uni"
    fp = _file_fingerprint(off_path)
    return f"{base}_d{max_depth}_c{capacity}_{mode}_f{feature_dim}_{fp}"


def _dump_partition_audit(path, graph_info, coords, leaf_of, highway_ids):
    """审查文件①：每个顶点属于哪个叶子盒、是否高速点、坐标。"""
    hw = set(highway_ids)
    with open(path, "w", encoding="utf-8") as f:
        f.write("node,leaf_id,is_highway,x,y\n")
        for nid in graph_info[0]:
            x, y = coords.get(nid, (float("nan"), float("nan")))
            f.write(f"{nid},{leaf_of.get(nid, -1)},{int(nid in hw)},{x},{y}\n")


def _dump_highway_edges_audit(path, context):
    """审查文件②：高速图的边（还原成原图全局节点 id，无向去重）。"""
    ids = context["highway_global_ids"]
    ei = context["edge_index_highway"]
    us, vs = ei[0].tolist(), ei[1].tolist()
    seen = set()
    with open(path, "w", encoding="utf-8") as f:
        f.write("u_global,v_global\n")
        for lu, lv in zip(us, vs):
            gu, gv = ids[lu], ids[lv]
            if gu == gv:
                continue
            key = (gu, gv) if gu <= gv else (gv, gu)
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{key[0]},{key[1]}\n")


def build_pipeline_inputs_cached(
    off_path,
    max_depth=3,
    capacity=32,
    adaptive=True,
    weighted=True,
    feature_dim=64,
    device="cpu",
    cache_dir=None,
    audit=True,
):
    """带磁盘缓存的 build_pipeline_inputs。

    首次计算后把（图 + 分区 + 高速上下文）存成 .pt，之后命中缓存直接加载，
    跳过四叉树分区与每个高速节点的全图 Dijkstra 预处理（训练/推理启动不再重复等待）。
    同时导出两份**人类可读的审查 CSV**（分区表 + 高速边表）。

    缓存按 `_pipeline_cache_key` 区分（.off 名 + max_depth/capacity/模式/feature_dim）；
    缓存张量统一存在 CPU，对 cpu/gpu 通用（逐样本前向时再 .to(device)）。
    cache_dir=None 时禁用缓存，每次重新计算（等价于 build_pipeline_inputs）。
    """
    if cache_dir is None:
        return build_pipeline_inputs(off_path, max_depth, capacity, adaptive, weighted, feature_dim, device)

    import torch

    os.makedirs(cache_dir, exist_ok=True)
    key = _pipeline_cache_key(off_path, max_depth, capacity, adaptive, feature_dim)
    cache_pt = os.path.join(cache_dir, key + ".pt")

    if os.path.exists(cache_pt):
        data = torch.load(cache_pt, map_location="cpu")
        print(f"[cache] 命中，直接加载高速上下文: {cache_pt}（跳过分区+Dijkstra 预处理）")
        return data["graph_info"], data["coords"], data["leaf_of"], data["num_leaves"], data["context"]

    graph_info, coords, leaf_of, num_leaves, context = build_pipeline_inputs(
        off_path, max_depth, capacity, adaptive, weighted, feature_dim, device="cpu"
    )
    torch.save(
        {
            "graph_info": graph_info,
            "coords": coords,
            "leaf_of": leaf_of,
            "num_leaves": num_leaves,
            "context": context,
        },
        cache_pt,
    )
    print(f"[cache] 已保存高速上下文: {cache_pt}")
    if audit:
        part_csv = os.path.join(cache_dir, key + "_partition.csv")
        edges_csv = os.path.join(cache_dir, key + "_highway_edges.csv")
        _dump_partition_audit(part_csv, graph_info, coords, leaf_of, context["highway_global_ids"])
        _dump_highway_edges_audit(edges_csv, context)
        print(f"[cache] 审查文件: {part_csv} / {edges_csv}")
    return graph_info, coords, leaf_of, num_leaves, context


# ---------------------------------------------------------------------------
# CLI：仅做纯图层面的分区+高速派生与可视化导出（不依赖 torch）
# ---------------------------------------------------------------------------
def write_partition_csv(path, coords, leaf_of, boundary):
    with open(path, "w", encoding="utf-8") as f:
        f.write("node_id,leaf_id,is_highway,x,y\n")
        for nid in sorted(coords.keys()):
            x, y = coords[nid]
            f.write(f"{nid},{leaf_of.get(nid, -1)},{1 if nid in boundary else 0},{x},{y}\n")


def build_parser():
    p = argparse.ArgumentParser(description="Derive quadtree partition + highway backbone from a .off terrain mesh.")
    p.add_argument("--off_file", type=str, required=True, help="input .off terrain mesh path")
    p.add_argument("--max_depth", type=int, default=3, help="quadtree max depth")
    p.add_argument("--capacity", type=int, default=32, help="max points per leaf (adaptive mode)")
    p.add_argument("--uniform", action="store_true", help="uniform quadtree (split to max_depth) instead of adaptive")
    p.add_argument("--out_prefix", type=str, required=True, help="output file name prefix")
    p.add_argument("--file_folder", type=str, default=".", help="base folder for relative paths / outputs")
    return p


def _main():
    args = build_parser().parse_args()
    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(args.file_folder, args.off_file)
    vertices, faces = load_off(off_path)
    graph_info, coords = build_mesh_graph(vertices, faces)
    leaf_of, num_leaves, occupied = build_quadtree(
        coords, args.max_depth, args.capacity, adaptive=not args.uniform
    )
    boundary = find_boundary_nodes(graph_info, leaf_of)
    if not boundary:
        boundary = set(range(len(graph_info[0])))

    part_path = os.path.join(args.file_folder, f"{args.out_prefix}_partition.csv")
    write_partition_csv(part_path, coords, leaf_of, boundary)

    n = len(graph_info[0])
    n_edges = len(graph_info[3][0]) // 2
    print(f"[build_highway] .off: |V|={n}, |F|={len(faces)}, undirected |E|={n_edges}")
    print(f"[build_highway] quadtree: mode={'uniform' if args.uniform else 'adaptive'}, "
          f"max_depth={args.max_depth}, capacity={args.capacity}")
    print(f"[build_highway] leaves={num_leaves} (occupied={occupied})")
    print(f"[build_highway] highway(boundary) nodes={len(boundary)} ({100.0 * len(boundary) / max(1, n):.1f}% of V)")
    print(f"[build_highway] wrote: {part_path}")


if __name__ == "__main__":
    _main()
