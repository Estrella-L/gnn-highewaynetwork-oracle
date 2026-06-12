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

    dist_from_boundary = {b: _dijkstra(adj, b) for b in boundary_sorted}
    edge_w = build_highway_edges(adj, boundary, leaf_of, dist_from_boundary)

    # access_dist[node][local] = 节点 node 到第 local 个高速入口的图最短路
    n_full = len(graph_info[0])
    K = len(boundary_sorted)
    g2l = {g: i for i, g in enumerate(boundary_sorted)}
    access_dist = [[float("inf")] * K for _ in range(n_full)]
    for local, g in enumerate(boundary_sorted):
        dl = dist_from_boundary[g]
        for node in range(n_full):
            access_dist[node][local] = dl[node]

    # 高速图内部 local 邻接 + 两两最短路
    local_u, local_v = [], []
    hadj = [[] for _ in range(K)]
    for (u, v), w in edge_w.items():
        if u in g2l and v in g2l:
            lu, lv = g2l[u], g2l[v]
            local_u.append(lu)
            local_v.append(lv)
            hadj[lu].append((lv, w))
    highway_pair_dist = [_dijkstra(hadj, s) for s in range(K)]

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
