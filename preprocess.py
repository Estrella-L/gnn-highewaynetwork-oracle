# 数据预处理：距离样本采样、特征构造、highway 上下文与 s/t 子图构造（供训练/推理调用）。
import math
import time
import random
import heapq
import os
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict


def _all_train_and_test(training_percent, name_list):
    example_name = name_list[0]
    train_name_list = list()
    test_name_list = list()
    potential_names_4 = list()
    potential_names_8 = list()
    potential_names_12 = list()
    potential_names_16 = list()
    if 'youtube' in example_name or 'eu2005' in example_name or 'patent' in example_name:
        for i in range(len(name_list)):
            if '_4_' in name_list[i]:
                potential_names_4.append(name_list[i])
            elif '_8_' in name_list[i]:
                potential_names_8.append(name_list[i])
        train_name_list.extend(potential_names_4[:math.floor(len(potential_names_4) * training_percent)])
        train_name_list.extend(potential_names_8[:math.floor(len(potential_names_8) * training_percent)])
        test_name_list.extend(potential_names_4[math.floor(len(potential_names_4) * training_percent):])
        test_name_list.extend(potential_names_8[math.floor(len(potential_names_8) * training_percent):])
        return train_name_list, test_name_list
    else:
        for i in range(len(name_list)):
            if '_4_' in name_list[i]:
                potential_names_4.append(name_list[i])
            elif '_8_' in name_list[i]:
                potential_names_8.append(name_list[i])
            elif '_16_' in name_list[i]:
                potential_names_16.append(name_list[i])
        # print(len(potential_names_4))
        train_name_list.extend(potential_names_4[:math.floor(len(potential_names_4) * training_percent)])
        train_name_list.extend(potential_names_8[:math.floor(len(potential_names_8) * training_percent)])
        train_name_list.extend(potential_names_16[:math.floor(len(potential_names_16) * training_percent)])
        test_name_list.extend(potential_names_4[math.floor(len(potential_names_4) * training_percent):])
        test_name_list.extend(potential_names_8[math.floor(len(potential_names_8) * training_percent):])
        test_name_list.extend(potential_names_16[math.floor(len(potential_names_16) * training_percent):])
        return train_name_list, test_name_list




def train_and_test(query_vertices_num, training_percent, name_list):
    train_name_list = list()
    test_name_list = list()
    if query_vertices_num == '4':
        target_string = 'dense_4_'
    elif query_vertices_num == '8':
        target_string = '_8_'
    elif query_vertices_num == '12':
        target_string = '_12_'
    elif query_vertices_num == '16':
        target_string = '_16_'
    elif query_vertices_num == '24':
        target_string = '_24_'
    elif query_vertices_num == '32':
        target_string = '_32_'
    elif query_vertices_num == 'all':
        return _all_train_and_test(training_percent, name_list)
    else:
        raise NotImplementedError('The query vertex number input is not supported')
    potential_names = list()
    for i in range(len(name_list)):
        if target_string in name_list[i]:
            potential_names.append(name_list[i])
    total_num = len(potential_names)
    train_num = math.floor(total_num*training_percent)
    test_num = total_num - train_num
    for i in range(train_num):
        train_name_list.append(potential_names[i])
    for i in range(test_num):
        test_name_list.append(potential_names[train_num+i])

    return train_name_list, test_name_list


def _build_weighted_adj_list(graph_info, weighted=False):
    node_ids = graph_info[0]
    edge_u = graph_info[3][0]
    edge_v = graph_info[3][1]
    edge_w = graph_info[4] if len(graph_info) > 4 else [1] * len(edge_u)
    n = len(node_ids)
    adj = [[] for _ in range(n)]
    for i in range(len(edge_u)):
        u = edge_u[i]
        v = edge_v[i]
        if u >= n or v >= n:
            continue
        w = float(edge_w[i]) if weighted else 1.0
        w = max(1e-9, w)  # 允许真实(浮点)边权；仅防止 0/负权
        adj[u].append((v, w))
    return adj


def _dijkstra_single_source(adj, src):
    n = len(adj)
    dist = [float("inf")] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _load_samples_csv(path):
    """读取缓存的采样 CSV(s,t,distance) → sample dict 列表。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            try:
                s = int(parts[0]); t = int(parts[1]); d = float(parts[2])
            except (ValueError, IndexError):
                continue  # 表头
            samples.append({"s": s, "t": t, "distance": d})
    return samples


def _save_samples_csv(path, samples):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,t,distance\n")
        for smp in samples:
            f.write(f"{int(smp['s'])},{int(smp['t'])},{float(smp['distance'])}\n")


def _canonical_pair(s, t, undirected):
    if undirected and s > t:
        return t, s
    return s, t


def _add_random_pairs(seen, n, target_count, rng, undirected=True, predicate=None, max_trials=None):
    max_trials = max_trials or max(1000, 50 * max(1, target_count))
    tries = 0
    while len(seen) < target_count and tries < max_trials:
        tries += 1
        s = rng.randrange(n)
        t = rng.randrange(n)
        if s == t:
            continue
        s, t = _canonical_pair(s, t, undirected)
        if predicate is not None and not predicate(s, t):
            continue
        seen.add((s, t))
    return tries


def _sample_distance_gap_pairs(graph_info, cap, weighted=True, seed=42, undirected=True, candidate_factor=8):
    """Sample a candidate pool, compute true distances, then balance short/mid/long pairs.

    This is a local/diagnostic version of distance-gap query generation: it keeps the
    exact Dijkstra labels, but makes the selected supervision pairs less dominated by
    the most common distance range. It is intentionally candidate-based so large graphs
    do not require all-pairs distances.
    """
    if cap <= 0:
        raise ValueError("distance_gap sampling requires a positive distance_samples cap.")

    from build_highway import iter_source_distances

    n = len(graph_info[0])
    total_pairs = n * (n - 1) // 2 if undirected else n * (n - 1)
    pool_target = min(total_pairs, max(cap, cap * candidate_factor))
    rng = random.Random(seed)
    seen = set()
    _add_random_pairs(seen, n, pool_target, rng, undirected=undirected, max_trials=100 * pool_target)
    pair_list = list(seen)

    by_src = defaultdict(list)
    for s, t in pair_list:
        by_src[s].append(t)

    eu, ev, ew = graph_info[3][0], graph_info[3][1], graph_info[4]
    reachable = []
    for s, dist in iter_source_distances(n, eu, ev, ew, list(by_src.keys()), weighted=weighted):
        for t in by_src[s]:
            d = dist[t]
            if d != float("inf"):
                reachable.append((s, t, float(d)))

    if not reachable:
        return []

    reachable.sort(key=lambda x: x[2])
    bins = [reachable[0::3], reachable[1::3], reachable[2::3]]
    selected = []
    per_bin = cap // 3
    remainder = cap - 3 * per_bin
    for idx, bucket in enumerate(bins):
        rng.shuffle(bucket)
        take = per_bin + (1 if idx < remainder else 0)
        selected.extend(bucket[:take])

    if len(selected) < min(cap, len(reachable)):
        selected_pairs = {(s, t) for s, t, _ in selected}
        leftovers = [x for x in reachable if (x[0], x[1]) not in selected_pairs]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: min(cap, len(reachable)) - len(selected)])

    rng.shuffle(selected)
    dists = [d for _, _, d in selected]
    q1 = sorted(dists)[len(dists) // 3]
    q2 = sorted(dists)[(2 * len(dists)) // 3]
    print(
        f"[distance] distance_gap pairs: total={len(selected)} "
        f"candidate_pool={len(reachable)} q1={q1:.6f} q2={q2:.6f}"
    )
    return [{"s": s, "t": t, "distance": d} for s, t, d in selected]


def _sample_oracle_mix_pairs(n, cap, leaf_of, seed=42, undirected=True):
    """
    按 EAR-Oracle 查询口径采样：约 1/3 同叶子、1/3 跨叶子、1/3 随机 mixed。

    这里只决定 (s,t) 集合，真实距离仍由后续 Dijkstra 精确计算，因此不会改变标签定义。
    """
    if leaf_of is None:
        raise ValueError("sample_strategy='oracle_mix' requires leaf_of from the quadtree partition.")

    rng = random.Random(seed)
    by_leaf = defaultdict(list)
    for node in range(n):
        if node in leaf_of:
            by_leaf[leaf_of[node]].append(node)
    leaves = [leaf for leaf, nodes in by_leaf.items() if len(nodes) > 0]
    inner_leaves = [leaf for leaf in leaves if len(by_leaf[leaf]) >= 2]
    if not leaves or not inner_leaves:
        raise ValueError("oracle_mix sampling needs at least one non-empty leaf with two or more nodes.")

    inner_target = cap // 3
    inter_target = cap // 3
    random_target = cap - inner_target - inter_target
    seen = set()

    def same_leaf(s, t):
        return leaf_of.get(s) == leaf_of.get(t)

    def different_leaf(s, t):
        return leaf_of.get(s) != leaf_of.get(t)

    # 同分区查询：对应教授项目里的 inner-box queries。
    inner_seen_target = len(seen) + inner_target
    inner_tries = 0
    while len(seen) < inner_seen_target and inner_tries < max(1000, 50 * max(1, inner_target)):
        inner_tries += 1
        leaf = rng.choice(inner_leaves)
        s, t = rng.sample(by_leaf[leaf], 2)
        s, t = _canonical_pair(s, t, undirected)
        seen.add((s, t))

    # 跨分区查询：对应教授项目里的 inter-box queries。
    inter_seen_target = len(seen) + inter_target
    inter_tries = 0
    if len(leaves) >= 2:
        while len(seen) < inter_seen_target and inter_tries < max(1000, 50 * max(1, inter_target)):
            inter_tries += 1
            leaf_s, leaf_t = rng.sample(leaves, 2)
            s = rng.choice(by_leaf[leaf_s])
            t = rng.choice(by_leaf[leaf_t])
            s, t = _canonical_pair(s, t, undirected)
            if different_leaf(s, t):
                seen.add((s, t))

    # mixed/random 查询：不强制同/跨分区，用来保留真实随机分布。
    random_seen_target = min(cap, len(seen) + random_target)
    _add_random_pairs(seen, n, random_seen_target, rng, undirected=undirected)

    # 极端情况下某一类因为叶子太小没采满，用全局随机补齐。
    if len(seen) < cap:
        _add_random_pairs(seen, n, cap, rng, undirected=undirected, max_trials=100 * cap)

    pair_list = list(seen)
    rng.shuffle(pair_list)
    inner_count = sum(1 for s, t in pair_list if same_leaf(s, t))
    inter_count = len(pair_list) - inner_count
    print(
        f"[distance] oracle_mix pairs: total={len(pair_list)} "
        f"same_leaf={inner_count} cross_leaf={inter_count} "
        f"leaves={len(leaves)}"
    )
    return pair_list


def build_distance_samples(graph_info, num_samples=None, weighted=True, seed=42, undirected=True,
                           cache_path=None, sample_strategy="random", leaf_of=None):
    """
    构造节点对最短路监督样本（唯一、无重复、无泄漏），并对大图高效。

    做法：
      1. 先确定要用的**唯一**节点对集合：num_samples 给定且小于全部对数时，**无放回**随机抽样
         num_samples 个唯一对；否则枚举全部对（仅适合小图）。
      2. 按源点分组，每个不同源点只跑**一次** Dijkstra（有 scipy 走 C 分块，否则回退 Python），
         读出该源到其目标的距离。Dijkstra 次数 = 不同源点数，避免全 APSP。

    Args:
        num_samples (int | None): 采样上限；None 或 <=0 时尝试全部对（大图会自动设上限保护）。
        undirected (bool): True 时只取 s < t 的无向对（无向网格距离对称）。
        cache_path (str | None): 给定则：命中直接读、未命中算完写盘（同图同参数第二次起跳过全部 Dijkstra）。
        sample_strategy: "random" 为原始全局随机采样；"oracle_mix" 为 inner/inter/mixed 均衡采样；
            "distance_gap" 为短/中/长距离分桶均衡采样。
        leaf_of: oracle_mix 需要的 quadtree 叶子映射。

    Returns:
        list[dict]: [{"s": int, "t": int, "distance": float}, ...]（每对唯一、无跨集泄漏）
    """
    if cache_path and os.path.exists(cache_path):
        samples = _load_samples_csv(cache_path)
        print(f"[distance] loaded cached samples: {cache_path} ({len(samples)} pairs) [跳过采样 Dijkstra]")
        return samples

    from build_highway import iter_source_distances  # 惰性导入，避免循环依赖

    random.seed(seed)
    n = len(graph_info[0])
    if n < 2:
        return []

    total_pairs = n * (n - 1) // 2 if undirected else n * (n - 1)
    cap = num_samples if (num_samples is not None and num_samples > 0) else None
    # 大图保护：未限制且对数过大时，自动设上限，避免内存/时间爆炸
    if cap is None and total_pairs > 50000:
        cap = 20000

    if sample_strategy == "distance_gap" and cap is not None and cap < total_pairs:
        samples = _sample_distance_gap_pairs(
            graph_info=graph_info,
            cap=cap,
            weighted=weighted,
            seed=seed,
            undirected=undirected,
        )
        if cache_path:
            _save_samples_csv(cache_path, samples)
            print(f"[distance] cached samples -> {cache_path}")
        return samples
    elif sample_strategy == "oracle_mix" and cap is not None and cap < total_pairs:
        pair_list = _sample_oracle_mix_pairs(n, cap, leaf_of=leaf_of, seed=seed, undirected=undirected)
    elif sample_strategy == "random" and (cap is None or cap >= total_pairs):
        pair_list = []
        for s in range(n):
            for t in range(n):
                if s == t or (undirected and s > t):
                    continue
                pair_list.append((s, t))
    elif sample_strategy == "random":
        seen = set()
        rng = random.Random(seed)
        _add_random_pairs(seen, n, cap, rng, undirected=undirected, max_trials=20 * cap)
        pair_list = list(seen)
    else:
        raise ValueError(f"unknown sample_strategy: {sample_strategy}")

    by_src = defaultdict(list)
    for s, t in pair_list:
        by_src[s].append(t)

    # 每个不同源点一次 Dijkstra（scipy C 分块 / Python 回退），读出到各目标的距离
    eu, ev, ew = graph_info[3][0], graph_info[3][1], graph_info[4]
    samples = []
    for s, dist in iter_source_distances(n, eu, ev, ew, list(by_src.keys()), weighted=weighted):
        for t in by_src[s]:
            d = dist[t]
            if d != float("inf"):
                samples.append({"s": s, "t": t, "distance": float(d)})

    print(
        f"[distance] sampled {len(samples)} unique reachable pairs "
        f"(requested cap={cap}, total possible={total_pairs}, "
        f"dijkstra_runs={len(by_src)}, undirected={undirected})"
    )
    if cache_path:
        _save_samples_csv(cache_path, samples)
        print(f"[distance] cached samples -> {cache_path}")
    return samples


def load_label_pairs_csv(path, value_col="distance", num_nodes=None):
    """读取外部标签 CSV，构造训练用样本列表。

    用途：当监督标签不是「网格图 Dijkstra」而是**外部算好的真值**（例如 pygeodesic 的
    精确曲面测地距离 exact_geo）时，直接用这份 CSV 当样本集，跳过内置采样与 Dijkstra。

    Args:
        path: CSV 路径，需含列 s, t 以及 value_col
        value_col: 作为监督距离的列名（默认 distance；精确测地结果里是 exact_geo）

    Returns:
        list[dict]: [{"s": int, "t": int, "distance": float}, ...]
    """
    import csv as _csv
    samples = []
    seen = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        required = {"s", "t", value_col}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"标签文件缺少必要列 {required}: {path}")
        for line, row in enumerate(reader, 2):
            try:
                s, t, d = int(row["s"]), int(row["t"]), float(row[value_col])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"标签第 {line} 行格式错误: {path}") from exc
            if min(s, t) < 0 or (num_nodes is not None and max(s, t) >= num_nodes):
                raise ValueError(f"标签第 {line} 行顶点编号越界: {(s, t)}")
            if not math.isfinite(d) or d < 0 or (s != t and d == 0):
                raise ValueError(f"标签第 {line} 行距离必须有限且非自身点对为正: {d}")
            if s == t:
                continue
            pair = (min(s, t), max(s, t))
            if pair in seen:
                if seen[pair] != d:
                    raise ValueError(f"标签第 {line} 行同一点对存在冲突距离: {pair}")
                continue
            seen[pair] = d
            samples.append({"s": pair[0], "t": pair[1], "distance": d})
    if not samples:
        raise ValueError(f"标签文件没有有效的非自身点对: {path}")
    return samples


def split_distance_dataset(sample_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    """对**唯一**样本对做无重叠切分；输入已去重，故 train/val/test 之间不会泄漏同一对。"""
    random.seed(seed)
    data = deepcopy(sample_list)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


def _collect_hop_subgraph(center, neighbors, max_hops=2):
    visited = set([center])
    frontier = [center]
    for _ in range(max_hops):
        next_frontier = []
        for u in frontier:
            for v in neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        if len(frontier) == 0:
            break
    node_list = sorted(list(visited))
    return node_list


def _reindex_edges_from_nodes(edge_u, edge_v, node_list):
    id_map = {nid: i for i, nid in enumerate(node_list)}
    new_u = []
    new_v = []
    for u, v in zip(edge_u, edge_v):
        if u in id_map and v in id_map:
            new_u.append(id_map[u])
            new_v.append(id_map[v])
    if len(new_u) == 0:
        for i in range(len(node_list)):
            new_u.append(i)
            new_v.append(i)
    return [new_u, new_v], id_map


def _dijkstra_multi_source(adj, sources):
    """从每个 source 单独跑 Dijkstra，返回 {source: dist_list}。"""
    return {src: _dijkstra_single_source(adj, src) for src in sources}


class FeatureBuilder:
    """
    统一的节点特征构造器：以 [label_norm, degree_norm, x_norm, y_norm] 为基础特征，
    重复填充到 feat_dim。坐标缺失的节点回退到坐标几何中心，保证特征维度一致。
    """

    def __init__(self, graph_info, node_coords, feat_dim):
        self.labels = graph_info[1]
        self.degree = graph_info[2]
        self.feat_dim = feat_dim
        self.node_coords = node_coords
        self.max_label = max(1, max(self.labels)) if self.labels else 1
        self.max_degree = max(1, max(self.degree)) if self.degree else 1
        if node_coords:
            xs = [c[0] for c in node_coords.values()]
            ys = [c[1] for c in node_coords.values()]
            self.x_min, self.x_max = min(xs), max(xs)
            self.y_min, self.y_max = min(ys), max(ys)
        else:
            self.x_min, self.x_max = 0.0, 1.0
            self.y_min, self.y_max = 0.0, 1.0
        self.x_range = max(1e-6, self.x_max - self.x_min)
        self.y_range = max(1e-6, self.y_max - self.y_min)
        self.x_center = 0.5 * (self.x_min + self.x_max)
        self.y_center = 0.5 * (self.y_min + self.y_max)

    def normalized_coord(self, nid):
        x, y = self.node_coords.get(nid, (self.x_center, self.y_center))
        fx = (x - self.x_min) / self.x_range
        fy = (y - self.y_min) / self.y_range
        return fx, fy

    def node_row(self, nid):
        f_label = float(self.labels[nid]) / float(self.max_label)
        f_deg = float(self.degree[nid]) / float(self.max_degree)
        fx, fy = self.normalized_coord(nid)
        base = [f_label, f_deg, fx, fy]
        repeat_n = math.ceil(self.feat_dim / len(base))
        return (base * repeat_n)[: self.feat_dim]

    def features(self, node_list):
        return [self.node_row(nid) for nid in node_list]


def _nearest_k_local_by_access(access_row, k):
    """
    给定某个查询节点到所有高速节点（local 索引）的图最短路距离行，
    返回距离最近的 k 个高速节点 local 索引（按真实图距离，而非 node id）。
    """
    order = sorted(range(len(access_row)), key=lambda j: access_row[j])
    picked = [j for j in order if access_row[j] != float("inf")][: max(1, k)]
    if not picked:
        picked = [order[0]]
    return picked


def precompute_nearest_k(access_dist, k_max=16):
    """
    一次性对**所有节点**预计算最近的 k_max 个高速入口（local 索引，按距离近→远排好）。

    训练/推理时每个样本只需查表 + O(k) 过滤，替代原来每样本每轮的 O(K·logK) 全排序。
    存 k_max（≥ 任何合理的 highway_k）与 highway_k 解耦；运行时按需切片 [:k]。

    Args:
        access_dist (np.ndarray | list): [N, K] 每个节点到各高速入口的图最短路（float32）。
        k_max (int): 每个节点预存的最近入口数上限。

    Returns:
        np.ndarray[int32]: [N, kk] 最近入口的 local 索引（近→远），kk = min(k_max, K)。
    """
    arr = np.asarray(access_dist, dtype=np.float32)
    n, K = arr.shape
    kk = min(max(1, k_max), K)
    if kk >= K:
        part = np.tile(np.arange(K), (n, 1))
    else:
        part = np.argpartition(arr, kk - 1, axis=1)[:, :kk]  # 每行最近 kk 个(未排序)
    rows = np.arange(n)[:, None]
    order = np.argsort(arr[rows, part], axis=1)              # 在这 kk 个里按距离排序
    return part[rows, order].astype(np.int32)                # [N, kk] 近→远


def build_highway_context(
    graph_info,
    coords,
    highway_global_ids,
    local_edges,
    access_dist,
    highway_pair_dist,
    leaf_of=None,
    feature_dim=64,
    device='cpu',
):
    """
    根据「全局图 + 节点坐标 + 已派生的高速(边界)节点/边/距离」组装 DistancePredictor 所需的
    高速上下文（含特征张量）。

    距离预计算（access_dist / highway_pair_dist）由 build_highway.py 在派生四叉树分区时一并完成
    并传入，这里只负责张量化与特征构造，因此本函数是 .off → 四叉树分区流水线的最后一步。

    Args:
        coords (dict[int, tuple[float,float]]): 全图每个节点的 (x, y) 坐标（来自 .off）。
        highway_global_ids (list[int]): 高速(边界)节点的全局 id（已排序）。
        local_edges (tuple[list[int], list[int]]): (local_u, local_v) 高速图内部边（local 索引）。
        access_dist (list[list[float]]): [N_full][K] 每个图节点到每个高速入口的图最短路。
        highway_pair_dist (list[list[float]]): [K][K] 高速图内部两两最短路。
    """
    global_to_local = {g: i for i, g in enumerate(highway_global_ids)}
    feature_builder = FeatureBuilder(graph_info, coords, feature_dim)
    x_highway = torch.tensor(
        feature_builder.features(highway_global_ids), dtype=torch.float, device=device
    )
    local_u, local_v = local_edges
    if len(local_u) == 0:
        # 退化情形：没有内部边时退回自环，保证 InterGNN 可运行
        local_u = list(range(len(highway_global_ids)))
        local_v = list(range(len(highway_global_ids)))
    edge_index_highway = torch.tensor([local_u, local_v], dtype=torch.long, device=device)

    # 叶子盒成员：用于 Inner-GNN 吃"四叉树分区子图"而非 2-hop ego 子图（对齐论文 G1~G4）
    leaf_members = defaultdict(list)
    if leaf_of:
        for nid, lid in leaf_of.items():
            leaf_members[lid].append(nid)
        for lid in list(leaf_members.keys()):
            leaf_members[lid].sort()

    return {
        "x_highway": x_highway,
        "edge_index_highway": edge_index_highway,
        "highway_global_ids": highway_global_ids,
        "global_to_local": global_to_local,
        "access_dist": access_dist,
        "highway_pair_dist": highway_pair_dist,
        "node_coords": coords,
        "feature_builder": feature_builder,
        "leaf_of": leaf_of,
        "leaf_members": leaf_members,
        "inner_cache": {},  # cell_id -> (x, edge_index, id_map)，跨样本/轮次复用分区子图张量
    }


def _build_inner_subgraph(node, context, graph_info, feature_dim, device, inner_mode="partition"):
    """
    构造单个查询点的 Inner-GNN 输入子图。

    - inner_mode="partition"（默认，对齐论文 G1~G4）：用 node 所在**四叉树叶子盒**的诱导子图；
      同一盒子的子图张量缓存复用（很多 (s,t) 对共享同一盒子，避免重复构建）。
    - inner_mode="ego"：退回以 node 为中心的 2-hop ego 子图（用于消融对比）。

    Returns:
        (x, edge_index, idx_tensor)：节点特征、子图边、node 在子图内的局部索引。
    """
    edge_u = graph_info[3][0]
    edge_v = graph_info[3][1]
    neighbors = graph_info[5]
    feature_builder = context["feature_builder"]
    leaf_of = context.get("leaf_of")

    if inner_mode == "partition" and leaf_of is not None and node in leaf_of:
        cell = leaf_of[node]
        cache = context["inner_cache"]
        if cell not in cache:
            node_list = context["leaf_members"][cell]
            edges, id_map = _reindex_edges_from_nodes(edge_u, edge_v, node_list)
            x = torch.tensor(feature_builder.features(node_list), dtype=torch.float, device=device)
            edge_index = torch.tensor(edges, dtype=torch.long, device=device)
            cache[cell] = (x, edge_index, id_map)
        x, edge_index, id_map = cache[cell]
        return x, edge_index, torch.tensor(id_map[node], dtype=torch.long, device=device)

    # ego 回退
    node_list = _collect_hop_subgraph(node, neighbors, max_hops=2)
    edges, id_map = _reindex_edges_from_nodes(edge_u, edge_v, node_list)
    x = torch.tensor(feature_builder.features(node_list), dtype=torch.float, device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    return x, edge_index, torch.tensor(id_map[node], dtype=torch.long, device=device)


def build_synthetic_partition_inputs(
    graph_info,
    sample,
    highway_ratio=0.15,
    k_highway=3,
    feature_dim=64,
    device='cpu',
    external_highway_context=None,
    inner_mode="partition",
):
    """
    将单个 (s,t) 样本转为 DistancePredictor 所需的输入。

    相比旧版本的改动：
      1. 节点特征与全局特征均使用真实坐标（来自 external_highway_context 的 feature_builder）。
      2. s/t 的高速连接点按“全图最短路”选最近的 k 个（而非 node id 差值近似）。
      3. 额外返回 highway 分解距离特征 highway_dist_feat：
         [d(s->入口), d_highway(入口s->入口t), d(t->入口), 三者之和]，经 log1p 压缩。
    """
    node_ids = graph_info[0]
    labels = graph_info[1]
    degree = graph_info[2]
    edge_u = graph_info[3][0]
    edge_v = graph_info[3][1]
    neighbors = graph_info[5]
    feat_dim = feature_dim

    s = int(sample["s"])
    t = int(sample["t"])

    if external_highway_context is None:
        raise ValueError(
            "external_highway_context is required. "
            "This pipeline is configured for external highway network only."
        )

    feature_builder = external_highway_context["feature_builder"]
    highway_nodes = external_highway_context["highway_global_ids"]
    x_highway = external_highway_context["x_highway"].to(device)
    edge_index_highway = external_highway_context["edge_index_highway"].to(device)
    access_dist = external_highway_context["access_dist"]
    highway_pair_dist = external_highway_context["highway_pair_dist"]

    # Inner-GNN 输入：四叉树分区(叶子盒)子图（默认）或 2-hop ego 子图（inner_mode="ego"）
    x_s, edge_index_s, s_idx = _build_inner_subgraph(
        s, external_highway_context, graph_info, feat_dim, device, inner_mode
    )
    x_t, edge_index_t, t_idx = _build_inner_subgraph(
        t, external_highway_context, graph_info, feat_dim, device, inner_mode
    )

    # 按真实图最短路选最近的 k 个高速入口（local 索引）
    # 优先用预计算表（precompute_nearest_k）O(k) 查表；无表则回退旧的 O(K·logK) 排序（如 infer_distance）。
    nearest_k = external_highway_context.get("nearest_k_local")
    if nearest_k is not None:
        def _pick(node):
            row = access_dist[node]
            cand = [int(j) for j in nearest_k[node] if row[j] != float("inf")][: max(1, k_highway)]
            return cand if cand else [int(nearest_k[node][0])]
        s_local = _pick(s)
        t_local = _pick(t)
    else:
        s_local = _nearest_k_local_by_access(access_dist[s], k_highway)
        t_local = _nearest_k_local_by_access(access_dist[t], k_highway)
    s_connect_idx = torch.tensor(s_local, dtype=torch.long, device=device)
    t_connect_idx = torch.tensor(t_local, dtype=torch.long, device=device)

    # 全局特征使用真实归一化坐标，让 InterGNN 能定位 s/t
    s_fx, s_fy = feature_builder.normalized_coord(s)
    t_fx, t_fy = feature_builder.normalized_coord(t)
    s_global_feat = torch.tensor([s_fx, s_fy], dtype=torch.float, device=device)
    t_global_feat = torch.tensor([t_fx, t_fy], dtype=torch.float, device=device)

    # highway 分解距离特征：access(s) + highway(入口s, 入口t) + access(t)
    s_entry = s_local[0]
    t_entry = t_local[0]
    access_s = access_dist[s][s_entry]
    access_t = access_dist[t][t_entry]
    seg = highway_pair_dist[s_entry][t_entry]
    big = 1e6
    access_s = big if access_s == float("inf") else access_s
    access_t = big if access_t == float("inf") else access_t
    seg = big if seg == float("inf") else seg
    est = access_s + seg + access_t
    highway_dist_feat = torch.log1p(
        torch.tensor([access_s, seg, access_t, est], dtype=torch.float, device=device)
    )
    node_coords3d = external_highway_context.get("node_coords3d") or external_highway_context.get("node_coords", {})
    s_coord = node_coords3d.get(s)
    t_coord = node_coords3d.get(t)
    if s_coord is None or t_coord is None:
        euclidean_dist = 0.0
    else:
        sx, sy = float(s_coord[0]), float(s_coord[1])
        tx, ty = float(t_coord[0]), float(t_coord[1])
        sz = float(s_coord[2]) if len(s_coord) > 2 else 0.0
        tz = float(t_coord[2]) if len(t_coord) > 2 else 0.0
        euclidean_dist = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + (sz - tz) ** 2)
    euclidean_dist_feat = torch.log1p(
        torch.tensor([euclidean_dist], dtype=torch.float, device=device)
    )

    return {
        "x_s": x_s,
        "edge_index_s": edge_index_s,
        "s_idx": s_idx,
        "x_t": x_t,
        "edge_index_t": edge_index_t,
        "t_idx": t_idx,
        "x_highway": x_highway,
        "edge_index_highway": edge_index_highway,
        "s_global_feat": s_global_feat,
        "t_global_feat": t_global_feat,
        "s_connect_idx": s_connect_idx,
        "t_connect_idx": t_connect_idx,
        "highway_dist_feat": highway_dist_feat,
        "euclidean_dist_feat": euclidean_dist_feat,
    }


# ---------------------------------------------------------------------------
# 消融实验支持：不使用地形分区 / highway 网络时，「单 GNN 全图」分支所需的张量构造
# ---------------------------------------------------------------------------
def build_full_graph_tensors(graph_info, coords, feature_dim=64, device="cpu"):
    """把整张地形网格图一次性张量化，供"单 GNN"消融分支使用。

    节点特征与 FeatureBuilder.node_row 完全同口径：
        base = [label/max_label, degree/max_degree, x_norm, y_norm]，再重复填充到 feature_dim。
    这样消融分支与三段式主方法的**输入特征完全一致**，差异只来自"结构"（有无分区/highway）。

    Returns:
        (x, edge_index):
            x (Tensor): [N, feature_dim]
            edge_index (LongTensor): [2, 2E]，无向边双向各存一条（SAGEConv 需要）
    """
    n = len(graph_info[0])
    labels = np.asarray(graph_info[1], dtype=np.float32)
    degree = np.asarray(graph_info[2], dtype=np.float32)
    max_label = float(max(1.0, labels.max() if n else 1.0))
    max_degree = float(max(1.0, degree.max() if n else 1.0))

    xs = np.array([coords[i][0] for i in range(n)], dtype=np.float32) if n else np.zeros(0, np.float32)
    ys = np.array([coords[i][1] for i in range(n)], dtype=np.float32) if n else np.zeros(0, np.float32)
    x_min, x_max = (float(xs.min()), float(xs.max())) if n else (0.0, 1.0)
    y_min, y_max = (float(ys.min()), float(ys.max())) if n else (0.0, 1.0)
    x_range = max(1e-6, x_max - x_min)
    y_range = max(1e-6, y_max - y_min)

    base = np.stack(
        [labels / max_label, degree / max_degree, (xs - x_min) / x_range, (ys - y_min) / y_range],
        axis=1,
    )  # [N, 4]
    repeat_n = int(math.ceil(feature_dim / base.shape[1]))
    x_np = np.tile(base, (1, repeat_n))[:, :feature_dim]
    x = torch.tensor(np.ascontiguousarray(x_np), dtype=torch.float, device=device)

    eu, ev = graph_info[3][0], graph_info[3][1]
    edge_index = torch.tensor(
        [np.asarray(eu, dtype=np.int64), np.asarray(ev, dtype=np.int64)],
        dtype=torch.long, device=device,
    )
    return x, edge_index


def build_pair_euclidean_features(samples, vertices3d, device="cpu"):
    """为样本列表构造 log1p(3D 欧氏直线距离) 特征，与 build_synthetic_partition_inputs 同口径。

    Returns:
        Tensor [len(samples)]：log1p(||v_s - v_t||_2)
    """
    vals = np.empty(len(samples), dtype=np.float32)
    for i, smp in enumerate(samples):
        s = int(smp["s"]); t = int(smp["t"])
        sx, sy, sz = vertices3d[s][0], vertices3d[s][1], vertices3d[s][2]
        tx, ty, tz = vertices3d[t][0], vertices3d[t][1], vertices3d[t][2]
        vals[i] = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + (sz - tz) ** 2)
    return torch.log1p(torch.tensor(vals, dtype=torch.float, device=device))



def build_point_feature_tensors(graph_info, vertices3d, device="cpu", with_normals=False):
    """给 baseline（GeGnn / NeuroGF / LiteGE）准备的「点特征」张量。

    与 build_full_graph_tensors 的区别：这里返回的是**真实 3D 几何**（坐标，可选法向），
    而不是 [label, degree, x, y] 那种为分区模型设计的统计特征——因为三篇 baseline
    的输入本来就是点坐标（NeuroGF/LiteGE）或 坐标+法向（GeGnn 的 6 通道输入）。

    Returns:
        (x, edge_index, edge_len, pos)
          x          [N, 3] 或 [N, 6]（with_normals=True 时拼接单位法向）
          edge_index [2, 2E] 无向边双向各一条
          edge_len   [2E]     每条边的 3D 欧氏长度
          pos        [N, 3]   归一化到单位盒的坐标（GeoConv 计算相对位置用）
    """
    n = len(graph_info[0])
    V = np.asarray([[vertices3d[i][0], vertices3d[i][1], vertices3d[i][2]] for i in range(n)],
                   dtype=np.float64) if n else np.zeros((0, 3))
    mins, maxs = V.min(axis=0), V.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    pos_np = ((V - mins) / span).astype(np.float32)   # 归一化到 [0,1]^3，数值稳定

    feats = [pos_np]
    if with_normals:
        normals = _vertex_normals(V, graph_info)
        feats.append(normals.astype(np.float32))
    x = torch.tensor(np.ascontiguousarray(np.concatenate(feats, axis=1)),
                     dtype=torch.float, device=device)

    eu, ev = graph_info[3][0], graph_info[3][1]
    edge_index = torch.tensor([np.asarray(eu, dtype=np.int64), np.asarray(ev, dtype=np.int64)],
                              dtype=torch.long, device=device)
    eu_np = np.asarray(eu, dtype=np.int64); ev_np = np.asarray(ev, dtype=np.int64)
    edge_len = torch.tensor(np.linalg.norm(V[eu_np] - V[ev_np], axis=1).astype(np.float32),
                            dtype=torch.float, device=device)
    pos = torch.tensor(pos_np, dtype=torch.float, device=device)
    return x, edge_index, edge_len, pos


def _vertex_normals(V, graph_info):
    """面法向按面积加权累加到顶点，再单位化（GeGnn 的输入是 坐标+法向 6 通道）。"""
    n = len(V)
    acc = np.zeros((n, 3), dtype=np.float64)
    eu = np.asarray(graph_info[3][0], dtype=np.int64)
    ev = np.asarray(graph_info[3][1], dtype=np.int64)
    # 网格边是无向且双向存储的，这里用 (u, v) 与共享邻居重建三角形代价高；
    # 改用更稳的近似：用 1-ring 邻域点做 PCA，最小特征向量即法向。
    nbr = graph_info[5]
    for i in range(n):
        ring = nbr[i]
        if len(ring) < 3:
            continue
        P = V[list(ring)] - V[i]
        C = P.T @ P
        w, vec = np.linalg.eigh(C)
        nrm = vec[:, 0]
        acc[i] = nrm
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.maximum(norm, 1e-9)
