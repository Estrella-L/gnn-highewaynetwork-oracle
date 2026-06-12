import math
import time
import random
import heapq
import os
import torch
from graph_operation import graph_depth
from copy import deepcopy
from collections import defaultdict


class SampleSubgraph:
    def __init__(self, query, data_graph):
        # data information contains:
        # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
        self.query = query
        self.data_graph = data_graph

    def find_subgraph(self, start_query_vertex, candidates):
        output_vertices = list()
        output_v_label = list()
        output_degree = list()
        output_edges = list()
        output_edge_label = list()
        output_v_neigh = list()
        depth = graph_depth(self.query, start_query_vertex)
        candidate_u = candidates[start_query_vertex]
        all_candidates = list()
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                all_candidates.append(candidates[i][j])
        all_candidates = list(set(all_candidates))
        # print(all_candidates)
        data_label = self.data_graph[1]
        data_neigh = self.data_graph[5]
        # two possible ways: 1. start from all candidates and perform BFS search
        # 2. when the candidate is visited, we don't do the search starting from that node.
        all_need_visited = deepcopy(candidate_u)
        while len(all_need_visited) > 0:
            search_depth = 0
            queue = list()
            depth_queue = list()
            new_graph_vertices = list()
            new_graph_v_label = dict()
            new_graph_v_degree = defaultdict(lambda : 0)
            new_e_u = list()
            new_e_v = list()
            new_edge_label = list()
            new_graph_v_neigh = defaultdict(list)
            start_data_vertex = all_need_visited.pop(0)
            queue.append(start_data_vertex)
            depth_queue.append(search_depth)
            new_graph_vertices.append(start_data_vertex)
            new_graph_v_label[start_data_vertex] = data_label[start_data_vertex]
            while len(queue)>0:
                current_data_vertex = queue.pop(0)
                search_depth = depth_queue.pop(0)
                if search_depth > depth:
                    break
                for v in data_neigh[current_data_vertex]:
                    if v in all_need_visited:
                        all_need_visited.remove(v)
                    if v not in new_graph_vertices and v in all_candidates:
                        new_graph_vertices.append(v)
                        new_graph_v_label[v] = data_label[v]
                        queue.append(v)
                        depth_queue.append(search_depth+1)
                        for neigh_v in data_neigh[v]:
                            if neigh_v in new_graph_vertices:
                                # two way (undirected) edges
                                new_e_u.append(v)
                                new_e_v.append(neigh_v)
                                new_e_u.append(neigh_v)
                                new_e_v.append(v)
                                new_graph_v_degree[v] += 1
                                new_graph_v_degree[neigh_v] += 1

                                # neighbor should be added only once.
                                # new_graph_v_neigh[v].append(neigh_v)
                                new_graph_v_neigh[neigh_v].append(v)

                                new_edge_label.append(1)
                                new_edge_label.append(1)
            new_graph_edges = [new_e_u, new_e_v]
            output_vertices.append(new_graph_vertices)
            output_v_label.append(deepcopy(new_graph_v_label))
            output_degree.append(deepcopy(new_graph_v_degree))
            output_edges.append(new_graph_edges)
            output_edge_label.append(new_edge_label)
            output_v_neigh.append(deepcopy(new_graph_v_neigh))
        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def find_subgraph_induced(self, candidates):
        t_0 = time.time()
        all_candidates = list()
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                all_candidates.append(candidates[i][j])
        all_candidates = list(set(all_candidates))
        all_need_visited = deepcopy(all_candidates)
        queue = list()
        depth_queue = list()
        new_graph_vertices = list()
        new_graph_v_label = dict()
        new_graph_v_degree = defaultdict(lambda : 0)
        new_e_u = list()
        new_e_v = list()
        new_edge_label = list()
        new_graph_v_neigh = defaultdict(list)

        # get data graph information
        data_label = self.data_graph[1]
        data_edge = self.data_graph[3]
        data_neigh = self.data_graph[5]
        
        new_graph_vertices = deepcopy(all_candidates)
        for v in new_graph_vertices:
            new_graph_v_label[v] = data_label[v]
        t_1 = time.time()
        print('sample satage 1: {}s'.format(t_1-t_0))
        # for i in range(len(data_edge[0])):
        #     # if two nodes are both in candidate set, the edge is included for new graph
        #     u = data_edge[0][i]
        #     v = data_edge[1][i]
        #     if u in all_candidates and v in all_candidates:
        #         new_e_u.append(u)
        #         new_e_v.append(v)
        #         # only add once, since the edge will appear twice.
        #         new_graph_v_degree[u] += 1
        #         new_graph_v_neigh[u].append(v)
        #         new_edge_label.append(1)

        for vertex in new_graph_vertices:
            # if two nodes are both in candidate set, the edge is included for new graph
            neigh_of_v = data_neigh[vertex]
            for u in neigh_of_v:
                if u in all_candidates:
                    new_e_u.append(u)
                    new_e_v.append(vertex)
                    # only add once, since the edge will appear twice.
                    new_graph_v_degree[vertex] += 1
                    new_graph_v_neigh[vertex].append(u)
                    new_edge_label.append(1)

        t_2 = time.time()
        print('sample stage 2: {}s'.format(t_2-t_1))
        new_edges = [deepcopy(new_e_u), deepcopy(new_e_v)]
        new_vertices = new_graph_vertices
        new_v_label = new_graph_v_label
        new_degree = deepcopy(new_graph_v_degree)
        new_edge_label = deepcopy(new_edge_label)
        new_v_neigh = new_graph_v_neigh

        check_info = [new_vertices, new_v_label, new_degree, new_edges, new_edge_label, new_v_neigh]
        output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh = self._split_graph(check_info)
        t_3 = time.time()
        print('sample stage 3: {}s'.format(t_3-t_2))
        # output_graph_info = [output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh]

        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def load_induced_subgraph(self, candidates, induced_subgraph_list, neighbor_offset):
        queue = list()
        depth_queue = list()
        new_graph_vertices = list()
        new_graph_v_label = dict()
        new_graph_v_degree = defaultdict(lambda : 0)
        new_e_u = list()
        new_e_v = list()
        new_edge_label = list()
        new_graph_v_neigh = defaultdict(list)

        # get data graph information
        data_label = self.data_graph[1]
        data_edge = self.data_graph[3]
        data_neigh = self.data_graph[5]

        new_graph_vertices = deepcopy(candidates)
        for v in new_graph_vertices:
            new_graph_v_label[v] = data_label[v]

        for i in range(len(candidates)):
            vertex = candidates[i]
            strat_index = neighbor_offset[i]
            end_index = neighbor_offset[i+1]
            for j in range(strat_index, end_index):
                u = induced_subgraph_list[j]
                new_e_u.append(u)
                new_e_v.append(vertex)
                # only add once, since the edge will appear twice.
                new_graph_v_degree[vertex] += 1
                new_graph_v_neigh[vertex].append(u)
                new_edge_label.append(1)
        
        new_edges = [deepcopy(new_e_u), deepcopy(new_e_v)]
        new_vertices = new_graph_vertices
        new_v_label = new_graph_v_label
        new_degree = deepcopy(new_graph_v_degree)
        new_edge_label = deepcopy(new_edge_label)
        new_v_neigh = new_graph_v_neigh

        check_info = [new_vertices, new_v_label, new_degree, new_edges, new_edge_label, new_v_neigh]
        output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh = self._split_graph(check_info)

        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def _split_graph(self, graph_info):
        vertices_id = graph_info[0]
        vertices_label = graph_info[1]
        vertices_neighbor = graph_info[5]
        num_vertices = len(vertices_id)
        to_be_visited = deepcopy(vertices_id)
        

        # initialize the output lists
        output_vertices = list()
        output_v_label = list()
        output_v_degree = list()
        output_edges = list()
        output_e_label = list()
        output_v_neigh = list()

        while len(to_be_visited) > 0:
            # initialize the temp containers.
            out_temp_vertices = list()
            out_temp_v_label = dict()
            out_temp_v_degree = defaultdict(lambda: 0)
            out_temp_e_u = list()
            out_temp_e_v = list()
            out_temp_e_label = list()
            out_temp_v_neigh = defaultdict(list)

            start_node = to_be_visited[0]
            # to_be_visited.remove(start_node)
            queue = list()
            queue.append(start_node)

            while len(queue) > 0:
                current_node = queue.pop(0)
                current_neighbors = vertices_neighbor[current_node]
                try:
                    to_be_visited.remove(current_node)
                except ValueError:
                    # print('node {} has been removed'.format(current_node))
                    continue   # if there is no this node in the to be visited set, we donot need to compute it again. will lead to bugs.  
                out_temp_vertices.append(current_node)        
                out_temp_v_label[current_node] = vertices_label[current_node]         
                for v in current_neighbors:
                    # a BFS, do we need to check whether it is in the to be visited set? (in Queue it should!)
                    out_temp_e_u.append(current_node)
                    out_temp_e_v.append(v)       # add a one-way edge, it will be added again.
                    out_temp_e_label.append(1)   # edge label is always 1.
                    out_temp_v_degree[current_node] += 1
                    out_temp_v_neigh[current_node].append(v)
                    if v in to_be_visited:
                        queue.append(v)

            output_vertices.append(deepcopy(out_temp_vertices))
            output_v_label.append(deepcopy(out_temp_v_label))
            output_v_degree.append(deepcopy(out_temp_v_degree))
            output_edges.append([deepcopy(out_temp_e_u), deepcopy(out_temp_e_v)])
            output_e_label.append(deepcopy(out_temp_e_label))
            output_v_neigh.append(deepcopy(out_temp_v_neigh))
        
        return output_vertices, output_v_label, output_v_degree, output_edges, output_e_label, output_v_neigh

    def update_query(self, query):
        self.query = query


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


def build_distance_samples(graph_info, num_samples=None, weighted=True, seed=42, undirected=True):
    """
    构造节点对最短路监督样本（唯一、无重复、无泄漏），并对大图高效。

    做法：
      1. 先确定要用的**唯一**节点对集合：num_samples 给定且小于全部对数时，**无放回**随机抽样
         num_samples 个唯一对；否则枚举全部对（仅适合小图）。
      2. 按源点分组，每个不同源点只跑**一次** Dijkstra，读出该源到其目标的距离。
         这样 Dijkstra 次数 = 不同源点数 ≤ num_samples，避免对大网格做全 APSP。

    Args:
        num_samples (int | None): 采样上限；None 或 <=0 时尝试全部对（大图会自动设上限保护）。
        undirected (bool): True 时只取 s < t 的无向对（无向网格距离对称）。

    Returns:
        list[dict]: [{"s": int, "t": int, "distance": float}, ...]（每对唯一、无跨集泄漏）
    """
    random.seed(seed)
    n = len(graph_info[0])
    if n < 2:
        return []
    adj = _build_weighted_adj_list(graph_info, weighted=weighted)

    total_pairs = n * (n - 1) // 2 if undirected else n * (n - 1)
    cap = num_samples if (num_samples is not None and num_samples > 0) else None
    # 大图保护：未限制且对数过大时，自动设上限，避免内存/时间爆炸
    if cap is None and total_pairs > 50000:
        cap = 20000

    if cap is None or cap >= total_pairs:
        pair_list = []
        for s in range(n):
            for t in range(n):
                if s == t or (undirected and s > t):
                    continue
                pair_list.append((s, t))
    else:
        seen = set()
        max_trials = 20 * cap
        tries = 0
        while len(seen) < cap and tries < max_trials:
            tries += 1
            s = random.randrange(n)
            t = random.randrange(n)
            if s == t:
                continue
            if undirected and s > t:
                s, t = t, s
            seen.add((s, t))
        pair_list = list(seen)

    by_src = defaultdict(list)
    for s, t in pair_list:
        by_src[s].append(t)

    samples = []
    for s, targets in by_src.items():
        dist = _dijkstra_single_source(adj, s)
        for t in targets:
            if dist[t] != float("inf"):
                samples.append({"s": s, "t": t, "distance": float(dist[t])})

    print(
        f"[distance] sampled {len(samples)} unique reachable pairs "
        f"(requested cap={cap}, total possible={total_pairs}, "
        f"dijkstra_runs={len(by_src)}, undirected={undirected})"
    )
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
    }
