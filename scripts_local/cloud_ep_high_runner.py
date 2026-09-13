# -*- coding: utf-8 -*-
"""EP_high 云端端到端实验：并行预处理 + 并行采样 + 网格训练（GPU）。"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from build_highway import load_off, build_mesh_graph, build_quadtree, find_boundary_nodes, _adj_from_graph_info

def _lazy_imports():
    """仅 grid 阶段需要 torch/torch_geometric，延迟导入以便纯 CPU 预处理环境可运行。"""
    from preprocess import build_highway_context, split_distance_dataset
    from main import build_loss, run_distance_epoch, evaluate_distance_groups
    from model import DistanceRegressionNet
    return (build_highway_context, split_distance_dataset, build_loss,
            run_distance_epoch, evaluate_distance_groups, DistanceRegressionNet)


def _canonical_pair_local(s, t, undirected=True):
    return (s, t) if (not undirected or s <= t) else (t, s)


def _add_random_pairs_local(seen, n, target_count, rng, undirected=True, max_trials=None):
    max_trials = max_trials or max(1000, 50 * max(1, target_count))
    tries = 0
    while len(seen) < target_count and tries < max_trials:
        tries += 1
        s = rng.randrange(n)
        t = rng.randrange(n)
        if s == t:
            continue
        s, t = _canonical_pair_local(s, t, undirected)
        seen.add((s, t))
    return tries


def _sample_oracle_mix_pairs_local(n, cap, leaf_of, seed=42, undirected=True):
    """对齐 preprocess._sample_oracle_mix_pairs：约 1/3 同叶 / 1/3 跨叶 / 1/3 随机。"""
    rng = random.Random(seed)
    by_leaf = defaultdict(list)
    for node in range(n):
        if node in leaf_of:
            by_leaf[leaf_of[node]].append(node)
    leaves = [leaf for leaf, nodes in by_leaf.items() if len(nodes) > 0]
    inner_leaves = [leaf for leaf in leaves if len(by_leaf[leaf]) >= 2]
    if not leaves or not inner_leaves:
        raise ValueError("oracle_mix needs non-empty leaf with 2+ nodes")
    inner_target = cap // 3
    inter_target = cap // 3
    random_target = cap - inner_target - inter_target
    seen = set()
    inner_seen_target = len(seen) + inner_target
    inner_tries = 0
    while len(seen) < inner_seen_target and inner_tries < max(1000, 50 * max(1, inner_target)):
        inner_tries += 1
        leaf = rng.choice(inner_leaves)
        s, t = rng.sample(by_leaf[leaf], 2)
        s, t = _canonical_pair_local(s, t, undirected)
        seen.add((s, t))
    inter_seen_target = len(seen) + inter_target
    inter_tries = 0
    if len(leaves) >= 2:
        while len(seen) < inter_seen_target and inter_tries < max(1000, 50 * max(1, inter_target)):
            inter_tries += 1
            leaf_s, leaf_t = rng.sample(leaves, 2)
            s = rng.choice(by_leaf[leaf_s])
            t = rng.choice(by_leaf[leaf_t])
            s, t = _canonical_pair_local(s, t, undirected)
            seen.add((s, t))
    random_seen_target = len(seen) + random_target
    random_tries = 0
    while len(seen) < random_seen_target and random_tries < max(1000, 50 * max(1, random_target)):
        random_tries += 1
        s = rng.randrange(n)
        t = rng.randrange(n)
        if s == t:
            continue
        s, t = _canonical_pair_local(s, t, undirected)
        seen.add((s, t))
    if len(seen) < cap:
        raise ValueError("oracle_mix could not generate enough unique pairs")
    return [{"s": int(s), "t": int(t)} for s, t in sorted(seen)]


# ---------------------------------------------------------------------------
# 并行 Dijkstra 工作进程（fork 继承 CSR/分区，避免重复序列化大对象）
# ---------------------------------------------------------------------------
_W_CSR = None
_W_LEAF_OF = None
_W_CELLS = None


def _dijkstra_worker_init(n, eu, ev, ew, leaf_of, cell_members):
    global _W_CSR, _W_LEAF_OF, _W_CELLS
    import scipy.sparse as sp
    w = np.maximum(1e-9, np.asarray(ew, dtype=np.float64))
    _W_CSR = sp.csr_matrix(
        (w, (np.asarray(eu, dtype=np.int64), np.asarray(ev, dtype=np.int64))),
        shape=(n, n),
    )
    _W_LEAF_OF = leaf_of
    _W_CELLS = cell_members


def _preprocess_worker_task(task):
    start, src_list, chunk = task
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    col_parts = []
    tu_l, tv_l, tw_l = [], [], []
    for i in range(0, len(src_list), chunk):
        batch = src_list[i:i + chunk]
        dmat = sp_dijkstra(_W_CSR, directed=False, indices=batch)
        col_parts.append(np.ascontiguousarray(dmat, dtype=np.float32))
        for j, g in enumerate(batch):
            dl = dmat[j]
            for b in _W_CELLS[_W_LEAF_OF[int(g)]]:
                if b != int(g):
                    db = dl[b]
                    if db != float("inf"):
                        tu_l.append(int(g))
                        tv_l.append(int(b))
                        tw_l.append(float(db))
    return start, np.concatenate(col_parts, axis=0), tu_l, tv_l, tw_l


def parallel_preprocess(n, eu, ev, ew, boundary_sorted, leaf_of, cell_members, adj, workers=32, chunk=64, k_max=16):
    """并行计算每节点 top-k 高速入口（增量化，内存 O(N*k + workers*chunk*N)）；
    同时派生高速边（full 与 top-k8/top-k16）。"""
    import multiprocessing as mp
    K = len(boundary_sorted)
    k = min(max(1, k_max), K)
    cur_idx = np.full((n, k), -1, dtype=np.int32)
    cur_dist = np.full((n, k), np.inf, dtype=np.float32)
    ctx = mp.get_context("fork")
    tasks = []
    for i in range(0, K, chunk):
        tasks.append((i, boundary_sorted[i:i + chunk], chunk))
    transit_u_all, transit_v_all, transit_w_all = [], [], []
    cnt = 0
    with ctx.Pool(
        processes=min(workers, len(tasks)),
        initializer=_dijkstra_worker_init,
        initargs=(n, eu, ev, ew, leaf_of, cell_members),
    ) as pool:
        for start, cols, tu, tv, tw in pool.imap_unordered(_preprocess_worker_task, tasks, chunksize=1):
            G = cols.shape[0]
            cand = np.concatenate([cur_dist, cols.T], axis=1)  # [N, k+G]
            part = np.argpartition(cand, k - 1, axis=1)[:, :k]
            rows = np.arange(n)[:, None]
            sel = np.take_along_axis(part, np.argsort(np.take_along_axis(cand, part, 1), axis=1, kind="stable"), 1)
            sel_dist = np.take_along_axis(cand, sel, 1)
            is_cur = sel < k
            old_idx = np.take_along_axis(cur_idx, np.where(is_cur, sel, 0), 1)
            global_idx = np.where(is_cur, old_idx, (sel - k) + start).astype(np.int32)
            cur_idx = global_idx
            cur_dist = np.ascontiguousarray(sel_dist, dtype=np.float32)
            transit_u_all.extend(tu)
            transit_v_all.extend(tv)
            transit_w_all.extend(tw)
            cnt += G
            if cnt % (chunk * workers * 6) == 0:
                print("  [prep] merged", cnt, "/", K, flush=True)
    transit_u = np.asarray(transit_u_all, dtype=np.int64)
    transit_v = np.asarray(transit_v_all, dtype=np.int64)
    transit_w = np.asarray(transit_w_all, dtype=np.float32)

    bset = set(boundary_sorted)
    cross_u, cross_v, cross_w = [], [], []
    for u in boundary_sorted:
        for v, w in adj[u]:
            if v in bset:
                cross_u.append(u)
                cross_v.append(v)
                cross_w.append(w)
    cu = np.asarray(cross_u, dtype=np.int64)
    cv = np.asarray(cross_v, dtype=np.int64)
    cw = np.asarray(cross_w, dtype=np.float32)

    def dedupe_min(us, vs, ws):
        a = np.minimum(us, vs)
        b = np.maximum(us, vs)
        key = a.astype(np.int64) * np.int64(n) + b.astype(np.int64)
        order = np.lexsort((ws, key))
        ks = key[order]
        ws_s = ws[order]
        uni, first = np.unique(ks, return_index=True)
        return a[order[first]], b[order[first]], ws_s[first]

    fu, fv, fw = dedupe_min(
        np.concatenate([transit_u, cu]),
        np.concatenate([transit_v, cv]),
        np.concatenate([transit_w, cw]),
    )

    topk_edges = {}
    if len(transit_u):
        order = np.lexsort((transit_w, transit_u))
        us_t = transit_u[order]
        vs_t = transit_v[order]
        ws_t = transit_w[order]
        for kk in (8, 16):
            keep = np.zeros(len(us_t), dtype=bool)
            lo = 0
            while lo < len(us_t):
                hi = lo + 1
                while hi < len(us_t) and us_t[hi] == us_t[lo]:
                    hi += 1
                take = min(kk, hi - lo)
                keep[lo:lo + take] = True
                lo = hi
            mi = np.concatenate([us_t[keep], cu])
            mv = np.concatenate([vs_t[keep], cv])
            mw = np.concatenate([ws_t[keep], cw])
            ku, kv, kw = dedupe_min(mi, mv, mw)
            topk_edges[kk] = (ku, kv, kw)
    return cur_idx, cur_dist, (fu, fv, fw), topk_edges


class _RowProxy:
    __slots__ = ("_idx", "_dist")

    def __init__(self, idx, dist):
        self._idx = idx
        self._dist = dist

    def __getitem__(self, j):
        pos = None
        for p in range(len(self._idx)):
            if int(self._idx[p]) == int(j):
                pos = p
                break
        return float(self._dist[pos]) if pos is not None else float("inf")


class AccessRows:
    """替代 [N,K] 稠密 access 矩阵的轻量只读视图（支撑 list 式访问 access_dist[node][j]）。"""

    def __init__(self, topk_idx, topk_dist, K):
        self._idx = topk_idx
        self._dist = topk_dist
        self.K = K
        self.shape = (topk_idx.shape[0], K)
        self.nbytes = topk_idx.nbytes + topk_dist.nbytes

    def __getitem__(self, node):
        return _RowProxy(self._idx[int(node)], self._dist[int(node)])


def edges_to_local(edge_triple, g2l, n):
    """无向边转成带双向索引的 local 边（对齐原直线 edge_index 双方向）。"""
    fu, fv, fw = edge_triple
    lu = np.array([g2l[int(u)] for u in fu], dtype=np.int64)
    lv = np.array([g2l[int(v)] for v in fv], dtype=np.int64)
    return np.concatenate([lu, lv]), np.concatenate([lv, lu])


def precompute_nearest_k_chunked(access_dist, k_max=16, row_chunk=100000):
    """分块 argpartition，避免 [N, K] 级 int64 临时矩阵（EP_high 内存安全）。"""
    n, K = access_dist.shape
    kk = min(max(1, k_max), K)
    out = np.zeros((n, kk), dtype=np.int32)
    for lo in range(0, n, row_chunk):
        hi = min(lo + row_chunk, n)
        arr = access_dist[lo:hi]
        if kk >= K:
            part = np.tile(np.arange(K), (hi - lo, 1))
        else:
            part = np.argpartition(arr, kk - 1, axis=1)[:, :kk]
        rows = np.arange(hi - lo)[:, None]
        order = np.argsort(arr[rows, part], axis=1)
        out[lo:hi] = part[rows, order].astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def build_pair_list_random(n, cap, seed=42):
    seen = set()
    rng = random.Random(seed)
    _add_random_pairs_local(seen, n, cap, rng, undirected=True, max_trials=100 * cap)
    return list(seen)


def build_pair_list_oracle_mix(n, cap, leaf_of, seed=42):
    pair_list = _sample_oracle_mix_pairs_local(n, cap, leaf_of=leaf_of, seed=seed, undirected=True)
    return [(int(p["s"]), int(p["t"])) for p in pair_list]


def resolve_distances(pair_list, dist_map):
    samples = []
    for s, t in pair_list:
        d = float(dist_map[s][t])
        if d != float("inf"):
            samples.append({"s": s, "t": t, "distance": d})
    return samples


def balance_by_distance(samples, cap, seed=42):
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    samples.sort(key=lambda x: x["distance"])
    n = len(samples)
    if n == 0:
        return []
    bins = [samples[0::3], samples[1::3], samples[2::3]]
    selected = []
    per_bin = cap // 3
    remainder = cap - 3 * per_bin
    for idx, bucket in enumerate(bins):
        rng.shuffle(bucket)
        take = per_bin + (1 if idx < remainder else 0)
        selected.extend(bucket[:take])
    if len(selected) < min(cap, n):
        chosen = {(s["s"], s["t"]) for s in selected}
        leftovers = [s for s in samples if (s["s"], s["t"]) not in chosen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:min(cap, n) - len(selected)])
    rng.shuffle(selected)
    return selected


def load_samples_csv(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            samples.append({"s": int(parts[0]), "t": int(parts[1]), "distance": float(parts[2])})
    return samples


def save_samples_csv(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,t,distance\n")
        for smp in samples:
            f.write(str(int(smp["s"])) + "," + str(int(smp["t"])) + "," + format(float(smp["distance"]), ".6f") + "\n")


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
class SimpleArgs:
    def __init__(self, device, batch_size, highway_k, in_feat, inner_mode="partition"):
        self.device = device
        self.batch_size = batch_size
        self.highway_k = highway_k
        self.in_feat = in_feat
        self.inner_mode = inner_mode


def train_one_config(cfg, context, graph_info, samples_by_strategy, args, log):
    import torch
    from preprocess import split_distance_dataset
    from main import build_loss, run_distance_epoch, evaluate_distance_groups
    from model import DistanceRegressionNet
    torch.manual_seed(args.seed)
    sample_strategy = cfg.get("sample_strategy", "random")
    samples = samples_by_strategy.get(sample_strategy)
    if samples is None:
        return {"name": cfg.get("name"), "error": "no samples for " + sample_strategy}
    train_samples, val_samples, test_samples = split_distance_dataset(
        sample_list=samples, train_ratio=0.8, val_ratio=0.1, seed=args.seed,
    )
    tk = int(cfg.get("transit_k", 0) or 0)
    if tk and tk in context.get("_topk_local", {}):
        lu, lv = context["_topk_local"][tk]
        context["edge_index_highway"] = torch.tensor([lu, lv], dtype=torch.long, device=args.device)
    model = DistanceRegressionNet(
        node_feat_dim=cfg.get("in_feat", args.in_feat),
        highway_feat_dim=cfg.get("in_feat", args.in_feat),
        global_feat_dim=2,
        hidden_dim=cfg.get("hidden_dim", 32),
        inner_out_dim=cfg.get("out_dim", 16),
        inter_out_dim=cfg.get("out_dim", 16),
        fusion_hidden_dim=cfg.get("hidden_dim", 32),
        dropout=0.1,
        use_highway_distance_feature=False,
        highway_distance_feat_dim=4,
        prediction_mode="euclidean_residual",
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = build_loss(cfg.get("loss_type", "huber"))
    num_epoch = int(cfg.get("num_epoch", args.epochs_screen))
    batch_size = int(cfg.get("batch_size", 16))
    early_stop_patience = int(cfg.get("early_stop_patience", 12))
    sa = SimpleArgs(args.device, batch_size, int(cfg.get("highway_k", 3)), cfg.get("in_feat", args.in_feat))
    best_val_metric = float("inf")
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0
    epoch = 0
    for epoch in range(num_epoch):
        t1 = time.time()
        _, train_metrics = run_distance_epoch(
            distance_model=model, sample_list=train_samples, data_graph_info=graph_info,
            args=sa, highway_context=context, criterion=criterion, optimizer=optimizer,
        )
        _, val_metrics = run_distance_epoch(
            distance_model=model, sample_list=val_samples, data_graph_info=graph_info,
            args=sa, highway_context=context, criterion=criterion, optimizer=None,
        )
        et = time.time() - t1
        cur_lr = optimizer.param_groups[0]["lr"]
        log("  [" + str(cfg.get("name")) + "] epoch=" + str(epoch).zfill(3) +
            " train_mae=" + format(train_metrics["mae"], ".4f") +
            " val_mae=" + format(val_metrics["mae"], ".4f") +
            " train_rel=" + format(train_metrics["relative_error"], ".6f") +
            " val_rel=" + format(val_metrics["relative_error"], ".6f") +
            " lr=" + format(cur_lr, ".2e") + " time=" + format(et, ".1f") + "s")
        scheduler.step(val_metrics["relative_error"])
        cur_val = val_metrics["relative_error"]
        if cur_val < best_val_metric:
            best_val_metric = cur_val
            best_val_mae = val_metrics["mae"]
            no_improve = 0
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= early_stop_patience:
            log("  [" + str(cfg.get("name")) + "] early stop at epoch=" + str(epoch) + " best_epoch=" + str(best_epoch))
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_metrics = run_distance_epoch(
        distance_model=model, sample_list=test_samples, data_graph_info=graph_info,
        args=sa, highway_context=context, criterion=criterion, optimizer=None,
    )
    group_metrics = evaluate_distance_groups(
        distance_model=model, sample_list=test_samples, data_graph_info=graph_info,
        args=sa, highway_context=context, leaf_of=context["leaf_of"],
    )
    res = {
        "name": cfg.get("name", "cfg"),
        "sample_strategy": sample_strategy,
        "loss_type": cfg.get("loss_type", "huber"),
        "highway_k": int(cfg.get("highway_k", 3)),
        "hidden_dim": int(cfg.get("hidden_dim", 32)),
        "out_dim": int(cfg.get("out_dim", 16)),
        "transit_k": tk,
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_relative_error": best_val_metric,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_relative_error": test_metrics["relative_error"],
    }
    for gname, m in group_metrics.items():
        res["group_" + gname + "_count"] = m["count"]
        res["group_" + gname + "_relative_error"] = m["relative_error"]
        res["group_" + gname + "_mae"] = m["mae"]
    return res


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--off_file", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--in_feat", type=int, default=64)
    p.add_argument("--epochs_screen", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid_json", default="scripts_local/ep_high_grid.json")
    p.add_argument("--run_tag", default="ep_high")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--train_samples", type=int, default=0)
    p.add_argument("--only", default="prep,sample,grid")
    return p


def main():
    args = build_parser().parse_args()
    stages = [s.strip() for s in args.only.split(",") if s.strip()]
    t0 = time.time()
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs", "results"), exist_ok=True)
    vlog = os.environ.get("CLOUD_LOG", os.path.join(ROOT, "logs", args.run_tag + "_runner.log"))
    logf = open(vlog, "a", encoding="utf-8")

    def log(msg):
        line = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(msg)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(ROOT, args.off_file)
    cache_dir = os.path.join(ROOT, "outputs", "cache_ep_high")
    os.makedirs(cache_dir, exist_ok=True)
    samples_by_strategy = {}

    graph_info = None
    leaf_of = None
    context = None
    vertices3d = None
    boundary = None
    coords = None
    full_edges = None
    topk_local = {}
    topk_idx = None
    topk_dist = None

    if "prep" in stages:
        log("[stage=prep] loading " + off_path)
        vertices, faces = load_off(off_path)
        graph_info, coords = build_mesh_graph(vertices, faces)
        n = len(graph_info[0])
        leaf_of, num_leaves, occupied = build_quadtree(coords, args.max_depth, args.capacity, adaptive=True)
        log("[stage=prep] |V|=" + str(n) + " leaves=" + str(num_leaves) + " occupied=" + str(occupied))
        boundary = find_boundary_nodes(graph_info, leaf_of)
        if not boundary:
            boundary = set(range(n))
        boundary_sorted = sorted(boundary)
        K = len(boundary_sorted)
        log("[stage=prep] boundary K=" + str(K) + " (" + format(100.0 * K / n, ".2f") + "% of V)")
        adj = _adj_from_graph_info(graph_info, weighted=True)
        cell_members = defaultdict(list)
        for u in boundary_sorted:
            cell_members[leaf_of[u]].append(u)
        eu, ev, ew = graph_info[3][0], graph_info[3][1], graph_info[4]

        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra as sp_dijkstra
        w = np.maximum(1e-9, np.asarray(ew, dtype=np.float64))
        csr = sp.csr_matrix((w, (np.asarray(eu, dtype=np.int64), np.asarray(ev, dtype=np.int64))), shape=(n, n))
        tt = time.time()
        sp_dijkstra(csr, directed=False, indices=boundary_sorted[:3])
        per_src = (time.time() - tt) / 3.0
        log("[stage=prep] single-source dijkstra ~" + format(per_src, ".3f") + "s -> seq est " +
            format(K * per_src / 3600, ".2f") + "h / " + str(args.workers) + " workers est " +
            format(K * per_src / args.workers / 3600, ".2f") + "h")
        art = os.path.join(cache_dir, "topk_artifacts_d" + str(args.max_depth) + "_c" + str(args.capacity) + ".npz")
        if os.path.exists(art):
            log("[stage=prep] loading topk artifacts: " + art)
            aa = np.load(art)
            topk_idx = aa["topk_idx"]
            topk_dist = aa["topk_dist"]
            lu = aa["lu"]
            lv = aa["lv"]
            full_edges = (lu, lv, np.zeros(len(lu), dtype=np.float32))
            topk_local = {}
            for kk in (8, 16):
                if ("lku" + str(kk)) in aa:
                    topk_local[kk] = (aa["lku" + str(kk)], aa["lkv" + str(kk)])
            log("[stage=prep] loaded: topk=" + str(topk_idx.shape) + " edges=" + str(len(lu)))
        else:
            t1 = time.time()
            topk_idx, topk_dist, full_edges, topk_edges = parallel_preprocess(
                n, eu, ev, ew, boundary_sorted, leaf_of, cell_members, adj,
                workers=args.workers, chunk=args.chunk, k_max=16,
            )
            log("[stage=prep] topk access tables computed in " + format(time.time() - t1, ".1f") +
                "s shape=" + str(topk_idx.shape) + " " + format(topk_idx.nbytes / 1e6, ".1f") + "MB")
            g2l = {g: i for i, g in enumerate(boundary_sorted)}
            lu, lv = edges_to_local(full_edges, g2l, n)
            log("[stage=prep] highway full edges=" + str(len(lu)))
            topk_local = {}
            for kk in (8, 16):
                if kk in topk_edges:
                    ku, kv, kw = topk_edges[kk]
                    lku, lkv = edges_to_local((ku, kv, kw), g2l, n)
                    topk_local[kk] = (lku, lkv)
                    log("[stage=prep] highway tk" + str(kk) + " edges=" + str(len(lku)))
            save_dict = {
                "topk_idx": topk_idx,
                "topk_dist": topk_dist.astype(np.float32),
                "lu": lu.astype(np.int64),
                "lv": lv.astype(np.int64),
            }
            for kk, (lku, lkv) in topk_local.items():
                save_dict["lku" + str(kk)] = lku.astype(np.int64)
                save_dict["lkv" + str(kk)] = lkv.astype(np.int64)
            np.savez_compressed(art, **save_dict)
            log("[stage=prep] artifacts saved: " + art)
        vertices3d = vertices
        del faces, vertices
    else:
        log("[stage=prep] skipped")

    if "sample" in stages:
        log("[stage=sample] begin")
        n = len(graph_info[0])
        eu, ev, ew = graph_info[3][0], graph_info[3][1], graph_info[4]
        pair_list = build_pair_list_random(n, args.samples, seed=args.seed)
        by_src = defaultdict(list)
        for s, t in pair_list:
            by_src[s].append(t)
        csv_path = os.path.join(cache_dir, "ep_high_random_n" + str(args.samples) + "_seed" + str(args.seed) + ".csv")
        if os.path.exists(csv_path):
            samples_random = load_samples_csv(csv_path)
            log("[stage=sample] random loaded from cache: " + str(len(samples_random)))
        else:
            t1 = time.time()
            samples_random = parallel_resolve_samples(n, eu, ev, ew, by_src, args.workers, chunk=args.chunk)
            log("[stage=sample] random samples=" + str(len(samples_random)) + " in " + format(time.time() - t1, ".1f") + "s")
            save_samples_csv(csv_path, samples_random)
        samples_by_strategy["random"] = samples_random
        del by_src, pair_list
        try:
            pair_list = build_pair_list_oracle_mix(n, args.samples, leaf_of, seed=args.seed)
            by_src = defaultdict(list)
            for s, t in pair_list:
                by_src[s].append(t)
            csv_path = os.path.join(cache_dir, "ep_high_oracle_mix_n" + str(args.samples) + "_seed" + str(args.seed) + ".csv")
            if os.path.exists(csv_path):
                samples_mix = load_samples_csv(csv_path)
                log("[stage=sample] oracle_mix loaded from cache: " + str(len(samples_mix)))
            else:
                t1 = time.time()
                samples_mix = parallel_resolve_samples(n, eu, ev, ew, by_src, args.workers, chunk=args.chunk)
                log("[stage=sample] oracle_mix samples=" + str(len(samples_mix)) + " in " + format(time.time() - t1, ".1f") + "s")
                save_samples_csv(csv_path, samples_mix)
            samples_by_strategy["oracle_mix"] = samples_mix
            del by_src, pair_list
        except Exception as e:
            log("[stage=sample] oracle_mix failed: " + repr(e))
        try:
            pool = build_pair_list_random(n, args.samples, seed=args.seed)
            by_src = defaultdict(list)
            for s, t in pool:
                by_src[s].append(t)
            csv_path = os.path.join(cache_dir, "ep_high_distance_gap_n" + str(args.samples) + "_seed" + str(args.seed) + ".csv")
            if os.path.exists(csv_path):
                samples_gap = load_samples_csv(csv_path)
                log("[stage=sample] distance_gap loaded from cache: " + str(len(samples_gap)))
                del pool, by_src
                samples_by_strategy["distance_gap"] = samples_gap
            else:
                t1 = time.time()
                log("[stage=sample] distance_gap pool=" + str(len(pool)) + " sources=" + str(len(by_src)))
                pool_samples = parallel_resolve_samples(n, eu, ev, ew, by_src, args.workers, chunk=args.chunk)
                samples_gap = balance_by_distance(pool_samples, args.samples, seed=args.seed)
                log("[stage=sample] distance_gap samples=" + str(len(samples_gap)) + " in " + format(time.time() - t1, ".1f") + "s")
                save_samples_csv(csv_path, samples_gap)
                del pool, by_src, pool_samples
                samples_by_strategy["distance_gap"] = samples_gap
        except Exception as e:
            log("[stage=sample] distance_gap failed: " + repr(e))

    if "grid" in stages:
        import torch
        from preprocess import build_highway_context  # noqa: F401
        boundary_sorted = sorted(boundary)
        g2l = {g: i for i, g in enumerate(boundary_sorted)}
        K = len(boundary_sorted)
        context = build_highway_context(
            graph_info=graph_info,
            coords=coords,
            highway_global_ids=boundary_sorted,
            local_edges=(lu, lv),
            access_dist=AccessRows(topk_idx, topk_dist, K),
            highway_pair_dist=np.zeros((K, K), dtype=np.float32),
            leaf_of=leaf_of,
            feature_dim=args.in_feat,
            device=args.device,
        )
        context["node_coords3d"] = {i: vertices3d[i] for i in range(len(vertices3d))}
        context["nearest_k_local"] = topk_idx
        context["graph_info"] = graph_info
        context["_topk_local"] = topk_local
        log("[stage=context] built. nearest_k=" + str(context["nearest_k_local"].shape))
        grid_path = args.grid_json if os.path.isabs(args.grid_json) else os.path.join(ROOT, args.grid_json)
        if not os.path.exists(grid_path):
            with open(grid_path, "w", encoding="utf-8") as f:
                json.dump(build_default_grid(), f)
        if args.train_samples and args.train_samples > 0 and args.train_samples < len(next(iter(samples_by_strategy.values()))):
            import random as _rnd
            for strat in list(samples_by_strategy.keys()):
                lst = samples_by_strategy[strat]
                _rnd.Random(args.seed).shuffle(lst)
                samples_by_strategy[strat] = lst[:args.train_samples]
                log("[stage=grid] downsample " + strat + " -> " + str(len(lst[:args.train_samples])))
        log("[stage=grid] running configs from " + grid_path + " (appended online)")
        results_path = os.path.join(ROOT, "outputs", "results", args.run_tag + "_grid.jsonl")
        seen_names = set()
        results = []
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            old_row = json.loads(line)
                            results.append(old_row)
                            seen_names.add(old_row.get("name"))
                        except Exception:
                            pass
            log("[stage=grid] resumed: " + str(len(results)) + " already done, " + str(len(seen_names)) + " names")
        counter = 0
        while True:
            with open(grid_path, "r", encoding="utf-8") as f:
                grid_now = json.load(f)
            pending = [c for c in grid_now if c.get("name") not in seen_names]
            if not pending:
                break
            cfg = pending[0]
            counter += 1
            log("[stage=grid] #" + str(counter) + " " + str(cfg.get("name")) + " start")
            t1 = time.time()
            try:
                res = train_one_config(cfg, context, graph_info, samples_by_strategy, args, log)
                res["elapsed_seconds"] = round(time.time() - t1, 3)
            except Exception as e:
                import traceback
                log("[stage=grid] cfg error: " + repr(e))
                traceback.print_exc()
                res = {"name": cfg.get("name"), "error": repr(e)}
            results.append(res)
            seen_names.add(cfg.get("name"))
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            log("[stage=grid] #" + str(counter) + " " + str(cfg.get("name")) + " done test_rel=" + str(res.get("test_relative_error")))
        valid = [r for r in results if r.get("test_relative_error") is not None]
        valid.sort(key=lambda r: r["test_relative_error"])
        log("[stage=grid] === RANKING (by test_relative_error) ===")
        for r in valid:
            log("  " + format(r["test_relative_error"], ".5f") + " | " + str(r["name"]) + " | best_ep=" + str(r.get("best_epoch")) +
                " mae=" + format(r.get("test_mae", float("nan")), ".2f") + " rmse=" + format(r.get("test_rmse", float("nan")), ".2f"))

    log("[ALL DONE] total " + format(time.time() - t0, ".1f") + "s")


def _sample_worker_task(task):
    src_list, chunk = task
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    parts = []
    for i in range(0, len(src_list), chunk):
        batch = src_list[i:i + chunk]
        dmat = sp_dijkstra(_W_CSR, directed=False, indices=batch)
        parts.append(np.ascontiguousarray(dmat, dtype=np.float32))
    return src_list, np.concatenate(parts, axis=0)


def parallel_resolve_samples(n, eu, ev, ew, by_src, workers=32, chunk=64):
    """流式并行采样：每块算完立即解析点对并释放距离行，内存 O(chunk*N)。"""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    srcs = list(by_src.keys())
    if not srcs:
        return []
    tasks = [(srcs[i:i + chunk], chunk) for i in range(0, len(srcs), chunk)]
    samples = []
    with ctx.Pool(processes=min(workers, len(tasks)), initializer=_dijkstra_worker_init,
                  initargs=(n, eu, ev, ew, {}, {})) as pool:
        for group, arr in pool.imap_unordered(_sample_worker_task, tasks, chunksize=1):
            for j, s in enumerate(group):
                d_row = arr[j]
                for t in by_src[int(s)]:
                    d = float(d_row[t])
                    if d != float("inf"):
                        samples.append({"s": int(s), "t": t, "distance": d})
    return samples


def _dijkstra_worker_batch(sources, chunk=64):
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    parts = []
    for i in range(0, len(sources), chunk):
        batch = sources[i:i + chunk]
        dmat = sp_dijkstra(_W_CSR, directed=False, indices=batch)
        parts.append(np.ascontiguousarray(dmat, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def build_default_grid():
    return [
        {"name": "best_from_ep_low", "sample_strategy": "random", "loss_type": "huber", "highway_k": 3, "hidden_dim": 32, "out_dim": 16, "transit_k": 0},
        {"name": "random_relative_k3_s", "sample_strategy": "random", "loss_type": "relative", "highway_k": 3, "hidden_dim": 32, "out_dim": 16},
        {"name": "random_logl1_k3_s", "sample_strategy": "random", "loss_type": "log_l1", "highway_k": 3, "hidden_dim": 32, "out_dim": 16},
        {"name": "random_huber_k1_s", "sample_strategy": "random", "loss_type": "huber", "highway_k": 1, "hidden_dim": 32, "out_dim": 16},
        {"name": "random_huber_k5_s", "sample_strategy": "random", "loss_type": "huber", "highway_k": 5, "hidden_dim": 32, "out_dim": 16},
        {"name": "random_huber_k3_m", "sample_strategy": "random", "loss_type": "huber", "highway_k": 3, "hidden_dim": 64, "out_dim": 32},
        {"name": "gap_huber_k3_s", "sample_strategy": "distance_gap", "loss_type": "huber", "highway_k": 3, "hidden_dim": 32, "out_dim": 16},
        {"name": "mix_huber_k3_s", "sample_strategy": "oracle_mix", "loss_type": "huber", "highway_k": 3, "hidden_dim": 32, "out_dim": 16},
        {"name": "random_huber_k3_tk8", "sample_strategy": "random", "loss_type": "huber", "highway_k": 3, "hidden_dim": 32, "out_dim": 16, "transit_k": 8},
        {"name": "random_huber_k3_b32", "sample_strategy": "random", "loss_type": "huber", "highway_k": 3, "hidden_dim": 32, "out_dim": 16, "batch_size": 32},
    ]


if __name__ == "__main__":
    main()
