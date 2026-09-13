# -*- coding: utf-8 -*-
"""冒烟测试：sample_terrain.off 上验证并行预处理与训练链路（云上跑）。"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from build_highway import load_off, build_mesh_graph, build_quadtree, find_boundary_nodes, _adj_from_graph_info, build_pipeline_inputs
from preprocess import build_highway_context, split_distance_dataset
from main import build_loss, run_distance_epoch
from model import DistanceRegressionNet
from scripts_local.cloud_ep_high_runner import (
    parallel_preprocess, edges_to_local, parallel_resolve_samples, build_pair_list_random,
    save_samples_csv, SimpleArgs, AccessRows,
)

OFF = sys.argv[1] if len(sys.argv) > 1 else "data/sample_terrain.off"
out = sys.stdout

print("[smoke] loading", OFF, flush=True)
vertices, faces = load_off(OFF)
graph_info, coords = build_mesh_graph(vertices, faces)
n = len(graph_info[0])
leaf_of, num_leaves, occupied = build_quadtree(coords, 3, 32, adaptive=True)
boundary = find_boundary_nodes(graph_info, leaf_of) or set(range(n))
boundary_sorted = sorted(boundary)
print("[smoke] V=", n, "leaves=", num_leaves, "K=", len(boundary_sorted), flush=True)
adj = _adj_from_graph_info(graph_info, weighted=True)
from collections import defaultdict
cell_members = defaultdict(list)
for u in boundary_sorted:
    cell_members[leaf_of[u]].append(u)
eu, ev, ew = graph_info[3][0], graph_info[3][1], graph_info[4]

print("[smoke] parallel_preprocess ...", flush=True)
t1 = time.time()
topk_idx_p, topk_dist_p, full_edges_p, topk_p = parallel_preprocess(n, eu, ev, ew, boundary_sorted, leaf_of, cell_members, adj, workers=8, chunk=32)
print("[smoke] parallel preprocess done in", time.time() - t1, "s topk shapes:", topk_idx_p.shape, flush=True)

print("[smoke] sequential reference ...", flush=True)
import torch
ref_graph_info, ref_coords, ref_leaf_of, ref_num_leaves, ref_context = build_pipeline_inputs(OFF, max_depth=3, capacity=32, feature_dim=16, device="cpu")
ref_access = np.asarray(ref_context["access_dist"])
n = topk_idx_p.shape[0]
K = ref_access.shape[1]
kk = topk_idx_p.shape[1]
ref_topk = np.zeros((n, kk), dtype=np.int32)
ref_topk_dist = np.zeros((n, kk), dtype=np.float32)
for i in range(n):
    order = np.argsort(ref_access[i], kind="stable")[:kk]
    ref_topk[i] = order
    ref_topk_dist[i] = ref_access[i][order]
print("[smoke] topk idx match ratio =", float((topk_idx_p == ref_topk).mean()), flush=True)
print("[smoke] topk dist max abs diff =", float(np.abs(topk_dist_p - ref_topk_dist).max()), flush=True)
# 边比较（转成 dict 集合）
g2l = {g: i for i, g in enumerate(boundary_sorted)}
lu_p, lv_p = edges_to_local(full_edges_p, g2l, n)
ref_ei = ref_context["edge_index_highway"]
lu_ref = ref_ei[0].tolist()
lv_ref = ref_ei[1].tolist()
set_p = set(zip(lu_p.tolist(), lv_p.tolist()))
set_r = set(zip(lu_ref, lv_ref))
print("[smoke] edges parallel=", len(set_p), "reference=", len(set_r), "only_parallel=", len(set_p - set_r), "only_ref=", len(set_r - set_p), flush=True)

print("[smoke] sampling ...", flush=True)
pair_list = build_pair_list_random(n, 300, seed=42)
by_src = defaultdict(list)
for s, t in pair_list:
    by_src[s].append(t)
samples = parallel_resolve_samples(n, eu, ev, ew, by_src, 8)
print("[smoke] samples=", len(samples), flush=True)
train, val, test = split_distance_dataset(sample_list=samples, train_ratio=0.8, val_ratio=0.1, seed=42)
print("[smoke] train/val/test=", len(train), len(val), len(test), flush=True)

print("[smoke] build context ...", flush=True)
K = len(boundary_sorted)
context = build_highway_context(
    graph_info=graph_info, coords=coords, highway_global_ids=boundary_sorted,
    local_edges=(lu_p, lv_p), access_dist=AccessRows(topk_idx_p, topk_dist_p, K),
    highway_pair_dist=np.zeros((K, K), dtype=np.float32),
    leaf_of=leaf_of, feature_dim=16, device="cpu",
)
context["node_coords3d"] = {i: vertices[i] for i in range(n)}
context["nearest_k_local"] = topk_idx_p
context["graph_info"] = graph_info

print("[smoke] training 2 epochs ...", flush=True)
torch.manual_seed(42)
model = DistanceRegressionNet(
    node_feat_dim=16, highway_feat_dim=16, global_feat_dim=2,
    hidden_dim=32, inner_out_dim=16, inter_out_dim=16, fusion_hidden_dim=32,
    dropout=0.1, use_highway_distance_feature=False, highway_distance_feat_dim=4,
    prediction_mode="euclidean_residual",
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = build_loss("huber")
sa = SimpleArgs("cpu", 16, 3, 16)
for epoch in range(2):
    _, tm = run_distance_epoch(model, train, graph_info, sa, context, criterion, optimizer)
    _, vm = run_distance_epoch(model, val, graph_info, sa, context, criterion, None)
    print("[smoke] epoch", epoch, "train_mae", round(tm["mae"], 4), "val_mae", round(vm["mae"], 4), "rel", round(vm["relative_error"], 5), flush=True)
print("[smoke] SMOKE_OK", flush=True)
