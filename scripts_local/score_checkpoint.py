# -*- coding: utf-8 -*-
"""用已训练好的 checkpoint 在一批点对上跑推理，导出逐对预测值。

为什么要它：main.py 只写汇总指标，不保存逐对预测。要做"对着精确测地真值重新打分"
就必须先拿到模型在每个 (s,t) 上的预测值。

用法：
  python3 scripts_local/score_checkpoint.py \
      --model_path outputs/models/<run>.pt \
      --params_file outputs/params/<run>.txt \
      --pairs_file outputs/geo_results/<tag>_exact_geo.csv \
      --pairs_format exact_geo \
      --out_file outputs/geo_results/<tag>_preds.csv
"""
import argparse
import os
import sys
import types

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from build_highway import (  # noqa: E402
    build_pipeline_inputs_cached, build_graph_and_partition, load_off,
)
from preprocess import (  # noqa: E402
    precompute_nearest_k, build_full_graph_tensors, build_pair_euclidean_features,
)
from model import build_distance_model  # noqa: E402
from main import run_distance_epoch, run_single_gnn_epoch  # noqa: E402


def read_params(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip(); v = v.strip()
            if v in ("True", "False"):
                out[k] = (v == "True")
            else:
                try:
                    out[k] = int(v)
                except ValueError:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
    return out


def read_pairs(path, fmt):
    """fmt: 's_t_true' -> (s,t,true)；'exact_geo' -> (s,t,graph_dist) 取前两列。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            try:
                pairs.append((int(p[0]), int(p[1])))
            except ValueError:
                continue
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--params_file", required=True)
    ap.add_argument("--pairs_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--off_file", default="", help="覆盖 params 里的地形路径")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    P = read_params(args.params_file)
    off_path = args.off_file or P["off_file"]
    if not os.path.isabs(off_path):
        ff = P.get("file_folder", "data")
        cand = os.path.join(PROJECT_ROOT, off_path)
        off_path = cand if os.path.exists(cand) else os.path.join(PROJECT_ROOT, ff, off_path)
    if not os.path.exists(off_path):
        print("[score] 找不到地形：" + off_path)
        return 1

    A = types.SimpleNamespace(
        in_feat=int(P.get("in_feat", 64)), hidden_dim=int(P.get("hidden_dim", 128)),
        out_dim=int(P.get("out_dim", 64)), dropout_ratio=float(P.get("dropout_ratio", 0.2)),
        highway_k=int(P.get("highway_k", 3)), inner_mode=P.get("inner_mode", "partition"),
        architecture=P.get("architecture", "three_stage"),
        prediction_mode=P.get("prediction_mode", "direct"),
        disable_highway_distance_feature=bool(P.get("disable_highway_distance_feature", False)),
        single_gnn_layers=int(P.get("single_gnn_layers", 3)),
        single_gnn_head_batch=int(P.get("single_gnn_head_batch", 8192)),
        single_gnn_step_samples=0,
        max_depth=int(P.get("max_depth", 3)), capacity=int(P.get("capacity", 32)),
        uniform=bool(P.get("uniform", False)), transit_k=int(P.get("transit_k", 0)),
        device=args.device, batch_size=int(P.get("batch_size", 16)), seed=int(P.get("seed", 42)),
    )

    model = build_distance_model(
        architecture=A.architecture, prediction_mode=A.prediction_mode,
        node_feat_dim=A.in_feat, highway_feat_dim=A.in_feat, global_feat_dim=2,
        hidden_dim=A.hidden_dim, out_dim=A.out_dim, fusion_hidden_dim=A.hidden_dim,
        dropout=A.dropout_ratio,
        use_highway_distance_feature=not A.disable_highway_distance_feature,
        highway_distance_feat_dim=4, single_gnn_layers=A.single_gnn_layers,
    ).to(A.device)
    state = torch.load(args.model_path, map_location=A.device)
    model.load_state_dict(state)
    model.eval()
    print("[score] loaded %s (arch=%s mode=%s)" % (os.path.basename(args.model_path),
                                                   A.architecture, A.prediction_mode))

    pairs = read_pairs(args.pairs_file, "exact_geo")
    samples = [{"s": s, "t": t, "distance": 0.0} for s, t in pairs]
    print("[score] %d pairs" % len(samples))

    vertices3d, _ = load_off(off_path)
    cache_dir = os.path.join(PROJECT_ROOT, "outputs", "cache")
    if A.architecture == "single_gnn":
        gi, coords, leaf_of, _nl = build_graph_and_partition(
            off_path, max_depth=A.max_depth, capacity=A.capacity, adaptive=not A.uniform)
        fx, fei = build_full_graph_tensors(gi, coords, feature_dim=A.in_feat, device=A.device)
        ef = build_pair_euclidean_features(samples, vertices3d, A.device)
        _, _m = run_single_gnn_epoch(model=model, sample_list=samples, euclid_feats=ef,
                                     full_x=fx, full_edge_index=fei, criterion=torch.nn.L1Loss(),
                                     optimizer=None, head_batch_size=A.single_gnn_head_batch)
        preds = _predict_single(model, samples, ef, fx, fei, A)
    else:
        gi, coords, leaf_of, _nl, hc = build_pipeline_inputs_cached(
            off_path=off_path, max_depth=A.max_depth, capacity=A.capacity,
            adaptive=not A.uniform, weighted=True, feature_dim=A.in_feat,
            device=A.device, cache_dir=cache_dir, transit_k=A.transit_k)
        hc["node_coords3d"] = {i: vertices3d[i] for i in range(len(vertices3d))}
        hc["nearest_k_local"] = precompute_nearest_k(hc["access_dist"], k_max=max(16, A.highway_k))
        preds = _predict_three(model, samples, gi, A, hc)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("s,t,pred\n")
        for (s, t), p in zip(pairs, preds):
            f.write("%d,%d,%.6f\n" % (s, t, p))
    print("[score] wrote -> %s" % args.out_file)
    return 0


def _predict_three(model, samples, gi, A, hc):
    from preprocess import build_synthetic_partition_inputs
    out = []
    bs = max(1, A.batch_size)
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            chunk = samples[i:i + bs]
            batch_inputs = [build_synthetic_partition_inputs(
                graph_info=gi, sample=s, k_highway=max(1, A.highway_k), feature_dim=A.in_feat,
                device=A.device, external_highway_context=hc, inner_mode=A.inner_mode) for s in chunk]
            p = model.forward_batch(batch_inputs)
            out.extend(float(v) for v in p.detach().cpu().view(-1))
    return out


def _predict_single(model, samples, ef, fx, fei, A):
    s_idx = torch.tensor([x["s"] for x in samples], dtype=torch.long, device=fx.device)
    t_idx = torch.tensor([x["t"] for x in samples], dtype=torch.long, device=fx.device)
    out = []
    bs = max(1, A.single_gnn_head_batch)
    with torch.no_grad():
        h = model.encode(fx, fei)
        for i in range(0, len(samples), bs):
            sl = slice(i, i + bs)
            p = model.predict_from_embeddings(h[s_idx[sl]], h[t_idx[sl]], ef[sl])
            out.extend(float(v) for v in p.detach().cpu().view(-1))
    return out


if __name__ == "__main__":
    sys.exit(main())
