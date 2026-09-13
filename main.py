# 训练入口：读 .off → 四叉树分区+高速上下文 → 采样节点对 → 训练三段式 GNN 回归最短路距离。
import argparse
import copy
import os
import time

import torch
import torch.nn as nn

from preprocess import (
    build_distance_samples,
    split_distance_dataset,
    build_synthetic_partition_inputs,
    precompute_nearest_k,
    build_full_graph_tensors,
    build_pair_euclidean_features,
    build_point_feature_tensors,
    load_label_pairs_csv,
)
from build_highway import (
    build_pipeline_inputs_cached,
    build_graph_and_partition,
    _file_fingerprint,
    load_off,
)
from model import build_distance_model, compute_distance_metrics
from gnn_baselines import BASELINE_ARCHITECTURES, build_baseline_model
from litege_descriptor import compute_udf_pca_descriptor


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="distance", choices=["distance"], help="only distance task is supported")
    parser.add_argument("--off_file", type=str, required=True, help="input .off terrain mesh path")
    parser.add_argument("--file_folder", type=str, default="data", help="base folder for relative --off_file paths (resolved under project root)")

    # 四叉树分区参数（对齐 EAR-Oracle 的 grid/quadtree）
    parser.add_argument("--max_depth", type=int, default=3, help="quadtree max depth")
    parser.add_argument("--capacity", type=int, default=32, help="max points per leaf (adaptive mode)")
    parser.add_argument("--uniform", action="store_true", help="uniform quadtree instead of adaptive")

    parser.add_argument("--in_feat", type=int, default=64, help="input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="output embedding dimension")
    parser.add_argument("--dropout_ratio", type=float, default=0.2, help="dropout ratio")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (初始学习率)")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "plateau"],
                        help="学习率调度器；plateau=ReduceLROnPlateau(按 val_mae 触发降 LR)")
    parser.add_argument("--lr_patience", type=int, default=5, help="plateau: val_mae 连续多少轮不降就降 LR")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="plateau: 每次降 LR 的乘数(如 0.5)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="plateau: LR 下限")
    parser.add_argument("--num_epoch", type=int, default=20, help="max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size (samples per GNN forward/step); 1 = 旧逐样本行为")
    parser.add_argument("--train_percent", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="early stop patience")
    parser.add_argument("--distance_samples", type=int, default=3000, help="cap on number of (s,t) pairs; <=0 means use ALL unique reachable pairs")
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="random",
        choices=["random", "oracle_mix", "distance_gap"],
        help="distance pair sampling: random=原始随机唯一点对；oracle_mix=同叶子/跨叶子/mixed 约 1:1:1；distance_gap=短/中/长距离均衡",
    )
    parser.add_argument("--highway_k", type=int, default=3, help="number of highway connectors for each endpoint")
    parser.add_argument("--transit_k", type=int, default=0, help="sparsify in-cell highway transit edges: keep nearest k per highway node; 0=full")
    parser.add_argument(
        "--inner_mode",
        type=str,
        default="partition",
        choices=["partition", "ego"],
        help="Inner-GNN subgraph: 'partition'=四叉树叶子盒子图(对齐G1~G4); 'ego'=2-hop ego(消融)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="log_l1",
        choices=["l1", "log_l1", "relative", "huber"],
        help="training loss; log_l1/relative normalize errors across distance scales",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="mae",
        choices=["mae", "relative_error"],
        help="validation metric used for best checkpoint and early stopping",
    )
    parser.add_argument(
        "--disable_highway_distance_feature",
        action="store_true",
        help="disable the highway-decomposition distance feature fed into the fusion MLP",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="direct",
        choices=["native", "direct", "highway_residual", "euclidean_residual"],
        help=(
            "native=baseline 论文原样输出；"
            "direct=原始距离回归；"
            "highway_residual=以 access+highway+access 强基线为中心学习修正；"
            "euclidean_residual=以两点 3D 欧氏直线距离为中心学习修正"
        ),
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="three_stage",
        choices=["three_stage", "single_gnn", "gegnn", "neurogf", "litege"],
        help=(
            "three_stage=主方法(四叉树地形分区子图 + highway 骨架 + Inner/Inter 双 GNN + Fusion)；"
            "single_gnn=消融分支(不使用地形分区与 highway 网络，只在全图上跑一个 GNN，"
            "直接吃 s/t 两个节点嵌入)"
        ),
    )
    parser.add_argument(
        "--single_gnn_layers",
        type=int,
        default=3,
        help="architecture=single_gnn 时全图 GNN 的层数（默认 3，≥ 主方法 Inner/Inter 的 2 层，保证消融分支不吃亏）",
    )
    parser.add_argument(
        "--single_gnn_step_samples",
        type=int,
        default=0,
        help=(
            "architecture=single_gnn 时每个 optimizer step 覆盖的样本数（整图前向一次、头分批累积梯度）。"
            "0 = 自动等于 --batch_size，即与三段式**每轮优化步数完全相同**（消融公平性要求）；"
            "大图上如嫌慢可显式调大，但会减少更新次数、对消融分支不利"
        ),
    )
    parser.add_argument(
        "--single_gnn_head_batch",
        type=int,
        default=8192,
        help="architecture=single_gnn 时融合头每个 mini-batch 的样本数（整图只做一次消息传递，头可大批量）",
    )
    parser.add_argument("--torch_threads", type=int, default=0,
                        help="PyTorch intra-op 线程数；0=用默认。多任务并行时设小值可避免线程超订把彼此拖慢 5~10 倍")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="梯度范数裁剪阈值；0=关闭。基线（尤其 GeGnn 的 max 聚合 + 平方差解码）"
                             "在本数据上会出现梯度消失/爆炸，开启后各方法同等受益")
    parser.add_argument("--litege_pca_dim", type=int, default=200,
                        help="architecture=litege 时 UDF-PCA 形状描述子维度")
    parser.add_argument("--gegnn_layers", type=int, default=4,
                        help="architecture=gegnn 时 GeoConv 层数")
    parser.add_argument("--labels_file",
        type=str,
        default="",
        help=(
            "可选：外部标签 CSV（含列 s,t,<value_col>），直接作为样本集并跳过内置采样/Dijkstra。"
            "用于把监督标签换成精确曲面测地距离（pygeodesic 生成）"
        ),
    )
    parser.add_argument(
        "--labels_col",
        type=str,
        default="distance",
        help="外部标签 CSV 中作为监督距离的列名（精确测地结果里是 exact_geo）",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="可选：固定本次运行的产物名（outputs 下用 <图名>_<run_tag>），便于消融矩阵脚本机械解析",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="enable PyTorch autograd anomaly detection for debugging (slower)",
    )
    parser.add_argument("--cache_dir", type=str, default="outputs/cache", help="高速上下文缓存目录（按项目根解析）；首次算完存盘，之后直接读")
    parser.add_argument("--no_cache", action="store_true", help="禁用缓存，每次重新计算分区+高速")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling, splitting, and model initialization")
    return parser


def build_loss(loss_type, eps=1e-6):
    """构造训练损失。log_l1/relative 用于跨距离尺度归一化误差（见 note.txt）。"""
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "huber":
        return nn.SmoothL1Loss()
    if loss_type == "log_l1":
        def _log_l1(pred, true):
            return torch.abs(torch.log1p(pred) - torch.log1p(true)).mean()
        return _log_l1
    if loss_type == "relative":
        def _relative(pred, true):
            return (torch.abs(pred - true) / (true + eps)).mean()
        return _relative
    raise ValueError(f"unknown loss_type: {loss_type}")


def run_distance_epoch(distance_model, sample_list, data_graph_info, args, highway_context, criterion, optimizer=None):
    is_train = optimizer is not None
    all_pred = []
    all_true = []
    total_loss = 0.0
    num_batches = 0

    if is_train:
        distance_model.train()
    else:
        distance_model.eval()

    batch_size = max(1, args.batch_size)
    for start in range(0, len(sample_list), batch_size):
        chunk = sample_list[start:start + batch_size]
        batch_inputs = [
            build_synthetic_partition_inputs(
                graph_info=data_graph_info,
                sample=sample,
                k_highway=max(1, args.highway_k),
                feature_dim=args.in_feat,
                device=args.device,
                external_highway_context=highway_context,
                inner_mode=args.inner_mode,
            )
            for sample in chunk
        ]
        y_true = torch.tensor([s["distance"] for s in chunk], dtype=torch.float, device=args.device)

        if is_train:
            optimizer.zero_grad()
            preds = distance_model.forward_batch(batch_inputs)  # [B]
            loss = criterion(preds, y_true)
            loss.backward()
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(distance_model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                preds = distance_model.forward_batch(batch_inputs)
                loss = criterion(preds, y_true)

        total_loss += float(loss.item())
        num_batches += 1
        all_pred.append(preds.detach().view(-1))
        all_true.append(y_true.view(-1))

    pred_tensor = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0)
    true_tensor = torch.cat(all_true, dim=0) if all_true else torch.empty(0)
    metrics = compute_distance_metrics(true_tensor, pred_tensor) if len(sample_list) > 0 else {
        "mae": float("nan"),
        "rmse": float("nan"),
        "relative_error": float("nan"),
    }
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, metrics


def evaluate_distance_groups(distance_model, sample_list, data_graph_info, args, highway_context, leaf_of):
    """Evaluate test subsets to reveal whether errors come from local, cross-leaf, short, or long queries."""
    if not sample_list:
        return {}

    distance_model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        batch_size = max(1, args.batch_size)
        for start in range(0, len(sample_list), batch_size):
            chunk = sample_list[start:start + batch_size]
            batch_inputs = [
                build_synthetic_partition_inputs(
                    graph_info=data_graph_info,
                    sample=sample,
                    k_highway=max(1, args.highway_k),
                    feature_dim=args.in_feat,
                    device=args.device,
                    external_highway_context=highway_context,
                    inner_mode=args.inner_mode,
                )
                for sample in chunk
            ]
            preds = distance_model.forward_batch(batch_inputs)
            y_true = torch.tensor([s["distance"] for s in chunk], dtype=torch.float, device=args.device)
            all_pred.extend(float(v) for v in preds.detach().cpu().view(-1))
            all_true.extend(float(v) for v in y_true.detach().cpu().view(-1))

    groups = {
        "same_leaf": [],
        "cross_leaf": [],
        "short_dist": [],
        "mid_dist": [],
        "long_dist": [],
    }
    sorted_dist = sorted(all_true)
    q1 = sorted_dist[len(sorted_dist) // 3]
    q2 = sorted_dist[(2 * len(sorted_dist)) // 3]

    for idx, sample in enumerate(sample_list):
        s, t = int(sample["s"]), int(sample["t"])
        pred = all_pred[idx]
        true = all_true[idx]
        if leaf_of.get(s) == leaf_of.get(t):
            groups["same_leaf"].append((true, pred))
        else:
            groups["cross_leaf"].append((true, pred))
        if true <= q1:
            groups["short_dist"].append((true, pred))
        elif true <= q2:
            groups["mid_dist"].append((true, pred))
        else:
            groups["long_dist"].append((true, pred))

    out = {}
    for name, pairs in groups.items():
        if not pairs:
            continue
        y_true = torch.tensor([p[0] for p in pairs], dtype=torch.float)
        y_pred = torch.tensor([p[1] for p in pairs], dtype=torch.float)
        metric = compute_distance_metrics(y_true, y_pred)
        metric["count"] = len(pairs)
        out[name] = metric
    return out


def _select_query_batches(sample_list, batch_size):
    for start in range(0, len(sample_list), batch_size):
        yield slice(start, start + batch_size)


def _group_metrics_from_preds(sample_list, all_pred, all_true, leaf_of):
    """把 (真值, 预测) 按同/跨分区、短/中/长距离分组，返回各组 mae/rmse/relative_error。

    三段式与单 GNN 两条分支共用，保证消融矩阵的分组口径完全一致。
    """
    groups = {"same_leaf": [], "cross_leaf": [], "short_dist": [], "mid_dist": [], "long_dist": []}
    if not sample_list:
        return {}
    sorted_dist = sorted(all_true)
    q1 = sorted_dist[len(sorted_dist) // 3]
    q2 = sorted_dist[(2 * len(sorted_dist)) // 3]
    for idx, sample in enumerate(sample_list):
        s, t = int(sample["s"]), int(sample["t"])
        pred, true = all_pred[idx], all_true[idx]
        (groups["same_leaf"] if leaf_of.get(s) == leaf_of.get(t) else groups["cross_leaf"]).append((true, pred))
        if true <= q1:
            groups["short_dist"].append((true, pred))
        elif true <= q2:
            groups["mid_dist"].append((true, pred))
        else:
            groups["long_dist"].append((true, pred))
    out = {}
    for name, pairs in groups.items():
        if not pairs:
            continue
        metric = compute_distance_metrics(
            torch.tensor([p[0] for p in pairs], dtype=torch.float),
            torch.tensor([p[1] for p in pairs], dtype=torch.float),
        )
        metric["count"] = len(pairs)
        out[name] = metric
    return out


def run_single_gnn_epoch(model, sample_list, euclid_feats, full_x, full_edge_index, criterion,
                         optimizer=None, head_batch_size=8192, step_samples=8192):
    """architecture=single_gnn 分支的一轮前向/反向。

    与三段式"逐样本重建分区子图 + 高速图"不同，这里对**整张地形图只做一次消息传递**：

    - 训练：h = encode(全图) → 逐批只跑融合头得到 loss，用 autograd.grad 累积 d loss/d h，
      最后一次性 h.backward(累积梯度) 更新 GNN 参数。图梯度只算一次，因此该分支
      不会因为"没有分区/highway 可复用"而被人为拖慢。
    - 评估：no_grad 下 encode 一次，再分批过融合头。

    注意：该分支**不使用** highway 上下文，也不使用分区子图；分区信息只用于评估分组。
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    empty = float("nan"), {"mae": float("nan"), "rmse": float("nan"), "relative_error": float("nan")}
    if not sample_list:
        return empty

    device = full_x.device
    s_idx_all = torch.tensor([int(x["s"]) for x in sample_list], dtype=torch.long, device=device)
    t_idx_all = torch.tensor([int(x["t"]) for x in sample_list], dtype=torch.long, device=device)
    y_all = torch.tensor([float(x["distance"]) for x in sample_list], dtype=torch.float, device=device)

    all_pred = []
    total_loss = 0.0
    num_batches = 0
    bs = max(1, int(head_batch_size))
    if not step_samples or int(step_samples) <= 0:
        # 默认与三段式对齐：每个 batch 一次优化步，保证两条分支每轮的**优化步数**一致
        step_samples = bs

    if is_train:
        n = len(sample_list)
        step_span = max(bs, int(step_samples))
        # 每个 optimizer step 覆盖 step_span 个样本：整图重新前向一次（保证图上梯度与当前参数一致），
        # 期间只跑便宜的融合头并累积 d loss/d h，最后一次性把图梯度回传。
        step_ranges = [(a, min(a + step_span, n)) for a in range(0, n, step_span)]
        # 融合头的参数必须**显式**求梯度：autograd.grad(loss, h) 只返回对 h 的梯度，
        # 不会写进 fusion_mlp 参数的 .grad。因此这里一次性对 [h] + 头参数求梯度，
        # 头参数梯度直接累积下来，h 的梯度再回传一次到 GNN（图只反传一次）。
        head_params = list(model.head_parameters()) if hasattr(model, "head_parameters") else list(model.fusion_mlp.parameters())
        for a, b in step_ranges:
            h = model.encode(full_x, full_edge_index)
            grad_h = None
            grad_head = [None] * len(head_params)
            for sl in _select_query_batches(sample_list[a:b], bs):
                lo, hi = a + sl.start, a + sl.stop
                preds = model.predict_from_embeddings(h[s_idx_all[lo:hi]], h[t_idx_all[lo:hi]], euclid_feats[lo:hi])
                loss = criterion(preds, y_all[lo:hi])
                grads = torch.autograd.grad(loss, [h] + head_params, retain_graph=False)
                grad_h = grads[0] if grad_h is None else grad_h + grads[0]
                for i in range(len(head_params)):
                    grad_head[i] = grads[i + 1] if grad_head[i] is None else grad_head[i] + grads[i + 1]
                total_loss += float(loss.item())
                num_batches += 1
                all_pred.append(preds.detach())
            optimizer.zero_grad()
            torch.autograd.backward([h], [grad_h])   # h 的图只反传一次 -> GNN 参数
            for p, gp in zip(head_params, grad_head):
                p.grad = gp                          # 融合头参数梯度
            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    else:
        with torch.no_grad():
            h = model.encode(full_x, full_edge_index)
            for sl in _select_query_batches(sample_list, bs):
                preds = model.predict_from_embeddings(h[s_idx_all[sl]], h[t_idx_all[sl]], euclid_feats[sl])
                loss = criterion(preds, y_all[sl])
                total_loss += float(loss.item())
                num_batches += 1
                all_pred.append(preds)

    pred_tensor = torch.cat(all_pred).view(-1) if all_pred else torch.empty(0, device=device)
    metrics = compute_distance_metrics(y_all, pred_tensor)
    return total_loss / max(1, num_batches), metrics


def evaluate_single_gnn_groups(model, sample_list, euclid_feats, full_x, full_edge_index, leaf_of,
                               head_batch_size=8192):
    """single_gnn 分支的分组评估（与 evaluate_distance_groups 同口径）。"""
    if not sample_list:
        return {}
    model.eval()
    device = full_x.device
    s_idx_all = torch.tensor([int(x["s"]) for x in sample_list], dtype=torch.long, device=device)
    t_idx_all = torch.tensor([int(x["t"]) for x in sample_list], dtype=torch.long, device=device)
    all_pred, all_true = [], []
    with torch.no_grad():
        h = model.encode(full_x, full_edge_index)
        for sl in _select_query_batches(sample_list, max(1, int(head_batch_size))):
            preds = model.predict_from_embeddings(h[s_idx_all[sl]], h[t_idx_all[sl]], euclid_feats[sl])
            all_pred.extend(float(v) for v in preds.detach().cpu().view(-1))
    all_true = [float(x["distance"]) for x in sample_list]
    return _group_metrics_from_preds(sample_list, all_pred, all_true, leaf_of)


def save_run_params(file_path, args):
    with open(file_path, "w", encoding="utf-8") as f:
        for key in sorted(vars(args).keys()):
            f.write(f"{key}: {getattr(args, key)}\n")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        print(f"[distance] torch intra-op threads = {args.torch_threads}")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        args.device = "cpu"

    project_root = os.path.dirname(os.path.abspath(__file__))
    base_folder = args.file_folder if os.path.isabs(args.file_folder) else os.path.join(project_root, args.file_folder)
    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(base_folder, args.off_file)
    base_name = os.path.splitext(os.path.basename(off_path))[0]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f"{base_name}_{args.run_tag}" if args.run_tag else f"{base_name}_distance_{current_time}"
    model_save_name = save_name + ".pt"
    result_save_name = save_name + ".txt"
    params_save_name = save_name + ".txt"

    model_save_path = os.path.join(project_root, "outputs", "models")
    result_save_path = os.path.join(project_root, "outputs", "results")
    params_save_path = os.path.join(project_root, "outputs", "params")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)
    os.makedirs(params_save_path, exist_ok=True)
    save_run_params(os.path.join(params_save_path, params_save_name), args)

    # .off → 全局图 + 四叉树分区 + 高速上下文（带磁盘缓存，避免每次重算）
    cache_dir = None if args.no_cache else (
        args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(project_root, args.cache_dir)
    )
    _t_prep_start = time.time()
    if args.architecture == "single_gnn" or args.architecture in BASELINE_ARCHITECTURES:
        # 消融/对照分支：只用全局图 + 四叉树分区（分区仅用于分组评估），
        # 完全不构建 highway 骨架，也就不付 K 次全图 Dijkstra 的预处理代价。
        # 三篇 baseline（gegnn/neurogf/litege）同样不需要 highway 上下文。
        data_graph_info, coords, leaf_of, num_leaves = build_graph_and_partition(
            off_path=off_path,
            max_depth=args.max_depth,
            capacity=args.capacity,
            adaptive=not args.uniform,
        )
        highway_context = None
        print(
            f"[distance] off={off_path} |V|={len(data_graph_info[0])} "
            f"leaves={num_leaves} (architecture=single_gnn: 无 highway 网络，跳过高速预处理)"
        )
    else:
        data_graph_info, coords, leaf_of, num_leaves, highway_context = build_pipeline_inputs_cached(
            off_path=off_path,
            max_depth=args.max_depth,
            capacity=args.capacity,
            adaptive=not args.uniform,
            weighted=True,
            feature_dim=args.in_feat,
            device=args.device,
            cache_dir=cache_dir,
            transit_k=args.transit_k,
        )
        print(
            f"[distance] off={off_path} |V|={len(data_graph_info[0])} "
            f"leaves={num_leaves}(occupied={highway_context['num_leaves_occupied']}) "
            f"highway_nodes={len(highway_context['highway_global_ids'])}"
        )
    vertices3d, _ = load_off(off_path)
    if highway_context is not None:
        highway_context["node_coords3d"] = {i: vertices3d[i] for i in range(len(vertices3d))}
        # 预计算每节点最近 k 个高速入口（一次性，替代每样本每轮的 O(K·logK) 排序）
        highway_context["nearest_k_local"] = precompute_nearest_k(
            highway_context["access_dist"], k_max=max(16, args.highway_k)
        )
    preprocess_seconds = time.time() - _t_prep_start
    print(f"[distance] preprocess done in {preprocess_seconds:.2f}s")

    # 采样结果缓存路径（同图同参数第二次起直接读，跳过全部采样 Dijkstra）
    samples_cache_path = None
    if cache_dir is not None:
        _fp = _file_fingerprint(off_path)
        if args.sample_strategy == "random":
            sample_cache_name = f"{base_name}_samples_n{args.distance_samples}_seed{args.seed}_{_fp}.csv"
        else:
            mode = "uni" if args.uniform else "ada"
            sample_cache_name = (
                f"{base_name}_samples_{args.sample_strategy}_n{args.distance_samples}_"
                f"d{args.max_depth}_c{args.capacity}_{mode}_tk{args.transit_k}_seed{args.seed}_{_fp}.csv"
            )
        samples_cache_path = os.path.join(cache_dir, sample_cache_name)
    if args.labels_file:
        labels_path = args.labels_file if os.path.isabs(args.labels_file) else os.path.join(project_root, args.labels_file)
        distance_samples = load_label_pairs_csv(labels_path, value_col=args.labels_col)
        print(f"[distance] 使用外部标签 {labels_path} (列={args.labels_col})："
              f"{len(distance_samples)} 对（跳过内置采样/Dijkstra）")
    else:
        distance_samples = build_distance_samples(
            graph_info=data_graph_info,
            num_samples=(None if args.distance_samples <= 0 else args.distance_samples),
            weighted=True,
            seed=args.seed,
            undirected=True,
            cache_path=samples_cache_path,
            sample_strategy=args.sample_strategy,
            leaf_of=leaf_of,
        )
    train_samples, val_samples, test_samples = split_distance_dataset(
        sample_list=distance_samples,
        train_ratio=args.train_percent,
        val_ratio=0.1,
        seed=args.seed,
    )
    print(
        f"[distance] samples: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    # 导出 train/val/test 查询点对 + 真实距离（用于审计、对齐基线、8:1:1 划分与泄漏检查）
    def _dump_pairs(split_name, samples):
        path = os.path.join(result_save_path, f"{save_name}_{split_name}_pairs.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("s,t,true_distance\n")
            for smp in samples:
                f.write(f"{int(smp['s'])},{int(smp['t'])},{float(smp['distance'])}\n")
        print(f"[distance] wrote {split_name} pairs: {path} ({len(samples)} pairs)")
        return path

    _dump_pairs("train", train_samples)
    _dump_pairs("val", val_samples)
    test_pairs_path = _dump_pairs("test", test_samples)  # baseline.py 用它对齐 test 集

    if args.architecture in BASELINE_ARCHITECTURES:
        # 三篇 baseline：按各自官方仓库的网络定义构造，并统一成同一套输出参数化选项
        _b_mode = args.prediction_mode
        if _b_mode == "direct":
            _b_mode = "softplus"            # direct 在 baseline 上对应 softplus(raw)
        elif _b_mode == "highway_residual":
            _b_mode = "native"
        _base_out = "native" if args.prediction_mode == "native" else _b_mode
        distance_model = build_baseline_model(
            architecture=args.architecture, prediction_mode=_base_out,
            in_dim=(6 if args.architecture == "gegnn" else 3),
            dropout=args.dropout_ratio, litege_pca_dim=args.litege_pca_dim,
            gegnn_layers=args.gegnn_layers,
        ).to(args.device)
        print(f"[distance] architecture={args.architecture}(baseline) output={_base_out} "
              f"params={sum(p.numel() for p in distance_model.parameters())}")
    else:
        distance_model = build_distance_model(
            architecture=args.architecture,
            prediction_mode=args.prediction_mode,
            node_feat_dim=args.in_feat,
            highway_feat_dim=args.in_feat,
            global_feat_dim=2,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            fusion_hidden_dim=args.hidden_dim,
            dropout=args.dropout_ratio,
            use_highway_distance_feature=not args.disable_highway_distance_feature,
            highway_distance_feat_dim=4,
            single_gnn_layers=args.single_gnn_layers,
        ).to(args.device)
        print(f"[distance] architecture={args.architecture} prediction_mode={args.prediction_mode} "
              f"params={sum(p.numel() for p in distance_model.parameters())}")
    # 整图张量只建一次；每个样本的 log1p(3D 欧氏距离) 也只算一次
    full_x = full_edge_index = None
    train_euclid = val_euclid = test_euclid = None
    if args.architecture in BASELINE_ARCHITECTURES:
        full_x, full_edge_index, edge_len, pos = build_point_feature_tensors(
            data_graph_info, vertices3d, device=args.device,
            with_normals=(args.architecture == "gegnn"),
        )
        distance_model.setup_graph(pos, edge_len)
        if args.architecture == "litege":
            _pca_cache = os.path.join(project_root, "outputs", "cache",
                                      f"{base_name}_udf_pca{args.litege_pca_dim}.npz")
            node_pca = compute_udf_pca_descriptor(
                vertices3d, pca_dim=args.litege_pca_dim, cache_path=_pca_cache)
            distance_model.set_node_pca(
                torch.tensor(node_pca, dtype=torch.float, device=args.device))
        train_euclid = build_pair_euclidean_features(train_samples, vertices3d, args.device)
        val_euclid = build_pair_euclidean_features(val_samples, vertices3d, args.device)
        test_euclid = build_pair_euclidean_features(test_samples, vertices3d, args.device)
        print(f"[distance] {args.architecture} tensors: x={tuple(full_x.shape)} "
              f"edge_index={tuple(full_edge_index.shape)}")
    elif args.architecture == "single_gnn":
        full_x, full_edge_index = build_full_graph_tensors(
            data_graph_info, coords, feature_dim=args.in_feat, device=args.device
        )
        train_euclid = build_pair_euclidean_features(train_samples, vertices3d, args.device)
        val_euclid = build_pair_euclidean_features(val_samples, vertices3d, args.device)
        test_euclid = build_pair_euclidean_features(test_samples, vertices3d, args.device)
        print(f"[distance] single_gnn full graph tensors: x={tuple(full_x.shape)} "
              f"edge_index={tuple(full_edge_index.shape)}")

    distance_optimizer = torch.optim.Adam(
        distance_model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            distance_optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        print(
            f"[distance] lr_scheduler=plateau (factor={args.lr_factor}, "
            f"patience={args.lr_patience}, min_lr={args.min_lr})"
        )
    criterion = build_loss(args.loss_type)

    best_val_metric = float("inf")
    best_val_mae = float("inf")
    best_state = None
    no_improve_epochs = 0
    best_epoch = -1

    _t_train_start = time.time()
    for epoch in range(args.num_epoch):
        _t_epoch_start = time.time()
        if args.architecture == "single_gnn" or args.architecture in BASELINE_ARCHITECTURES:
            _, train_metrics = run_single_gnn_epoch(
                model=distance_model, sample_list=train_samples, euclid_feats=train_euclid,
                full_x=full_x, full_edge_index=full_edge_index, criterion=criterion,
                optimizer=distance_optimizer, head_batch_size=args.single_gnn_head_batch,
                step_samples=args.single_gnn_step_samples,
            )
            _, val_metrics = run_single_gnn_epoch(
                model=distance_model, sample_list=val_samples, euclid_feats=val_euclid,
                full_x=full_x, full_edge_index=full_edge_index, criterion=criterion,
                optimizer=None, head_batch_size=args.single_gnn_head_batch,
            )
        else:
            _, train_metrics = run_distance_epoch(
                distance_model=distance_model,
                sample_list=train_samples,
                data_graph_info=data_graph_info,
                args=args,
                highway_context=highway_context,
                criterion=criterion,
                optimizer=distance_optimizer,
            )
            _, val_metrics = run_distance_epoch(
                distance_model=distance_model,
                sample_list=val_samples,
                data_graph_info=data_graph_info,
                args=args,
                highway_context=highway_context,
                criterion=criterion,
                optimizer=None,
            )
        epoch_seconds = time.time() - _t_epoch_start
        cur_lr = distance_optimizer.param_groups[0]["lr"]
        print(
            f"[distance] epoch={epoch:03d} "
            f"train_mae={train_metrics['mae']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"train_rel={train_metrics['relative_error']:.6f} val_rel={val_metrics['relative_error']:.6f} "
            f"lr={cur_lr:.2e} time={epoch_seconds:.2f}s"
        )

        # 学习率调度：按选择的验证指标触发降 LR（下一轮生效）
        if scheduler is not None:
            scheduler.step(val_metrics[args.selection_metric])
            new_lr = distance_optimizer.param_groups[0]["lr"]
            if new_lr < cur_lr:
                print(f"[distance] lr reduced: {cur_lr:.2e} -> {new_lr:.2e}")

        cur_val_metric = val_metrics[args.selection_metric]
        if cur_val_metric < best_val_metric:
            best_val_metric = cur_val_metric
            best_val_mae = val_metrics["mae"]
            no_improve_epochs = 0
            best_epoch = epoch
            best_state = copy.deepcopy(distance_model.state_dict())
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[distance] early stop at epoch={epoch}, "
                f"best_epoch={best_epoch}, best_val_{args.selection_metric}={best_val_metric:.6f}"
            )
            break

    train_seconds = time.time() - _t_train_start

    if best_state is not None:
        distance_model.load_state_dict(best_state)

    if args.architecture == "single_gnn" or args.architecture in BASELINE_ARCHITECTURES:
        _, test_metrics = run_single_gnn_epoch(
            model=distance_model, sample_list=test_samples, euclid_feats=test_euclid,
            full_x=full_x, full_edge_index=full_edge_index, criterion=criterion,
            optimizer=None, head_batch_size=args.single_gnn_head_batch,
        )
        group_metrics = evaluate_single_gnn_groups(
            model=distance_model, sample_list=test_samples, euclid_feats=test_euclid,
            full_x=full_x, full_edge_index=full_edge_index, leaf_of=leaf_of,
            head_batch_size=args.single_gnn_head_batch,
        )
    else:
        _, test_metrics = run_distance_epoch(
            distance_model=distance_model,
            sample_list=test_samples,
            data_graph_info=data_graph_info,
            args=args,
            highway_context=highway_context,
            criterion=criterion,
            optimizer=None,
        )
        group_metrics = evaluate_distance_groups(
            distance_model=distance_model,
            sample_list=test_samples,
            data_graph_info=data_graph_info,
            args=args,
            highway_context=highway_context,
            leaf_of=leaf_of,
        )
    print(
        f"[distance] test_mae={test_metrics['mae']:.6f}, "
        f"test_rmse={test_metrics['rmse']:.6f}, "
        f"test_relative_error={test_metrics['relative_error']:.6f}"
    )
    for group_name, metric in group_metrics.items():
        print(
            f"[distance][group] {group_name} count={metric['count']} "
            f"mae={metric['mae']:.6f} rmse={metric['rmse']:.6f} "
            f"relative_error={metric['relative_error']:.6f}"
        )
    print(
        f"[distance] timing: preprocess={preprocess_seconds:.2f}s, "
        f"train={train_seconds:.2f}s, best_epoch={best_epoch}"
    )

    torch.save(distance_model.state_dict(), os.path.join(model_save_path, model_save_name))
    with open(os.path.join(result_save_path, result_save_name), "w", encoding="utf-8") as f:
        f.write("metric value\n")
        f.write(f"architecture {args.architecture}\n")
        f.write(f"prediction_mode {args.prediction_mode}\n")
        f.write(f"loss_type {args.loss_type}\n")
        f.write(f"seed {args.seed}\n")
        f.write(f"best_val_mae {best_val_mae}\n")
        if args.selection_metric != "mae":
            f.write(f"best_val_{args.selection_metric} {best_val_metric}\n")
        f.write(f"best_epoch {best_epoch}\n")
        f.write(f"test_mae {test_metrics['mae']}\n")
        f.write(f"test_rmse {test_metrics['rmse']}\n")
        f.write(f"test_relative_error {test_metrics['relative_error']}\n")
        for group_name, metric in group_metrics.items():
            f.write(f"group_{group_name}_count {metric['count']}\n")
            f.write(f"group_{group_name}_mae {metric['mae']}\n")
            f.write(f"group_{group_name}_rmse {metric['rmse']}\n")
            f.write(f"group_{group_name}_relative_error {metric['relative_error']}\n")
        f.write(f"preprocess_seconds {preprocess_seconds}\n")
        f.write(f"train_seconds {train_seconds}\n")
