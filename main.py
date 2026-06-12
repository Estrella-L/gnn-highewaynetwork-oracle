import argparse
import os
import time

import torch
import torch.nn as nn

from preprocess import (
    build_distance_samples,
    split_distance_dataset,
    build_synthetic_partition_inputs,
)
from build_highway import build_pipeline_inputs
from model import DistanceRegressionNet, compute_distance_metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="distance", choices=["distance"], help="only distance task is supported")
    parser.add_argument("--off_file", type=str, required=True, help="input .off terrain mesh path")
    parser.add_argument("--file_folder", type=str, default="./", help="base folder for relative paths")

    # 四叉树分区参数（对齐 EAR-Oracle 的 grid/quadtree）
    parser.add_argument("--max_depth", type=int, default=3, help="quadtree max depth")
    parser.add_argument("--capacity", type=int, default=32, help="max points per leaf (adaptive mode)")
    parser.add_argument("--uniform", action="store_true", help="uniform quadtree instead of adaptive")

    parser.add_argument("--in_feat", type=int, default=64, help="input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="output embedding dimension")
    parser.add_argument("--dropout_ratio", type=float, default=0.2, help="dropout ratio")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epoch", type=int, default=20, help="max training epochs")
    parser.add_argument("--train_percent", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="early stop patience")
    parser.add_argument("--distance_samples", type=int, default=3000, help="cap on number of (s,t) pairs; <=0 means use ALL unique reachable pairs")
    parser.add_argument("--highway_k", type=int, default=3, help="number of highway connectors for each endpoint")
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
        "--disable_highway_distance_feature",
        action="store_true",
        help="disable the highway-decomposition distance feature fed into the fusion MLP",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
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

    if is_train:
        distance_model.train()
    else:
        distance_model.eval()

    for sample in sample_list:
        model_inputs = build_synthetic_partition_inputs(
            graph_info=data_graph_info,
            sample=sample,
            k_highway=max(1, args.highway_k),
            feature_dim=args.in_feat,
            device=args.device,
            external_highway_context=highway_context,
            inner_mode=args.inner_mode,
        )
        if is_train:
            optimizer.zero_grad()
            pred = distance_model(**model_inputs)
        else:
            with torch.no_grad():
                pred = distance_model(**model_inputs)

        y_true = torch.tensor([sample["distance"]], dtype=torch.float, device=args.device)
        loss = criterion(pred, y_true)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        all_pred.append(pred.detach().view(-1))
        all_true.append(y_true.view(-1))

    pred_tensor = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0)
    true_tensor = torch.cat(all_true, dim=0) if all_true else torch.empty(0)
    metrics = compute_distance_metrics(true_tensor, pred_tensor) if len(sample_list) > 0 else {
        "mae": float("nan"),
        "rmse": float("nan"),
        "relative_error": float("nan"),
    }
    avg_loss = total_loss / max(1, len(sample_list))
    return avg_loss, metrics


def save_run_params(file_path, args):
    with open(file_path, "w", encoding="utf-8") as f:
        for key in sorted(vars(args).keys()):
            f.write(f"{key}: {getattr(args, key)}\n")


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.autograd.set_detect_anomaly(True)
    print(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        args.device = "cpu"

    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(args.file_folder, args.off_file)
    base_name = os.path.splitext(os.path.basename(off_path))[0]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f"{base_name}_distance_{current_time}"
    model_save_name = save_name + ".pt"
    result_save_name = save_name + ".txt"
    params_save_name = save_name + ".txt"

    model_save_path = "./saved_models"
    result_save_path = "./saved_results"
    params_save_path = "./saved_params"
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)
    os.makedirs(params_save_path, exist_ok=True)
    save_run_params(os.path.join(params_save_path, params_save_name), args)

    # .off → 全局图 + 四叉树分区 + 高速上下文
    data_graph_info, coords, leaf_of, num_leaves, highway_context = build_pipeline_inputs(
        off_path=off_path,
        max_depth=args.max_depth,
        capacity=args.capacity,
        adaptive=not args.uniform,
        weighted=True,
        feature_dim=args.in_feat,
        device=args.device,
    )
    print(
        f"[distance] off={off_path} |V|={len(data_graph_info[0])} "
        f"leaves={num_leaves}(occupied={highway_context['num_leaves_occupied']}) "
        f"highway_nodes={len(highway_context['highway_global_ids'])}"
    )

    distance_samples = build_distance_samples(
        graph_info=data_graph_info,
        num_samples=(None if args.distance_samples <= 0 else args.distance_samples),
        weighted=True,
        seed=42,
        undirected=True,
    )
    train_samples, val_samples, test_samples = split_distance_dataset(
        sample_list=distance_samples,
        train_ratio=args.train_percent,
        val_ratio=0.1,
        seed=42,
    )
    print(
        f"[distance] samples: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    distance_model = DistanceRegressionNet(
        node_feat_dim=args.in_feat,
        highway_feat_dim=args.in_feat,
        global_feat_dim=2,
        hidden_dim=args.hidden_dim,
        inner_out_dim=args.out_dim,
        inter_out_dim=args.out_dim,
        fusion_hidden_dim=args.hidden_dim,
        dropout=args.dropout_ratio,
        use_highway_distance_feature=not args.disable_highway_distance_feature,
        highway_distance_feat_dim=4,
    ).to(args.device)
    distance_optimizer = torch.optim.Adam(
        distance_model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4,
    )
    criterion = build_loss(args.loss_type)

    best_val_mae = float("inf")
    best_state = None
    no_improve_epochs = 0
    best_epoch = -1

    for epoch in range(args.num_epoch):
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
        print(
            f"[distance] epoch={epoch:03d} "
            f"train_mae={train_metrics['mae']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} val_rmse={val_metrics['rmse']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            no_improve_epochs = 0
            best_epoch = epoch
            best_state = distance_model.state_dict()
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[distance] early stop at epoch={epoch}, "
                f"best_epoch={best_epoch}, best_val_mae={best_val_mae:.6f}"
            )
            break

    if best_state is not None:
        distance_model.load_state_dict(best_state)

    _, test_metrics = run_distance_epoch(
        distance_model=distance_model,
        sample_list=test_samples,
        data_graph_info=data_graph_info,
        args=args,
        highway_context=highway_context,
        criterion=criterion,
        optimizer=None,
    )
    print(
        f"[distance] test_mae={test_metrics['mae']:.6f}, "
        f"test_rmse={test_metrics['rmse']:.6f}, "
        f"test_relative_error={test_metrics['relative_error']:.6f}"
    )

    torch.save(distance_model.state_dict(), os.path.join(model_save_path, model_save_name))
    with open(os.path.join(result_save_path, result_save_name), "w", encoding="utf-8") as f:
        f.write("metric value\n")
        f.write(f"best_val_mae {best_val_mae}\n")
        f.write(f"test_mae {test_metrics['mae']}\n")
        f.write(f"test_rmse {test_metrics['rmse']}\n")
        f.write(f"test_relative_error {test_metrics['relative_error']}\n")
