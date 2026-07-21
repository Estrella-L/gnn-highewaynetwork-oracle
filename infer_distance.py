# 推理入口：加载训练好的模型，预测单对顶点 (s,t) 的近似最短路距离。
import argparse
import os

import torch

from model import DistanceRegressionNet
from preprocess import build_synthetic_partition_inputs
from build_highway import build_pipeline_inputs_cached


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="trained model checkpoint path (.pt)")
    parser.add_argument("--off_file", type=str, required=True, help="input .off terrain mesh path (same as training)")
    parser.add_argument("--file_folder", type=str, default="data", help="base folder for relative --off_file paths (resolved under project root)")
    parser.add_argument("--s", type=int, required=True, help="source vertex id")
    parser.add_argument("--t", type=int, required=True, help="target vertex id")

    # 四叉树参数必须与训练时一致，否则分区/高速上下文不同
    parser.add_argument("--max_depth", type=int, default=3, help="quadtree max depth (must match training)")
    parser.add_argument("--capacity", type=int, default=32, help="max points per leaf (must match training)")
    parser.add_argument("--uniform", action="store_true", help="uniform quadtree (must match training)")

    parser.add_argument("--in_feat", type=int, default=64, help="feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="output embedding dimension")
    parser.add_argument("--dropout_ratio", type=float, default=0.2, help="dropout ratio")
    parser.add_argument("--num_inner_layers", type=int, default=2, help="InnerGNN 层数（须与训练一致）")
    parser.add_argument("--num_inter_layers", type=int, default=4, help="InterGNN 层数（须与训练一致）")
    parser.add_argument("--highway_k", type=int, default=3, help="number of nearest highway nodes for s/t connections")
    parser.add_argument(
        "--inner_mode",
        type=str,
        default="partition",
        choices=["partition", "ego"],
        help="Inner-GNN subgraph mode (must match training)",
    )
    parser.add_argument(
        "--disable_highway_distance_feature",
        action="store_true",
        help="disable the highway-decomposition distance feature (must match training config)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--cache_dir", type=str, default="outputs/cache", help="高速上下文缓存目录（按项目根解析）")
    parser.add_argument("--no_cache", action="store_true", help="禁用缓存，每次重新计算分区+高速")
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU.")
        args.device = "cpu"

    project_root = os.path.dirname(os.path.abspath(__file__))
    base_folder = args.file_folder if os.path.isabs(args.file_folder) else os.path.join(project_root, args.file_folder)
    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(base_folder, args.off_file)
    cache_dir = None if args.no_cache else (
        args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(project_root, args.cache_dir)
    )
    graph_info, coords, leaf_of, num_leaves, highway_context = build_pipeline_inputs_cached(
        off_path=off_path,
        max_depth=args.max_depth,
        capacity=args.capacity,
        adaptive=not args.uniform,
        weighted=True,
        feature_dim=args.in_feat,
        device=args.device,
        cache_dir=cache_dir,
    )

    n_nodes = len(graph_info[0])
    if not (0 <= args.s < n_nodes and 0 <= args.t < n_nodes):
        raise ValueError(f"s/t out of range: valid vertex id should be within [0, {n_nodes - 1}]")

    model = DistanceRegressionNet(
        node_feat_dim=args.in_feat,
        highway_feat_dim=args.in_feat,
        global_feat_dim=2,
        hidden_dim=args.hidden_dim,
        inner_out_dim=args.out_dim,
        inter_out_dim=args.out_dim,
        fusion_hidden_dim=args.hidden_dim,
        num_inner_layers=args.num_inner_layers,
        num_inter_layers=args.num_inter_layers,
        dropout=args.dropout_ratio,
        use_highway_distance_feature=not args.disable_highway_distance_feature,
        highway_distance_feat_dim=4,
    ).to(args.device)

    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    sample = {"s": args.s, "t": args.t, "distance": 0.0}
    inputs = build_synthetic_partition_inputs(
        graph_info=graph_info,
        sample=sample,
        k_highway=max(1, args.highway_k),
        feature_dim=args.in_feat,
        device=args.device,
        external_highway_context=highway_context,
        inner_mode=args.inner_mode,
    )

    with torch.no_grad():
        pred = model(**inputs)
    print(f"predicted_distance({args.s}->{args.t}) = {float(pred.item()):.6f}")


if __name__ == "__main__":
    main()
