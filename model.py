# 模型包装与评估：DistanceRegressionNet（三段式距离回归）+ 距离指标 compute_distance_metrics。
import torch
import torch.nn as nn
from gnn import DistancePredictor as ThreeStageDistancePredictor
from gnn import SingleGNNPredictor


class DistanceRegressionNet(nn.Module):
    """
    三段式距离回归模型包装器，复用 gnn.py 中的 DistancePredictor。
    """

    def __init__(
        self,
        node_feat_dim,
        highway_feat_dim,
        global_feat_dim,
        hidden_dim=64,
        inner_out_dim=64,
        inter_out_dim=64,
        fusion_hidden_dim=128,
        num_inner_layers=2,
        num_inter_layers=2,
        dropout=0.1,
        use_highway_distance_feature=True,
        highway_distance_feat_dim=4,
        prediction_mode="direct",
    ):
        super().__init__()
        self.backbone = ThreeStageDistancePredictor(
            node_feat_dim=node_feat_dim,
            highway_feat_dim=highway_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            inner_out_dim=inner_out_dim,
            inter_out_dim=inter_out_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            num_inner_layers=num_inner_layers,
            num_inter_layers=num_inter_layers,
            dropout=dropout,
            use_highway_distance_feature=use_highway_distance_feature,
            highway_distance_feat_dim=highway_distance_feat_dim,
            prediction_mode=prediction_mode,
        )

    def forward(
        self,
        x_s,
        edge_index_s,
        s_idx,
        x_t,
        edge_index_t,
        t_idx,
        x_highway,
        edge_index_highway,
        s_global_feat,
        t_global_feat,
        s_connect_idx,
        t_connect_idx,
        highway_dist_feat=None,
        euclidean_dist_feat=None,
        return_aux=False,
    ):
        return self.backbone(
            x_s=x_s,
            edge_index_s=edge_index_s,
            s_idx=s_idx,
            x_t=x_t,
            edge_index_t=edge_index_t,
            t_idx=t_idx,
            x_highway=x_highway,
            edge_index_highway=edge_index_highway,
            s_global_feat=s_global_feat,
            t_global_feat=t_global_feat,
            s_connect_idx=s_connect_idx,
            t_connect_idx=t_connect_idx,
            highway_dist_feat=highway_dist_feat,
            euclidean_dist_feat=euclidean_dist_feat,
            return_aux=return_aux,
        )

    def forward_batch(self, samples):
        """批量前向，透传到 backbone；samples 为单样本输入字典的列表，返回 [B] 预测距离。"""
        return self.backbone.forward_batch(samples)


def build_distance_model(
    architecture,
    prediction_mode,
    node_feat_dim,
    highway_feat_dim,
    global_feat_dim,
    hidden_dim=64,
    out_dim=32,
    fusion_hidden_dim=128,
    dropout=0.1,
    use_highway_distance_feature=True,
    highway_distance_feat_dim=4,
    single_gnn_layers=3,
):
    """按消融配置构造模型。

    - architecture="three_stage"（默认主线）：四叉树分区子图 + highway 骨架 + Inner/Inter 双 GNN + Fusion。
    - architecture="single_gnn"（消融）：不使用地形分区、不使用 highway 网络，只在全图上跑一个 GNN。
      两条分支的 prediction_mode（direct / euclidean_residual）与输出参数化完全一致，
      因此任意两次运行之间的差异只来自被消融掉的结构组件。
    """
    if architecture == "three_stage":
        return DistanceRegressionNet(
            node_feat_dim=node_feat_dim,
            highway_feat_dim=highway_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            inner_out_dim=out_dim,
            inter_out_dim=out_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            use_highway_distance_feature=use_highway_distance_feature,
            highway_distance_feat_dim=highway_distance_feat_dim,
            prediction_mode=prediction_mode,
        )
    if architecture == "single_gnn":
        return SingleGNNPredictor(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=single_gnn_layers,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            prediction_mode=prediction_mode,
        )
    raise ValueError(f"unknown architecture: {architecture}")


def compute_distance_metrics(y_true, y_pred, eps=1e-9):
    """
    Args:
        y_true (torch.Tensor): [N] 真实最短路距离
        y_pred (torch.Tensor): [N] 预测距离
    Returns:
        dict: {"mae", "rmse", "relative_error"}
    """
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()
    abs_err = torch.abs(y_pred - y_true)
    mae = abs_err.mean()
    rmse = torch.sqrt(((y_pred - y_true) ** 2).mean())
    relative_error = (abs_err / (y_true + eps)).mean()
    return {
        "mae": mae.item(),
        "rmse": rmse.item(),
        "relative_error": relative_error.item(),
    }
