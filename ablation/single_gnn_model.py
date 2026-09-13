# -*- coding: utf-8 -*-
"""单 GNN 消融分支（A2/A3）：不使用四叉树地形分区、不使用 highway 网络。

对照关系（只改结构，不改输出参数化）：
  - 表征：整张地形网格图上跑一个 GraphSAGE，取 s、t 节点嵌入；
  - 融合头输入与三段式同维度 [h_s | h_t | |h_s-h_t| | h_s*h_t]（+ 欧氏先验），保证头容量可比；
  - 输出层与三段式一致：direct -> softplus(raw)；euclidean_residual -> d_3D*(1+0.5*tanh(raw))。

逐字取自 gnn-euclidean-local/gnn.py 的 SingleGNNPredictor（未改动数学）。
"""
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn


class SingleGNNPredictor(nn.Module):
    """消融用「单 GNN」预测器：**不使用**四叉树地形分区，也**不使用** highway 网络。

    与三段式 DistancePredictor 的对照关系（消融实验要求只改结构、不改输出参数化）：

    - 表征：整张地形网格图上跑一个 GraphSAGE，取 s、t 两个节点的嵌入；
      不做分区子图（InnerGNN），不做高速骨架 + 虚拟节点（InterGNN）。
    - 融合头输入与三段式**同维度**：三段式是 [h_s_inner | h_t_inner | h_s_inter | h_t_inter] = 4*out_dim；
      这里是 [h_s | h_t | |h_s-h_t| | h_s⊙h_t] = 4*out_dim，保证 MLP 头容量可比（不靠"砍容量"制造差距）。
    - 输出层与三段式逐字一致：
        direct            -> softplus(raw)
        euclidean_residual-> d_3D * (1 + 0.5*tanh(raw))，并把 log1p(d_3D) 拼进融合头输入
      因此 A2/A3 与 M0/A1 的差别只来自「有没有地形分区 + highway 网络」。
    """

    def __init__(
        self,
        node_feat_dim,
        hidden_dim=64,
        out_dim=32,
        num_layers=3,
        fusion_hidden_dim=128,
        dropout=0.1,
        prediction_mode="direct",
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for SingleGNNPredictor.")
        if prediction_mode not in {"direct", "highway_residual", "euclidean_residual"}:
            raise ValueError(f"unknown prediction_mode: {prediction_mode}")
        self.prediction_mode = prediction_mode
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(geo_nn.SAGEConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(geo_nn.SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(geo_nn.SAGEConv(hidden_dim, out_dim))
        self.act = nn.ReLU()

        euclidean_prior_dim = 1 if prediction_mode == "euclidean_residual" else 0
        head_in_dim = 4 * out_dim + euclidean_prior_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(head_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )
        self.output_activation = nn.Softplus()

    def head_parameters(self):
        return self.fusion_mlp.parameters()

    def encode(self, x, edge_index):
        """整图一次消息传递，返回全部节点嵌入 [N, out_dim]。"""
        h = x
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer_idx != len(self.convs) - 1:
                h = self.act(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h

    def _pair_repr(self, h_s, h_t):
        return torch.cat([h_s, h_t, (h_s - h_t).abs(), h_s * h_t], dim=-1)

    def predict_from_embeddings(self, h_s, h_t, euclidean_dist_feat=None):
        """由 s/t 节点嵌入直接得到预测距离 [B]。

        Args:
            h_s / h_t (Tensor): [B, out_dim]
            euclidean_dist_feat (Tensor | None): [B] 或 [B,1]，log1p(3D 欧氏距离)
        """
        parts = [self._pair_repr(h_s, h_t)]
        edf = None
        if self.prediction_mode == "euclidean_residual" and euclidean_dist_feat is not None:
            edf = euclidean_dist_feat.to(h_s.device).view(-1, 1)
            parts.append(edf)
        fusion_input = torch.cat(parts, dim=-1)
        raw = self.fusion_mlp(fusion_input).view(-1)
        if self.prediction_mode == "euclidean_residual" and edf is not None:
            euclidean_base = torch.expm1(edf.view(-1)).clamp_min(1e-6)
            correction = 0.5 * torch.tanh(raw)
            return (euclidean_base * (1.0 + correction)).clamp_min(1e-6)
        return self.output_activation(raw)


if __name__ == "__main__":
    torch.manual_seed(7)
    node_feat_dim = 16
    highway_feat_dim = 16
    global_feat_dim = 8

    x_s = torch.randn(12, node_feat_dim)
    edge_index_s = torch.randint(0, 12, (2, 40), dtype=torch.long)
    s_idx = torch.tensor(3, dtype=torch.long)

    x_t = torch.randn(10, node_feat_dim)
    edge_index_t = torch.randint(0, 10, (2, 34), dtype=torch.long)
    t_idx = torch.tensor(6, dtype=torch.long)

    x_highway = torch.randn(20, highway_feat_dim)
    edge_index_highway = torch.randint(0, 20, (2, 72), dtype=torch.long)
    s_global_feat = torch.randn(global_feat_dim)
    t_global_feat = torch.randn(global_feat_dim)
    s_connect_idx = torch.tensor([0, 4, 8], dtype=torch.long)
    t_connect_idx = torch.tensor([11, 15, 19], dtype=torch.long)

    model = DistancePredictor(
        node_feat_dim=node_feat_dim,
        highway_feat_dim=highway_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=64,
        inner_out_dim=64,
        inter_out_dim=64,
        use_highway_distance_feature=True,
        highway_distance_feat_dim=4,
        prediction_mode="direct",
    )
    highway_dist_feat = torch.log1p(torch.tensor([2.0, 5.0, 3.0, 10.0]))
    pred, aux = model(
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
        return_aux=True,
    )

    print("h_s_inner shape:", aux["h_s_inner"].shape)
    print("h_t_inner shape:", aux["h_t_inner"].shape)
    print("h_st_inter shape:", aux["h_st_inter"].shape)
    print("y_hat shape:", pred.shape, "| y_hat:", pred)
