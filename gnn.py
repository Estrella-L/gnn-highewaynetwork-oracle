import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geo_nn


class GIN(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, out_dim, train_eps=True):
        super(GIN, self).__init__()
        # we can change the sequential nn
        nn_module_for_gin_1 = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU()
        )
        nn_module_for_gin_2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        self.GIN_layer_1 = geo_nn.GINConv(nn_module_for_gin_1, train_eps=train_eps)
        self.GIN_layer_2 = geo_nn.GINConv(nn_module_for_gin_2, train_eps=train_eps)

    def forward(self, in_feat, edge_list):
        x = self.GIN_layer_1(in_feat, edge_list)
        x = self.GIN_layer_2(x, edge_list)

        return x


class GAT(nn.Module):
    def __init__(self, input_feat_dim, out_dim, train_eps=True):
        super(GAT, self).__init__()
        # we can change the sequential nn
        self.GAT_layer = geo_nn.GATConv(input_feat_dim, out_dim, add_self_loops=False)

    def forward(self, in_feat, edge_list):
        x = self.GAT_layer(in_feat, edge_list)
        # x = self.GIN_layer_2(x, edge_list)

        return x


class InnerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for InnerGNN.")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(geo_nn.SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(geo_nn.SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(geo_nn.SAGEConv(hidden_dim, output_dim))
        self.act = nn.ReLU()

    def forward(self, x, edge_index, query_idx):
        """
        Args:
            x (torch.Tensor): [num_nodes, in_dim] 分区子图节点特征。
            edge_index (torch.LongTensor): [2, num_edges] 分区子图边索引。
            query_idx (int | torch.LongTensor): 查询节点索引。若为张量，形状可为 [B]。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - node_emb: [num_nodes, output_dim] 子图全部节点嵌入
                - query_emb: [B, output_dim] 或 [1, output_dim] 查询节点嵌入
        """
        h = x
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer_idx != len(self.convs) - 1:
                h = self.act(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        if isinstance(query_idx, int):
            query_idx = torch.tensor([query_idx], dtype=torch.long, device=h.device)
        elif query_idx.dim() == 0:
            query_idx = query_idx.view(1).to(h.device)
        else:
            query_idx = query_idx.to(h.device)
        query_emb = h[query_idx]
        return h, query_emb


class InterGNN(nn.Module):
    def __init__(self, highway_in_dim, hidden_dim, output_dim, global_feat_dim, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for InterGNN.")
        self.dropout = dropout
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, highway_in_dim),
        )
        self.convs = nn.ModuleList()
        self.convs.append(geo_nn.SAGEConv(highway_in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(geo_nn.SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(geo_nn.SAGEConv(hidden_dim, output_dim))
        self.act = nn.ReLU()
        self.readout = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def _build_virtual_edges(s_virtual_idx, t_virtual_idx, s_connect_idx, t_connect_idx, device):
        if s_connect_idx.numel() == 0 or t_connect_idx.numel() == 0:
            raise ValueError("s_connect_idx and t_connect_idx must contain at least one highway node index.")

        s_connect_idx = s_connect_idx.to(device)
        t_connect_idx = t_connect_idx.to(device)

        s_to_h = torch.stack([torch.full_like(s_connect_idx, s_virtual_idx), s_connect_idx], dim=0)
        h_to_s = torch.stack([s_connect_idx, torch.full_like(s_connect_idx, s_virtual_idx)], dim=0)
        t_to_h = torch.stack([torch.full_like(t_connect_idx, t_virtual_idx), t_connect_idx], dim=0)
        h_to_t = torch.stack([t_connect_idx, torch.full_like(t_connect_idx, t_virtual_idx)], dim=0)
        return torch.cat([s_to_h, h_to_s, t_to_h, h_to_t], dim=1)

    def forward(
        self,
        x_highway,
        edge_index_highway,
        s_global_feat,
        t_global_feat,
        s_connect_idx,
        t_connect_idx,
    ):
        """
        Args:
            x_highway (torch.Tensor): [num_highway_nodes, highway_in_dim] 高速图节点特征。
            edge_index_highway (torch.LongTensor): [2, num_highway_edges] 高速图边索引。
            s_global_feat (torch.Tensor): [global_feat_dim] 或 [1, global_feat_dim] 起点全局特征。
            t_global_feat (torch.Tensor): [global_feat_dim] 或 [1, global_feat_dim] 终点全局特征。
            s_connect_idx (torch.LongTensor): [Ks] 与 s 虚拟节点相连的高速节点索引。
            t_connect_idx (torch.LongTensor): [Kt] 与 t 虚拟节点相连的高速节点索引。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - node_emb: [num_highway_nodes + 2, output_dim]
                - pair_emb: [1, output_dim] 经过 readout 的跨分区嵌入 h_st_inter
                - st_virtual_emb: [1, 2*output_dim] s/t 虚拟节点拼接嵌入
        """
        device = x_highway.device
        s_global_feat = s_global_feat.view(1, -1).to(device)
        t_global_feat = t_global_feat.view(1, -1).to(device)
        s_virtual_feat = self.global_encoder(s_global_feat)
        t_virtual_feat = self.global_encoder(t_global_feat)

        x_aug = torch.cat([x_highway, s_virtual_feat, t_virtual_feat], dim=0)
        s_virtual_idx = x_highway.size(0)
        t_virtual_idx = x_highway.size(0) + 1

        virtual_edges = self._build_virtual_edges(
            s_virtual_idx=s_virtual_idx,
            t_virtual_idx=t_virtual_idx,
            s_connect_idx=s_connect_idx.long(),
            t_connect_idx=t_connect_idx.long(),
            device=device,
        )
        edge_index_aug = torch.cat([edge_index_highway.to(device), virtual_edges], dim=1)

        h = x_aug
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index_aug)
            if layer_idx != len(self.convs) - 1:
                h = self.act(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        s_virtual_emb = h[s_virtual_idx : s_virtual_idx + 1]
        t_virtual_emb = h[t_virtual_idx : t_virtual_idx + 1]
        st_virtual_emb = torch.cat([s_virtual_emb, t_virtual_emb], dim=-1)
        pair_emb = self.readout(st_virtual_emb)
        return h, pair_emb, st_virtual_emb


class DistancePredictor(nn.Module):
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
    ):
        super().__init__()
        self.inner_gnn = InnerGNN(
            input_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=inner_out_dim,
            num_layers=num_inner_layers,
            dropout=dropout,
        )
        self.inter_gnn = InterGNN(
            highway_in_dim=highway_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=inter_out_dim,
            global_feat_dim=global_feat_dim,
            num_layers=num_inter_layers,
            dropout=dropout,
        )
        self.use_highway_distance_feature = use_highway_distance_feature
        self.highway_distance_feat_dim = highway_distance_feat_dim if use_highway_distance_feature else 0
        # 融合输入为 4 块嵌入：[h_s_inner | h_t_inner | h_s_inter | h_t_inter]
        # （h_s_inter/h_t_inter 由 InterGNN 的两个虚拟节点给出，对应论文图中的绿/棕两块）
        # 可选再拼接 highway 分解距离特征。
        fusion_in_dim = 2 * inner_out_dim + 2 * inter_out_dim + self.highway_distance_feat_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )
        self.output_activation = nn.Softplus()

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
        return_aux=False,
    ):
        """
        Args:
            x_s (torch.Tensor): [num_nodes_s, node_feat_dim] 起点分区子图节点特征。
            edge_index_s (torch.LongTensor): [2, num_edges_s] 起点分区子图边索引。
            s_idx (int | torch.LongTensor): 起点节点在 x_s 内的索引。
            x_t (torch.Tensor): [num_nodes_t, node_feat_dim] 终点分区子图节点特征。
            edge_index_t (torch.LongTensor): [2, num_edges_t] 终点分区子图边索引。
            t_idx (int | torch.LongTensor): 终点节点在 x_t 内的索引。
            x_highway (torch.Tensor): [num_highway_nodes, highway_feat_dim] 高速图节点特征。
            edge_index_highway (torch.LongTensor): [2, num_highway_edges] 高速图边索引。
            s_global_feat (torch.Tensor): [global_feat_dim] 或 [1, global_feat_dim] 起点全局位置特征。
            t_global_feat (torch.Tensor): [global_feat_dim] 或 [1, global_feat_dim] 终点全局位置特征。
            s_connect_idx (torch.LongTensor): [Ks] s 虚拟节点连接到的高速节点索引。
            t_connect_idx (torch.LongTensor): [Kt] t 虚拟节点连接到的高速节点索引。
            highway_dist_feat (torch.Tensor | None): [D] 或 [1, D] highway 分解距离特征（可选）。
            return_aux (bool): 是否返回中间嵌入用于调试/可视化。

        Returns:
            torch.Tensor | tuple:
                - y_hat: [1] 预测距离标量（非负）
                - 若 return_aux=True，额外返回中间嵌入字典。
        """
        _, h_s_inner = self.inner_gnn(x_s, edge_index_s, s_idx)
        _, h_t_inner = self.inner_gnn(x_t, edge_index_t, t_idx)
        # st_virtual_emb: [1, 2*inter_out_dim] = cat([h_s_inter, h_t_inter])，保持 s/t 两块独立
        _, _, st_virtual_emb = self.inter_gnn(
            x_highway=x_highway,
            edge_index_highway=edge_index_highway,
            s_global_feat=s_global_feat,
            t_global_feat=t_global_feat,
            s_connect_idx=s_connect_idx,
            t_connect_idx=t_connect_idx,
        )
        fusion_parts = [h_s_inner, h_t_inner, st_virtual_emb]
        if self.use_highway_distance_feature and highway_dist_feat is not None:
            fusion_parts.append(highway_dist_feat.view(1, -1).to(h_s_inner.device))
        fusion_input = torch.cat(fusion_parts, dim=-1)
        y_hat = self.output_activation(self.fusion_mlp(fusion_input)).view(-1)

        if not return_aux:
            return y_hat
        return y_hat, {
            "h_s_inner": h_s_inner,
            "h_t_inner": h_t_inner,
            "h_st_inter": st_virtual_emb,
            "fusion_input": fusion_input,
        }


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