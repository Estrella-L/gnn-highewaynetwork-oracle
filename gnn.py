# GNN 模块：三段式 DistancePredictor（Inner 本地段 / Inter 高速段 / Fusion 融合段）。
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn


class InnerGNN(nn.Module):
    """
    Inner-GNN（局部段）：v1.0.0 起改为 pre-norm + residual + LayerNorm 深度块
    （DeepGCNs / GCNII 风格），支持任意层数 num_layers ≥ 2 稳定训练。

    结构：
        x → input_proj → [LN → ReLU → Dropout → SAGEConv → +residual] × L → output_proj
    等宽实现：所有卷积统一 hidden_dim → hidden_dim，首尾用 Linear 投影解决维度对齐。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for InnerGNN.")
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList(
            [geo_nn.SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()

    def _residual_stack(self, h, edge_index):
        """pre-norm residual 主体循环，forward 与 forward_batch 共用。"""
        for i, conv in enumerate(self.convs):
            h_res = h
            h_pre = self.norms[i](h)
            h_pre = self.act(h_pre)
            h_pre = nn.functional.dropout(h_pre, p=self.dropout, training=self.training)
            h_conv = conv(h_pre, edge_index)
            h = h_conv + h_res
        return h

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
        h = self.input_proj(x)
        h = self._residual_stack(h, edge_index)
        h = self.output_proj(h)

        if isinstance(query_idx, int):
            query_idx = torch.tensor([query_idx], dtype=torch.long, device=h.device)
        elif query_idx.dim() == 0:
            query_idx = query_idx.view(1).to(h.device)
        else:
            query_idx = query_idx.to(h.device)
        query_emb = h[query_idx]
        return h, query_emb

    def forward_batch(self, x_list, edge_index_list, query_idx_list):
        """
        批量编码多个子图：合并成一张不连通大图做一次消息传递，返回各自查询点嵌入。

        Args:
            x_list (list[Tensor]): 每个样本的子图节点特征 [Ni, in_dim]。
            edge_index_list (list[LongTensor]): 每个样本的子图边 [2, Ei]。
            query_idx_list (list[int|Tensor]): 每个样本查询点在其子图内的局部索引。

        Returns:
            Tensor: [B, output_dim] 各样本查询点嵌入。
        """
        device = x_list[0].device
        xs, es, qidx = [], [], []
        offset = 0
        for x, e, q in zip(x_list, edge_index_list, query_idx_list):
            xs.append(x)
            es.append(e.to(device) + offset)
            qi = q if isinstance(q, int) else int(torch.as_tensor(q).reshape(-1)[0].item())
            qidx.append(offset + qi)
            offset += x.size(0)
        h = self.input_proj(torch.cat(xs, dim=0))
        edge_index = torch.cat(es, dim=1)
        h = self._residual_stack(h, edge_index)
        h = self.output_proj(h)
        query = torch.tensor(qidx, dtype=torch.long, device=h.device)
        return h[query]


class InterGNN(nn.Module):
    """
    Inter-GNN（高速段）：v1.0.0 起同样改为 pre-norm + residual 深度块，支持 4~6 层以上稳定训练。
    这是主要修复目标——高速图直径远>2，原 2 层欠传播 (under-reaching)。
    """

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
        self.input_proj = nn.Linear(highway_in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList(
            [geo_nn.SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.readout = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _residual_stack(self, h, edge_index):
        """pre-norm residual 主体循环，forward 与 forward_batch 共用。"""
        for i, conv in enumerate(self.convs):
            h_res = h
            h_pre = self.norms[i](h)
            h_pre = self.act(h_pre)
            h_pre = nn.functional.dropout(h_pre, p=self.dropout, training=self.training)
            h_conv = conv(h_pre, edge_index)
            h = h_conv + h_res
        return h

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

        h = self.input_proj(x_aug)
        h = self._residual_stack(h, edge_index_aug)
        h = self.output_proj(h)

        s_virtual_emb = h[s_virtual_idx : s_virtual_idx + 1]
        t_virtual_emb = h[t_virtual_idx : t_virtual_idx + 1]
        st_virtual_emb = torch.cat([s_virtual_emb, t_virtual_emb], dim=-1)
        pair_emb = self.readout(st_virtual_emb)
        return h, pair_emb, st_virtual_emb

    def forward_batch(self, x_highway, edge_index_highway, s_global_feats, t_global_feats,
                      s_connect_list, t_connect_list):
        """
        批量版 Inter 段：共享的高速图复制 B 份、各加 s/t 两个虚拟节点，合并成一张大图做一次消息传递。

        Args:
            x_highway (Tensor): [K, highway_in_dim] 共享高速图节点特征。
            edge_index_highway (LongTensor): [2, Eh] 共享高速图边。
            s_global_feats / t_global_feats (Tensor): [B, global_feat_dim] 各样本 s/t 全局特征。
            s_connect_list / t_connect_list (list[LongTensor]): 各样本 s/t 连接的高速节点 local 索引。

        Returns:
            Tensor: [B, 2*output_dim] 各样本 s/t 两个虚拟节点嵌入的拼接。
        """
        device = x_highway.device
        K = x_highway.size(0)
        edge_index_highway = edge_index_highway.to(device)
        s_virt = self.global_encoder(s_global_feats.to(device))  # [B, highway_in_dim]
        t_virt = self.global_encoder(t_global_feats.to(device))  # [B, highway_in_dim]
        B = s_virt.size(0)

        x_parts, edge_parts, s_idx_list, t_idx_list = [], [], [], []
        offset = 0
        for b in range(B):
            x_parts.extend([x_highway, s_virt[b:b + 1], t_virt[b:b + 1]])
            s_vidx, t_vidx = offset + K, offset + K + 1
            s_idx_list.append(s_vidx)
            t_idx_list.append(t_vidx)
            edge_parts.append(edge_index_highway + offset)
            sc = s_connect_list[b].to(device).long() + offset
            tc = t_connect_list[b].to(device).long() + offset
            s_full = torch.full_like(sc, s_vidx)
            t_full = torch.full_like(tc, t_vidx)
            edge_parts.append(torch.cat([
                torch.stack([s_full, sc], dim=0),
                torch.stack([sc, s_full], dim=0),
                torch.stack([t_full, tc], dim=0),
                torch.stack([tc, t_full], dim=0),
            ], dim=1))
            offset += K + 2

        h = self.input_proj(torch.cat(x_parts, dim=0))
        edge_all = torch.cat(edge_parts, dim=1)
        h = self._residual_stack(h, edge_all)
        h = self.output_proj(h)
        s_emb = h[torch.tensor(s_idx_list, dtype=torch.long, device=device)]
        t_emb = h[torch.tensor(t_idx_list, dtype=torch.long, device=device)]
        return torch.cat([s_emb, t_emb], dim=-1)  # [B, 2*output_dim]


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

    def forward_batch(self, samples):
        """
        批量前向：一次处理多个 (s,t) 样本，返回 [B] 预测距离。

        Args:
            samples (list[dict]): 每个元素是 build_synthetic_partition_inputs 产出的单样本输入字典。
                同一批样本共享同一份高速图（x_highway / edge_index_highway 取自 samples[0]）。

        Returns:
            Tensor: [B] 非负预测距离。
        """
        h_s_inner = self.inner_gnn.forward_batch(
            [d["x_s"] for d in samples],
            [d["edge_index_s"] for d in samples],
            [d["s_idx"] for d in samples],
        )
        h_t_inner = self.inner_gnn.forward_batch(
            [d["x_t"] for d in samples],
            [d["edge_index_t"] for d in samples],
            [d["t_idx"] for d in samples],
        )
        device = h_s_inner.device
        s_global = torch.stack([d["s_global_feat"].reshape(-1) for d in samples], dim=0)
        t_global = torch.stack([d["t_global_feat"].reshape(-1) for d in samples], dim=0)
        st_virtual_emb = self.inter_gnn.forward_batch(
            samples[0]["x_highway"],
            samples[0]["edge_index_highway"],
            s_global,
            t_global,
            [d["s_connect_idx"] for d in samples],
            [d["t_connect_idx"] for d in samples],
        )
        fusion_parts = [h_s_inner, h_t_inner, st_virtual_emb]
        if self.use_highway_distance_feature and samples[0].get("highway_dist_feat") is not None:
            hdf = torch.stack([d["highway_dist_feat"].reshape(-1) for d in samples], dim=0).to(device)
            fusion_parts.append(hdf)
        fusion_input = torch.cat(fusion_parts, dim=-1)  # [B, fusion_in_dim]
        return self.output_activation(self.fusion_mlp(fusion_input)).view(-1)  # [B]


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