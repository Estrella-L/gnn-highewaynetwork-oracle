# -*- coding: utf-8 -*-
"""三篇 baseline 论文的**距离回归方法**在本地形数据上的忠实实现。

严格按各自官方仓库的模型定义实现（不是自己想象的版本）：

  1. GeGnn  (ACM TOG 42(6), SIGGRAPH Asia 2023)
     repo: https://github.com/IntelligentGeometry/GeGnn
     - GeoConv  : MessagePassing(aggr='max')，消息里拼接「邻居特征 + 边长 + 相对位置」
                  (源码 hgraph/modules/modules.py: class MyConvOp，aggr='max'，
                   config include_distance=True -> 把两点距离拼进特征)
     - 解码器   : embedding_decoder_mlp = Linear(256,256)+ReLU -> Linear(256,256)+ReLU -> Linear(256,1)
                  作用在 (embd_i - embd_j) ** 2 上   (源码 GnnDist.py: embd_decoder_func)
     - 论文：一次整图前向得到每顶点 256 维嵌入，查询时用小 MLP 解码任意两点测地距离
     简化：略去 U-Net 层级池化（hgraph）与自研 CUDA 算子，其余按原样。

  2. NeuroGF (NeurIPS 2023)
     repo: https://github.com/keeganhk/NeuroGF
     - lifting : FC(3, D/4)+ReLU -> FC(D/4, D/2)+ReLU -> FC(D/2, D)+ReLU   (D=256)
                 (源码 code/neurogf_ovft/neurogf_ovft_models.py: self.lifting)
     - gdf_head: FC(D, D/4)+ReLU -> FC(D/4, 64)+ReLU -> FC(64, 1)+none
                 (源码同上: self.gdf_head)
     - 前向    : diff = |lifting(p_s) - lifting(p_t)|;  gdist = gdf_head(diff)
                 (源码同上: diff = (ftr_qs - ftr_qe).abs(); gdist_out = self.gdf_head(diff))
     简化：只取测地距离分支 B_gdist（我们比较的就是测地距离）；略去 SDF 预训练分支与
           最短路径分支（开曲面无 inside/outside，SDF 分支本身不适用）。

  3. LiteGE (AAAI 2026)
     repo: https://github.com/yya-111/LiteGE
     - mlp_coord = InputMLP(3, neurons, no_layers-1)，逐点坐标 MLP
     - mlp_pca   = MLP(embedding_size, neurons, no_layers)，把形状 UDF-PCA 向量编码
     - 融合      = cat([pca_feat, coord_feat]) -> Linear(2N,N) -> MLP -> Linear(N,N)
     - 解码      = final_linear(final_layers(embed_src - embed_dst)).abs()
                   (源码 CoordMLP.py: CoordMLP.forward_distonly / forward_multi_sources_alldest)
     适配：原文的形状描述子是「数据集级 UDF-PCA」（需要一批形状 + 体素 inside/outside）。
           地形是**单场景开曲面**，我们按四叉树叶子盒把地形切成 K 个 patch 当作"形状集"，
           每个 patch 在统一体素网格上算**无符号距离场(UDF)**再做 PCA —— 这一步是显式适配，
           已在论文对比表中标注。

三个类都提供与主流程一致的接口：
    encode(x, edge_index) -> [N, D]        整图/逐点嵌入
    predict_from_embeddings(h_s, h_t, euclid_feat) -> [B]
因此可以直接复用 main.py 里 single_gnn 的「整图前向 + 单头批次梯度累积」训练循环。
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 共用的输出参数化，保证与主方法可比
# ---------------------------------------------------------------------------
def apply_output_parameterization(raw, mode, euclidean_feat=None):
    """把解码器原始输出变成非负距离预测。

    - "native"            : 各自论文原样（raw 线性输出，不加激活）
    - "softplus"          : softplus(raw)，与主方法 direct 模式一致
    - "euclidean_residual": d_3D * (1 + 0.5*tanh(raw))，与主方法一致
    """
    if mode == "native":
        return raw
    if mode == "softplus":
        return nn.functional.softplus(raw)
    if mode == "euclidean_residual":
        if euclidean_feat is None:
            raise ValueError("euclidean_residual 需要 euclidean_feat")
        base = torch.expm1(euclidean_feat.to(raw.device).view(-1)).clamp_min(1e-6)
        return (base * (1.0 + 0.5 * torch.tanh(raw))).clamp_min(1e-6)
    raise ValueError("unknown prediction_mode: " + mode)


# ---------------------------------------------------------------------------
# 1) GeGnn
# ---------------------------------------------------------------------------
class GeoConv(nn.Module):
    """GeoConv：max 聚合 + 显式几何消息（邻居特征 | 边长 | 相对位置）。

    对应源码 hgraph/modules/modules.py 的 MyConvOp(aggr='max')，
    消息里拼接「邻居特征 x_j」「两点距离 edge_len」「相对位置 pos_i - pos_j」，
    再用 **max** 在邻域上聚合（论文说明这是受 Dijkstra 波前传播启发）。

    这里手写 scatter_reduce(amax) 而不是用 MessagePassing，
    避免 PyG 对非节点张量做 size 推断时报错，语义完全一致。
    """

    def __init__(self, in_channels, out_channels, include_distance=True):
        super().__init__()
        self.include_distance = include_distance
        msg_in = in_channels + 3 + (1 if include_distance else 0)
        self.lin = nn.Linear(msg_in, out_channels)

    def forward(self, x, edge_index, pos, edge_len):
        row, col = edge_index[0], edge_index[1]
        parts = [x[col]]
        if self.include_distance:
            parts.append(edge_len.view(-1, 1).to(x.dtype))
        parts.append(pos[row] - pos[col])
        msg = self.lin(torch.cat(parts, dim=-1))                 # [E, out]
        out = torch.full((x.size(0), msg.size(1)), float("-inf"),
                         device=msg.device, dtype=msg.dtype)
        out = out.scatter_reduce(0, row.view(-1, 1).expand_as(msg), msg,
                                 reduce="amax", include_self=True)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


class GeGnnPredictor(nn.Module):
    def __init__(self, in_dim=6, out_dim=256, num_layers=4, dropout=0.0,
                 prediction_mode="native", include_distance=True):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GeoConv(in_dim if i == 0 else out_dim, out_dim,
                                      include_distance=include_distance))
        # 源码 hgraph/modules/modules.py: GraphConvBnRelu = MyConv -> GroupNorm(4) -> ReLU
        # （normalization = lambda x: GroupNorm(num_groups=4, num_channels=x, eps=bn_eps)）
        # 缺这层归一化时，max 聚合 + ReLU 会让嵌入逐层爆炸，解码器输入 (e_i-e_j)^2 到 1e6 量级，
        # tanh 饱和成 ±1、梯度为 0，训练完全冻结（实测 val_rel 卡死在 0.414 = 1.5*d_3D 的误差）。
        self.norms = nn.ModuleList([nn.GroupNorm(4, out_dim, eps=1e-5) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.dropout = dropout
        # 与源码 embedding_decoder_mlp 同构，作用在 (e_i - e_j)**2 上
        self.embedding_decoder_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.register_buffer("pos", torch.zeros(1, 3), persistent=False)
        self.register_buffer("edge_len", torch.zeros(1), persistent=False)

    def head_parameters(self):
        return self.embedding_decoder_mlp.parameters()

    def setup_graph(self, pos, edge_len):
        self.pos = pos
        self.edge_len = edge_len

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, self.pos, self.edge_len)
            h = self.norms[i](h)      # 与源码 GraphConvBnRelu 一致：conv -> GroupNorm(4)
            if i != len(self.convs) - 1:
                h = self.act(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        embd = (h_s - h_t) ** 2
        raw = self.embedding_decoder_mlp(embd).view(-1)
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


# ---------------------------------------------------------------------------
# 2) NeuroGF
# ---------------------------------------------------------------------------
class NeuroGFPredictor(nn.Module):
    def __init__(self, in_dim=3, D=256, dropout=0.0, prediction_mode="native"):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.D = D
        self.lifting = nn.Sequential(
            nn.Linear(in_dim, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2), nn.ReLU(),
            nn.Linear(D // 2, D), nn.ReLU(),
        )
        self.gdf_head = nn.Sequential(
            nn.Linear(D, D // 4), nn.ReLU(),
            nn.Linear(D // 4, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.dropout = dropout

    def head_parameters(self):
        return self.gdf_head.parameters()

    def setup_graph(self, pos, edge_len):
        return None  # NeuroGF 不用图

    def encode(self, x, edge_index):
        # 逐点 lifting，完全不使用图结构（与原论文一致）
        h = self.lifting(x)
        return nn.functional.dropout(h, p=self.dropout, training=self.training)

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        diff = (h_s - h_t).abs()
        raw = self.gdf_head(diff).view(-1)
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


# ---------------------------------------------------------------------------
# 3) LiteGE
# ---------------------------------------------------------------------------
class _BNMLP(nn.Module):
    """与源码 CoordMLP.py 里 MLP 同构：Linear(no bias) + BatchNorm + ReLU。"""

    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class LiteGEPredictor(nn.Module):
    def __init__(self, coord_dim=3, pca_dim=200, neurons=200, num_layers=3,
                 prediction_mode="native"):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.neurons = neurons
        # 与源码 CoordMLP.__init__ 一一对应
        self.mlp_coord = _BNMLP(coord_dim, neurons, num_layers - 1)
        self.mlp_pca = _BNMLP(pca_dim, neurons, num_layers)
        self.mlp_pcacoord = _BNMLP(neurons, neurons, num_layers - 1)
        self.project_down_pcacoord = nn.Linear(2 * neurons, neurons)
        self.project_embed = nn.Linear(neurons, neurons)
        self.final_layers = _BNMLP(neurons, 2 * neurons, num_layers - 1)
        self.final_linear = nn.Linear(2 * neurons, 1)
        self.register_buffer("node_pca", torch.zeros(1, pca_dim), persistent=False)

    def head_parameters(self):
        return list(self.final_layers.parameters()) + list(self.final_linear.parameters())

    def setup_graph(self, pos, edge_len):
        return None

    def set_node_pca(self, node_pca):
        self.node_pca = node_pca

    def encode(self, x, edge_index):
        coord_feat = self.mlp_coord(x)
        pca_feat = self.mlp_pca(self.node_pca.to(x.device))
        fused = torch.cat([pca_feat, coord_feat], dim=-1)
        fused = self.project_down_pcacoord(fused)
        fused = self.mlp_pcacoord(fused)
        return self.project_embed(fused)

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        out = self.final_linear(self.final_layers(h_s - h_t)).view(-1)
        raw = out.abs()  # 源码 forward_distonly 末尾的 .abs()
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


BASELINE_ARCHITECTURES = ("gegnn", "neurogf", "litege")


def build_baseline_model(architecture, prediction_mode, in_dim=None, hidden_dim=None,
                         dropout=0.0, litege_pca_dim=200, gegnn_layers=4):
    """按 architecture 构造对应 baseline 模型。"""
    if architecture == "gegnn":
        return GeGnnPredictor(in_dim=in_dim or 6, out_dim=256, num_layers=gegnn_layers,
                              dropout=dropout, prediction_mode=prediction_mode)
    if architecture == "neurogf":
        return NeuroGFPredictor(in_dim=in_dim or 3, D=256, dropout=dropout,
                                prediction_mode=prediction_mode)
    if architecture == "litege":
        return LiteGEPredictor(coord_dim=in_dim or 3, pca_dim=litege_pca_dim, neurons=200,
                               num_layers=3, prediction_mode=prediction_mode)
    raise ValueError("unknown baseline architecture: " + architecture)
