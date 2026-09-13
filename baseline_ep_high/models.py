# -*- coding: utf-8 -*-
"""三篇 baseline 的忠实实现（GeGnn / NeuroGF / LiteGE），接口统一：

    model.encode(x, edge_index) -> [N, D]                整图逐点嵌入
    model.predict_from_embeddings(h_s, h_t, euclid_feat) -> [B]

输出参数化三种（与主实验同口径）：
    native            : 论文原样（线性输出，不加激活）
    softplus          : softplus(raw)
    euclidean_residual: d_3D * (1 + 0.5*tanh(raw))
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def apply_output_parameterization(raw, mode, euclidean_feat=None):
    if mode == 'native':
        return raw
    if mode == 'softplus':
        return nn.functional.softplus(raw)
    if mode == 'euclidean_residual':
        if euclidean_feat is None:
            raise ValueError('euclidean_residual 需要 euclidean_feat')
        base = torch.expm1(euclidean_feat.to(raw.device).view(-1)).clamp_min(1e-6)
        return (base * (1.0 + 0.5 * torch.tanh(raw))).clamp_min(1e-6)
    raise ValueError('unknown prediction_mode: ' + mode)


# ---------------------------------------------------------------------------
# 1) GeGnn：GeoConv（max 聚合 + 显式几何消息）+ GroupNorm(4)
# ---------------------------------------------------------------------------
class GeoConv(nn.Module):
    """GeoConv：max 聚合 + 显式几何消息（与官方 MyConvOp(aggr='max') 数学等价）。

    为适配 EP_high（E≈8.3M），消息按**目的节点块**分块计算：
    每块只处理目的节点落在该块内的边，写入块内局部输出，最后拼接。
    这样 autograd 不需要为每次 in-place 聚合保存整张 V×D 的副本，
    峰值显存 = 单块消息张量 + 单块局部输出。
    """

    def __init__(self, in_channels, out_channels, include_distance=True):
        super().__init__()
        self.include_distance = include_distance
        self.lin_x = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_d = nn.Linear(1, out_channels, bias=False) if include_distance else None
        self.lin_p = nn.Linear(3, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._n_local = 0

    def forward(self, x, pos, rc_local, rc_global, cc, el):
        msg = self.lin_x(x[cc])
        if self.lin_d is not None:
            msg = msg + self.lin_d(el.view(-1, 1).to(x.dtype))
        msg = msg + self.lin_p(pos[rc_global] - pos[cc]) + self.bias
        D = msg.size(1)
        out = torch.full((self._n_local, D), float('-inf'), device=msg.device, dtype=msg.dtype)
        out = out.scatter_reduce(0, rc_local.view(-1, 1).expand_as(msg), msg,
                                 reduce='amax', include_self=True)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class GeGnnPredictor(nn.Module):
    def __init__(self, in_dim=6, out_dim=256, num_layers=4, dropout=0.0,
                 prediction_mode='native', include_distance=True, use_checkpoint=True):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.use_checkpoint = use_checkpoint
        self.convs = nn.ModuleList([
            GeoConv(in_dim if i == 0 else out_dim, out_dim, include_distance=include_distance)
            for i in range(num_layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, out_dim, eps=1e-5) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.embedding_decoder_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, 1))
        self.register_buffer('pos', torch.zeros(1, 3), persistent=False)
        self.register_buffer('edge_len', torch.zeros(1), persistent=False)

    def setup_graph(self, pos, edge_len, edge_index=None, edge_ptr=None, block_size=200000):
        """pos/edge_len + 按目的节点排序后的边表（edge_index 已排序，edge_ptr 为 CSR 指针）。"""
        self.pos, self.edge_len = pos, edge_len
        if edge_index is not None and edge_ptr is not None:
            self.row = edge_index[0]
            self.col = edge_index[1]
            self.edge_ptr = edge_ptr
            n = pos.size(0)
            self.block_bounds = list(range(0, n, block_size)) + [n]

    def _conv_block(self, x, conv, lo, hi, e0, e1):
        rc_global = self.row[e0:e1]
        rc_local = rc_global - lo
        cc = self.col[e0:e1]
        el = self.edge_len[e0:e1]
        conv._n_local = hi - lo
        return conv(x, self.pos, rc_local, rc_global, cc, el)

    def _conv_all(self, x, conv):
        outs = []
        for bi in range(len(self.block_bounds) - 1):
            lo, hi = self.block_bounds[bi], self.block_bounds[bi + 1]
            e0, e1 = int(self.edge_ptr[lo]), int(self.edge_ptr[hi])
            if e1 <= e0:
                outs.append(torch.zeros(hi - lo, conv.bias.numel(), device=x.device, dtype=x.dtype))
                continue
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self._conv_block, x, conv, lo, hi, e0, e1, use_reentrant=False))
            else:
                outs.append(self._conv_block(x, conv, lo, hi, e0, e1))
        return torch.cat(outs, dim=0)

    def _layer(self, h, conv, i):
        h = self._conv_all(h, conv)
        h = self.norms[i](h)
        if i != len(self.convs) - 1:
            h = self.act(h)
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h

    def encode(self, x, edge_index=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = self._layer(h, conv, i)
        return h

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        raw = self.embedding_decoder_mlp((h_s - h_t) ** 2).view(-1)
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


# ---------------------------------------------------------------------------
# 2) NeuroGF：逐点 lifting + GDF 差分头（不使用图结构，与原论文一致）
# ---------------------------------------------------------------------------
class NeuroGFPredictor(nn.Module):
    def __init__(self, in_dim=3, D=256, dropout=0.0, prediction_mode='native'):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.lifting = nn.Sequential(
            nn.Linear(in_dim, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2), nn.ReLU(),
            nn.Linear(D // 2, D), nn.ReLU())
        self.gdf_head = nn.Sequential(
            nn.Linear(D, D // 4), nn.ReLU(),
            nn.Linear(D // 4, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.dropout = dropout

    def setup_graph(self, pos, edge_len):
        return None

    def encode(self, x, edge_index):
        h = self.lifting(x)
        return nn.functional.dropout(h, p=self.dropout, training=self.training)

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        raw = self.gdf_head((h_s - h_t).abs()).view(-1)
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


# ---------------------------------------------------------------------------
# 3) LiteGE：CoordMLP（BN-MLP）+ UDF-PCA 形状描述子
# ---------------------------------------------------------------------------
class _BNMLP(nn.Module):
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
                 prediction_mode='native'):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.mlp_coord = _BNMLP(coord_dim, neurons, num_layers - 1)
        self.mlp_pca = _BNMLP(pca_dim, neurons, num_layers)
        self.mlp_pcacoord = _BNMLP(neurons, neurons, num_layers - 1)
        self.project_down_pcacoord = nn.Linear(2 * neurons, neurons)
        self.project_embed = nn.Linear(neurons, neurons)
        self.final_layers = _BNMLP(neurons, 2 * neurons, num_layers - 1)
        self.final_linear = nn.Linear(2 * neurons, 1)
        self.register_buffer('node_pca', torch.zeros(1, pca_dim), persistent=False)

    def setup_graph(self, pos, edge_len):
        return None

    def set_node_pca(self, node_pca):
        self.node_pca = node_pca

    def encode(self, x, edge_index):
        coord_feat = self.mlp_coord(x)
        pca_feat = self.mlp_pca(self.node_pca.to(x.device))
        fused = self.project_down_pcacoord(torch.cat([pca_feat, coord_feat], dim=-1))
        return self.project_embed(self.mlp_pcacoord(fused))

    def predict_from_embeddings(self, h_s, h_t, euclidean_feat=None):
        raw = self.final_linear(self.final_layers(h_s - h_t)).view(-1).abs()
        return apply_output_parameterization(raw, self.prediction_mode, euclidean_feat)


ARCHITECTURES = ('gegnn', 'neurogf', 'litege')


def build_model(arch, prediction_mode, in_dim=3, width=256, layers=4, dropout=0.0,
                litege_pca_dim=200, use_checkpoint=True):
    if arch == 'gegnn':
        return GeGnnPredictor(in_dim=in_dim, out_dim=width, num_layers=layers, dropout=dropout,
                              prediction_mode=prediction_mode, use_checkpoint=use_checkpoint)
    if arch == 'neurogf':
        return NeuroGFPredictor(in_dim=in_dim, D=width, dropout=dropout,
                                prediction_mode=prediction_mode)
    if arch == 'litege':
        return LiteGEPredictor(coord_dim=in_dim, pca_dim=litege_pca_dim, neurons=200,
                               num_layers=3, prediction_mode=prediction_mode)
    raise ValueError('unknown architecture: ' + arch)
