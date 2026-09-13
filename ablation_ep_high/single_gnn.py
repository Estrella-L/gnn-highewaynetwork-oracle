# -*- coding: utf-8 -*-
"""消融实验 A2/A3：单 GNN（去掉地形分区 + highway 网络）在 EP_high 上的训练/评估。

A2 = 单 GNN + 欧氏残差输出；A3 = 单 GNN + 直接预测（软加号输出）。
协议与主方法（M0）完全一致：同一份精确表面测地标签、同一 80/10/10 切分（seed 42）、
同样 100 轮、Adam lr 1e-3 wd 5e-4、ReduceLROnPlateau(0.5, patience 5)、huber 损失。
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common import build_graph, euclidean_pairs, load_labels, load_mesh, metrics, split_samples  # noqa: E402
from single_gnn_model import SingleGNNPredictor  # noqa: E402


def build_node_features(V, edge_index_np, feature_dim=64):
    """与 preprocess.build_full_graph_tensors 完全同口径的节点特征。

    base = [label/max_label, degree/max_degree, x_norm, y_norm]，再平铺到 feature_dim。
    （label = 连通分量编号；EP_high 是单连通网格，故该列近似常数，与主方法输入一致。）
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    src, dst = edge_index_np[0], edge_index_np[1]
    A = coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    degree = np.bincount(src, minlength=n).astype(np.float32)
    labels = labels.astype(np.float32)
    max_label = float(max(1.0, labels.max()))
    max_degree = float(max(1.0, degree.max()))
    xs, ys = V[:, 0].astype(np.float32), V[:, 1].astype(np.float32)
    xs = (xs - xs.min()) / max(1e-6, xs.max() - xs.min())
    ys = (ys - ys.min()) / max(1e-6, ys.max() - ys.min())
    base = np.stack([labels / max_label, degree / max_degree, xs, ys], axis=1)
    rep = int(math.ceil(feature_dim / base.shape[1]))
    return np.tile(base, (1, rep))[:, :feature_dim].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--prediction_mode', default='direct',
                    choices=['direct', 'euclidean_residual'])
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--out_dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--fusion_hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_json', default='')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    V, F = load_mesh(args.off)
    edge_index_np, edge_len_np = build_graph(V, F)
    # SAGEConv 需要无向图双向边（build_graph 已含双向）
    x_np = build_node_features(V, edge_index_np, feature_dim=64)
    labels = load_labels(args.labels_csv)
    tr, va, te = split_samples(labels, 0.8, 0.1, args.seed)
    print('[data] V=%d E=%d pairs=%d/%d/%d x=%s' % (len(V), edge_index_np.shape[1],
                                                    len(tr), len(va), len(te), x_np.shape), flush=True)

    x = torch.from_numpy(x_np).to(device)
    edge_index = torch.from_numpy(edge_index_np).to(device)

    def pack(rows):
        s = torch.tensor([r[0] for r in rows], dtype=torch.long)
        t = torch.tensor([r[1] for r in rows], dtype=torch.long)
        y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        e = torch.log1p(torch.from_numpy(euclidean_pairs(V, [(r[0], r[1]) for r in rows])).float())
        return s, t, y, e

    s_tr, t_tr, y_tr, e_tr = pack(tr)
    s_va, t_va, y_va, e_va = pack(va)
    s_te, t_te, y_te, e_te = pack(te)

    model = SingleGNNPredictor(node_feat_dim=x.shape[1], hidden_dim=args.hidden_dim,
                               out_dim=args.out_dim, num_layers=args.layers,
                               fusion_hidden_dim=args.fusion_hidden,
                               prediction_mode=args.prediction_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=5, min_lr=1e-6)
    print('[model] single_gnn mode=%s params=%.2fM' %
          (args.prediction_mode, sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    def head(s_idx, t_idx, euc):
        h = model.encode(x, edge_index)
        return model.predict_from_embeddings(h[s_idx], h[t_idx], euc)

    def evaluate(s_idx, t_idx, y, euc):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(s_idx), args.batch):
                sl = slice(i, i + args.batch)
                preds.append(head(s_idx[sl].to(device), t_idx[sl].to(device), euc[sl].to(device)).float().cpu())
        pred = torch.cat(preds).numpy()
        return metrics(y.numpy().tolist(), pred.tolist())

    best = {'rel': float('inf'), 'epoch': -1, 'state': None}
    hist = []
    n_tr = len(s_tr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        te0 = time.time()
        total = 0.0
        n_batches = 0
        for i in range(0, n_tr, args.batch):
            sl = slice(i, i + args.batch)
            opt.zero_grad(set_to_none=True)
            h = model.encode(x, edge_index)          # 整图一次消息传递（每批重算，便于反传）
            pred = model.predict_from_embeddings(h[s_tr[sl].to(device)], h[t_tr[sl].to(device)],
                                                 e_tr[sl].to(device))
            loss = nn.functional.huber_loss(pred, y_tr[sl].to(device), delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * (sl.stop - sl.start)
            n_batches += 1
        vm = evaluate(s_va, t_va, y_va, e_va)
        sched.step(vm['relative_error'])
        hist.append({'epoch': epoch, 'loss': total / max(1, n_tr), 'val_rel': vm['relative_error'],
                     'val_mae': vm['mae'], 'sec': time.time() - te0})
        if vm['relative_error'] < best['rel']:
            best = {'rel': vm['relative_error'], 'epoch': epoch,
                    'state': copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        if epoch % 5 == 0 or epoch == args.epochs:
            print('[single_gnn/%s] epoch=%d loss=%.1f val_rel=%.5f (%.1fs)' %
                  (args.prediction_mode, epoch, total / n_tr, vm['relative_error'], time.time() - te0),
                  flush=True)

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    tm = evaluate(s_te, t_te, y_te, e_te)
    euc_m = metrics(y_te.numpy().tolist(), np.expm1(e_te.numpy()).tolist())
    result = {
        'tag': args.tag or ('single_gnn_%s' % args.prediction_mode),
        'config': 'A2（单 GNN + 欧氏残差）' if args.prediction_mode == 'euclidean_residual'
                  else 'A3（单 GNN + 直接预测）',
        'architecture': 'single_gnn', 'prediction_mode': args.prediction_mode,
        'off': os.path.basename(args.off), 'n_vertices': int(len(V)),
        'n_edges': int(edge_index_np.shape[1]), 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'epochs': args.epochs, 'hidden_dim': args.hidden_dim, 'out_dim': args.out_dim,
        'best_epoch': best['epoch'], 'best_val_rel': best['rel'],
        'test_relative_error': tm['relative_error'], 'test_mae': tm['mae'], 'test_rmse': tm['rmse'],
        'euclid_relative_error': euc_m['relative_error'], 'euclid_mae': euc_m['mae'],
        'elapsed_sec': time.time() - t0, 'history': hist,
    }
    print('[single_gnn/%s] TEST rel=%.5f mae=%.2f rmse=%.2f | 纯欧氏 %.5f | %.0fs' %
          (args.prediction_mode, tm['relative_error'], tm['mae'], tm['rmse'],
           euc_m['relative_error'], result['elapsed_sec']), flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('[save] ' + args.out_json, flush=True)


if __name__ == '__main__':
    main()
