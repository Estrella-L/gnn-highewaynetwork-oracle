# -*- coding: utf-8 -*-
"""在 EP_high（或任意地形网格）上训练/评估三篇 baseline。

用法示例：
  python train.py --off dataset/EP_high/EP_high.off \
      --labels_csv outputs/cache_ep_high/surface/surface_random.csv \
      --arch neurogf --out_mode euclidean_residual --epochs 20 --out_json results/neurogf_er.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (build_graph, compute_normals, euclidean_pairs, load_labels,  # noqa: E402
                    load_mesh, metrics, split_samples)
from models import ARCHITECTURES, build_model  # noqa: E402


def huber(pred, target, delta=1.0):
    return nn.functional.huber_loss(pred, target, delta=delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', required=True)
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--labels_col', default='true_distance')
    ap.add_argument('--arch', required=True, choices=list(ARCHITECTURES))
    ap.add_argument('--out_mode', default='native',
                    choices=['native', 'softplus', 'euclidean_residual'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--max_train_pairs', type=int, default=0, help='>0 时只用前 N 对做快速自检')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--amp', default='bf16', choices=['none', 'bf16', 'fp16'])
    ap.add_argument('--no_checkpoint', action='store_true', help='关闭梯度检查点（更吃显存）')
    ap.add_argument('--litege_pca_dim', type=int, default=200)
    ap.add_argument('--pca_cache', default='', help='LiteGE 描述子缓存 npz 路径')
    ap.add_argument('--out_json', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--log_every', type=int, default=1)
    ap.add_argument('--scheduler', default='plateau', choices=['none', 'plateau', 'cosine'])
    ap.add_argument('--lr_factor', type=float, default=0.5)
    ap.add_argument('--lr_patience', type=int, default=8)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t_start = time.time()

    # ---------------- 数据 ----------------
    V, F = load_mesh(args.off)
    edge_index_np, edge_len_np = build_graph(V, F)
    labels = load_labels(args.labels_csv, args.labels_col)
    tr, va, te = split_samples(labels, 0.8, 0.1, args.seed)
    if args.max_train_pairs:
        tr = tr[:args.max_train_pairs]
        va = va[:max(64, args.max_train_pairs // 8)]
        te = te[:max(64, args.max_train_pairs // 8)]

    n_v, n_e = len(V), edge_index_np.shape[1]
    print('[data] V=%d E=%d | pairs train/val/test = %d/%d/%d' % (n_v, n_e, len(tr), len(va), len(te)),
          flush=True)

    x_np = V.astype(np.float32)
    if args.arch == 'gegnn':
        normals = compute_normals(V, F)
        x_np = np.concatenate([x_np, normals], axis=1)   # GeGnn 官方输入 6 维（坐标+法向）
    x = torch.from_numpy(x_np).to(device)
    edge_index = torch.from_numpy(edge_index_np).to(device)
    edge_len = torch.from_numpy(edge_len_np).to(device)
    pos = torch.from_numpy(V.astype(np.float32)).to(device)

    def to_tensors(rows):
        s = torch.tensor([r[0] for r in rows], dtype=torch.long)
        t = torch.tensor([r[1] for r in rows], dtype=torch.long)
        y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        return s, t, y

    s_tr, t_tr, y_tr = to_tensors(tr)
    s_va, t_va, y_va = to_tensors(va)
    s_te, t_te, y_te = to_tensors(te)

    euc_tr = torch.log1p(torch.from_numpy(euclidean_pairs(V, [(r[0], r[1]) for r in tr])).float())
    euc_va = torch.log1p(torch.from_numpy(euclidean_pairs(V, [(r[0], r[1]) for r in va])).float())
    euc_te = torch.log1p(torch.from_numpy(euclidean_pairs(V, [(r[0], r[1]) for r in te])).float())

    # ---------------- 模型 ----------------
    model = build_model(args.arch, args.out_mode, in_dim=x.shape[1], width=args.width,
                        layers=args.layers, litege_pca_dim=args.litege_pca_dim,
                        use_checkpoint=not args.no_checkpoint).to(device)
    if args.arch == 'gegnn':
        # GeGnn 需要按**聚合目标节点**（index[0]）排序的 CSR 边表（分块卷积用）
        target = edge_index_np[0]
        order = np.argsort(target, kind='stable')
        ei_sorted = torch.from_numpy(edge_index_np[:, order]).to(device)
        counts = np.bincount(target, minlength=n_v)
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        model.setup_graph(pos, edge_len, ei_sorted, torch.from_numpy(ptr).to(device))
    else:
        model.setup_graph(pos, edge_len)
    if args.arch == 'litege':
        from litege_desc import compute_udf_pca_descriptor
        cache = args.pca_cache or (args.off + '.litege_pca%d.npz' % args.litege_pca_dim)
        node_pca = compute_udf_pca_descriptor(V, pca_dim=args.litege_pca_dim, cache_path=cache)
        model.set_node_pca(torch.from_numpy(np.asarray(node_pca, dtype=np.float32)))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    elif args.scheduler == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    else:
        sched = None
    n_params = sum(p.numel() for p in model.parameters())
    print('[model] arch=%s out=%s params=%.2fM' % (args.arch, args.out_mode, n_params / 1e6), flush=True)

    def encode_and_head(s_idx, t_idx, euc, train=True):
        h = model.encode(x, edge_index)
        h_s = h[s_idx.to(h.device)]
        h_t = h[t_idx.to(h.device)]
        return model.predict_from_embeddings(h_s, h_t, euc.to(h.device))

    def evaluate(rows_s, rows_t, y, euc):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows_s), args.batch):
                sl = slice(i, i + args.batch)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                    p = encode_and_head(rows_s[sl], rows_t[sl], euc[sl], train=False)
                preds.append(p.float().cpu())
        pred = torch.cat(preds).numpy()
        return metrics(y.numpy().tolist(), pred.tolist()), pred

    best = {'rel': float('inf'), 'state': None, 'epoch': -1}
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        n = len(s_tr)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch))
        loss_sum = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                pred = encode_and_head(s_tr[idx], t_tr[idx], euc_tr[idx], train=True)
                loss = huber(pred, y_tr[idx].to(pred.device))
            (loss / max(1, (n + args.batch - 1) // args.batch)).backward()
            loss_sum += float(loss) * len(idx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tr_loss = loss_sum / n
        val_metrics, _ = evaluate(s_va, t_va, y_va, euc_va)
        hist.append({'epoch': epoch, 'train_loss': tr_loss, 'val_rel': val_metrics['relative_error'],
                     'val_mae': val_metrics['mae'], 'sec': time.time() - t0})
        if sched is not None:
            if args.scheduler == 'plateau':
                sched.step(val_metrics['relative_error'])
            else:
                sched.step()
        if val_metrics['relative_error'] < best['rel']:
            best = {'rel': val_metrics['relative_error'], 'epoch': epoch,
                    'state': copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print('[%s/%s] epoch=%d loss=%.2f val_rel=%.5f val_mae=%.2f (%.1fs)'
                  % (args.arch, args.out_mode, epoch, tr_loss, val_metrics['relative_error'],
                     val_metrics['mae'], time.time() - t0), flush=True)

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    test_metrics, test_pred = evaluate(s_te, t_te, y_te, euc_te)
    euc_pred = np.expm1(euc_te.numpy())
    euc_metrics = metrics(y_te.numpy().tolist(), euc_pred.tolist())

    result = {
        'tag': args.tag or ('%s_%s' % (args.arch, args.out_mode)),
        'arch': args.arch, 'out_mode': args.out_mode,
        'off': os.path.basename(args.off), 'labels_csv': os.path.basename(args.labels_csv),
        'n_vertices': n_v, 'n_edges': n_e,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'epochs': args.epochs, 'lr': args.lr, 'width': args.width, 'layers': args.layers,
        'best_epoch': best['epoch'], 'best_val_rel': best['rel'],
        'test_relative_error': test_metrics['relative_error'],
        'test_mae': test_metrics['mae'], 'test_rmse': test_metrics['rmse'],
        'euclid_relative_error': euc_metrics['relative_error'],
        'euclid_mae': euc_metrics['mae'],
        'gain_vs_euclid': (euc_metrics['relative_error'] / test_metrics['relative_error']
                           if test_metrics['relative_error'] > 0 else float('inf')),
        'n_params': n_params, 'elapsed_sec': time.time() - t_start,
        'history': hist,
    }
    print('[%s/%s] TEST rel=%.5f mae=%.2f rmse=%.2f | 纯欧氏 rel=%.5f | 提升 %.2fx | 用时 %.0fs'
          % (args.arch, args.out_mode, test_metrics['relative_error'], test_metrics['mae'],
             test_metrics['rmse'], euc_metrics['relative_error'], result['gain_vs_euclid'],
             result['elapsed_sec']), flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('[save] ' + args.out_json, flush=True)


if __name__ == '__main__':
    main()
