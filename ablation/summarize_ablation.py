# -*- coding: utf-8 -*-
"""汇总解剖式消融结果，生成对比表（markdown）。"""
import argparse
import glob
import json
import os

# 主方法 M0：三段式 + 欧氏残差（100 轮），来自主实验
M0 = {'name': 'M0 总体方法（分区+highway+三段式 GNN + 欧氏残差）', 'rel': 0.032943,
      'mae': 205.11, 'rmse': 548.00, 'source': 'ep_high_surface_grid.jsonl (surf_best_h64_k3_100ep)'}
M0_PLUS = {'name': 'M0+ 同上，但残差幅度放宽到 ±1.0', 'rel': 0.032199, 'mae': 199.40,
           'rmse': 499.76, 'source': 'ep_high_surface_res1_grid.jsonl'}
A4 = {'rel': 0.19306, 'mae': 1590.73}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--a1_jsonl', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        with open(p, encoding='utf-8') as f:
            rows.append(json.load(f))
    a1 = None
    if args.a1_jsonl and os.path.exists(args.a1_jsonl):
        for line in open(args.a1_jsonl, encoding='utf-8'):
            line = line.strip()
            if line:
                a1 = json.loads(line)

    def row(tag, rel, mae, rmse, note):
        return '| %s | %.5f | %.2f | %s | %s |' % (
            tag, rel, mae, ('%.2f' % rmse) if rmse else '—', note)

    lines = ['# EP_high 解剖式消融（Ablation Study）', '',
             '- 数据：EP_high.off（1,392,236 顶点 / 8,337,166 条边）',
             '- 真值：flip-out 精确表面测地距离，50,000 对；80/10/10 切分 seed=42（test 5,000 对）',
             '- 指标：MRE = mean(|pred-true|/true)；同一 test 集', '',
             '| 配置 | test MRE | MAE | RMSE | 相对 M0 |', '|---|---|---|---|---|']
    base = M0['rel']
    lines.append(row('**M0 总体方法（全）**', M0['rel'], M0['mae'], M0['rmse'], '1.00x（基准）'))
    if a1:
        lines.append(row('**A1 去掉欧氏残差**（保留分区+highway+三段式）',
                         a1['test_relative_error'], a1['test_mae'], a1['test_rmse'],
                         '%.1fx' % (a1['test_relative_error'] / base)))
    for r in rows:
        if r.get('prediction_mode') == 'euclidean_residual':
            lines.append(row('**A2 去掉分区+highway**（保留单 GNN + 欧氏残差）',
                             r['test_relative_error'], r['test_mae'], r['test_rmse'],
                             '%.2fx' % (r['test_relative_error'] / base)))
    for r in rows:
        if r.get('prediction_mode') == 'direct':
            lines.append(row('**A3 两者都去掉**（单 GNN + 直接预测）',
                             r['test_relative_error'], r['test_mae'], r['test_rmse'],
                             '%.1fx' % (r['test_relative_error'] / base)))
    lines.append(row('**A4 纯 3D 欧氏直线（无学习）**', A4['rel'], A4['mae'], None,
                     '%.1fx' % (A4['rel'] / base)))
    lines.append(row('M0+ 参考：同一 M0 但残差幅度放宽到 ±1.0', M0_PLUS['rel'], M0_PLUS['mae'],
                     M0_PLUS['rmse'], '%.2fx' % (M0_PLUS['rel'] / base)))

    text = chr(10).join(lines)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + chr(10))
        print('[save] ' + args.out)


if __name__ == '__main__':
    main()
