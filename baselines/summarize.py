# -*- coding: utf-8 -*-
"""汇总 train.py 的结果 JSON 成对比表（markdown）。"""
import argparse
import glob
import json
import os

# 我方方法在主数据集 EP_high 上的参考值（表面测地标签口径，见 工作日志/EP_high_表面距离实验结果.md）
OURS = {'name': '我方方法（三段式 + 欧氏残差）', 'test_relative_error': 0.032199,
        'test_mae': 199.40, 'test_rmse': 499.76, 'source': 'ep_high_surface_res1_grid.jsonl'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        with open(p, encoding='utf-8') as f:
            rows.append(json.load(f))
    if not rows:
        print('没有结果文件')
        return
    rows.sort(key=lambda r: r['test_relative_error'])
    euc = rows[0].get('euclid_relative_error')

    lines = ['# EP_high：三篇 baseline vs 我方方法（精确表面测地标签）', '',
             '- 真值：flip-out 精确表面测地距离（50,000 对，与主实验同一份标签）',
             '- 切分：80/10/10，seed=42，与主实验一致（test = %d 对）' % rows[0]['n_test'],
             '- 数据：EP_high.off，%d 顶点 / %d 条边' % (rows[0]['n_vertices'], rows[0]['n_edges']),
             '- 指标：MRE = mean(|pred-true|/true)', '',
             '| 方法 | 输出参数化 | test MRE | MAE | RMSE | 相对纯欧氏 |',
             '|---|---|---|---|---|---|']
    lines.append('| **纯 3D 欧氏直线（无学习）** | — | %.5f | %.2f | — | 1.00x |'
                 % (euc, rows[0].get('euclid_mae', float('nan'))))
    lines.append('| **%s** | 欧氏残差 | %.5f | %.2f | %.2f | %.2fx |'
                 % (OURS['name'], OURS['test_relative_error'], OURS['test_mae'], OURS['test_rmse'],
                    euc / OURS['test_relative_error']))
    for r in rows:
        mode = {'native': '原样', 'euclidean_residual': '欧氏残差', 'softplus': 'softplus'}.get(
            r['out_mode'], r['out_mode'])
        lines.append('| %s | %s | %.5f | %.2f | %.2f | %.2fx |'
                     % (r['arch'], mode, r['test_relative_error'], r['test_mae'], r['test_rmse'],
                        euc / r['test_relative_error']))
    lines += ['', '说明：每个方法的最佳结果取自验证集最优轮次；同一方法在两种输出参数化下其余设置完全相同。']
    text = chr(10).join(lines)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + chr(10))
        print('[save] ' + args.out)


if __name__ == '__main__':
    main()
