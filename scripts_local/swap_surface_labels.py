# -*- coding: utf-8 -*-
"""把 runner 采样缓存替换为【表面测地标签】（保留原图距离为 *_graph.csv 备份）。

对三个策略：random / oracle_mix / distance_gap
输出：原路径写回 (s,t,distance) —— 距离=表面测地；并打印图折线 vs 表面 的偏差统计。
"""
import csv
import os
import shutil
import sys

CACHE = '/root/autodl-tmp/gnn-euclidean-local/outputs/cache_ep_high'


def load_surface(path):
    d = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            d[(int(row['s']), int(row['t']))] = float(row['true_distance'])
    return d


def main():
    for strat in ['random', 'oracle_mix', 'distance_gap']:
        orig = os.path.join(CACHE, f'ep_high_{strat}_n50000_seed42.csv')
        surf = os.path.join(CACHE, 'surface', f'surface_{strat}.csv')
        backup = os.path.join(CACHE, f'ep_high_{strat}_n50000_seed42_graph.csv')
        if not os.path.exists(surf):
            print(f'[swap] 缺少 {surf}，跳过 {strat}')
            continue
        if not os.path.exists(backup):
            shutil.copyfile(orig, backup)
            print(f'[swap] 已备份图距离标签 -> {os.path.basename(backup)}')
        smap = load_surface(surf)
        rows = []
        with open(backup, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                s, t = int(row['s']), int(row['t'])
                if (s, t) in smap:
                    rows.append((s, t, smap[(s, t)], float(row['distance'])))
        missing = 50000 - len(rows)
        with open(orig, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['s', 't', 'distance'])
            for s, t, dsurf, _ in rows:
                w.writerow([s, t, f'{dsurf:.6f}'])
        diffs = [(g - d) / d * 100 for _, _, d, g in rows if d > 0]
        import statistics
        print(f'[swap] {strat}: 替换 {len(rows)} 对 (缺 {missing})；图折线相对表面真值 '
              f'平均 {statistics.mean(diffs):+.2f}% 中位 {statistics.median(diffs):+.2f}% '
              f'最大 {max(diffs):+.2f}%')


if __name__ == '__main__':
    main()
