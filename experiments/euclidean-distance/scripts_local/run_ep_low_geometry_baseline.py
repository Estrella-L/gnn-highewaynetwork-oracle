#!/usr/bin/env python3
"""Run exact graph-distance geometry baselines on a reproducible EP_low sample."""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline import compute_metrics, euclidean_2d, euclidean_3d  # noqa: E402
from build_highway import build_mesh_graph, iter_source_distances, load_off  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off_file", required=True)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_prefix", required=True)
    return parser


def sample_pairs(num_vertices, num_pairs, seed):
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < num_pairs:
        s = rng.randrange(num_vertices)
        t = rng.randrange(num_vertices)
        if s == t:
            continue
        if s > t:
            s, t = t, s
        pairs.add((s, t))
    return sorted(pairs)


def add_metric(rows, scope, method, true_values, predictions):
    metrics = compute_metrics(true_values, predictions)
    rows.append({
        "scope": scope,
        "method": method,
        "count": len(true_values),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "relative_error": metrics["relative_error"],
    })


def main():
    args = build_parser().parse_args()
    started = time.time()
    print(f"[geometry-baseline] loading {args.off_file}", flush=True)
    vertices, faces = load_off(args.off_file)
    print(
        f"[geometry-baseline] vertices={len(vertices)} faces={len(faces)}; building graph",
        flush=True,
    )
    graph_info, _ = build_mesh_graph(vertices, faces)
    pairs = sample_pairs(len(vertices), args.num_pairs, args.seed)

    by_source = defaultdict(list)
    for s, t in pairs:
        by_source[s].append(t)
    print(
        f"[geometry-baseline] pairs={len(pairs)} unique_sources={len(by_source)}; running exact Dijkstra",
        flush=True,
    )

    edge_u, edge_v = graph_info[3]
    edge_w = graph_info[4]
    true_by_pair = {}
    completed = 0
    for source, distances in iter_source_distances(
        len(vertices), edge_u, edge_v, edge_w, sorted(by_source), weighted=True, chunk=64
    ):
        for target in by_source[source]:
            true_by_pair[(source, target)] = float(distances[target])
        completed += 1
        if completed % 64 == 0 or completed == len(by_source):
            print(
                f"[geometry-baseline] Dijkstra sources {completed}/{len(by_source)}",
                flush=True,
            )

    records = []
    for s, t in pairs:
        true_distance = true_by_pair[(s, t)]
        if not math.isfinite(true_distance):
            continue
        records.append({
            "s": s,
            "t": t,
            "true_distance": true_distance,
            "euclidean_2d": euclidean_2d(vertices, s, t),
            "euclidean_3d": euclidean_3d(vertices, s, t),
        })

    records.sort(key=lambda row: row["true_distance"])
    n = len(records)
    first_cut = n // 3
    second_cut = 2 * n // 3
    groups = {
        "overall": records,
        "short_dist": records[:first_cut],
        "mid_dist": records[first_cut:second_cut],
        "long_dist": records[second_cut:],
    }

    metric_rows = []
    for scope, group in groups.items():
        y_true = [row["true_distance"] for row in group]
        add_metric(metric_rows, scope, "euclidean_2d", y_true, [row["euclidean_2d"] for row in group])
        add_metric(metric_rows, scope, "euclidean_3d", y_true, [row["euclidean_3d"] for row in group])

    output_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(output_dir, exist_ok=True)
    pairs_path = f"{args.out_prefix}_pairs.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    with open(pairs_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["s", "t", "true_distance", "euclidean_2d", "euclidean_3d"],
        )
        writer.writeheader()
        writer.writerows(records)
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "method", "count", "mae", "rmse", "relative_error"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    for row in metric_rows:
        print(
            f"[geometry-baseline] {row['scope']:<12} {row['method']:<13} "
            f"count={row['count']} mae={row['mae']:.6f} "
            f"rmse={row['rmse']:.6f} rel={row['relative_error']:.6f}",
            flush=True,
        )
    print(f"[geometry-baseline] elapsed={time.time() - started:.2f}s", flush=True)
    print(f"[geometry-baseline] wrote {pairs_path}", flush=True)
    print(f"[geometry-baseline] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
