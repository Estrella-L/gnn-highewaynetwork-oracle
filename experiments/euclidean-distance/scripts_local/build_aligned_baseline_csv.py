#!/usr/bin/env python3
"""Build a baseline table aligned to the exact test pairs used by a model run."""

import argparse
import csv
import math
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline import (  # noqa: E402
    compute_metrics,
    euclidean_2d,
    euclidean_3d,
    highway_decomposition_preds,
    load_test_pairs,
)
from build_highway import load_off  # noqa: E402


TEST_METRICS_RE = re.compile(
    r"test_mae=(?P<mae>[\d.eE+-]+),\s*"
    r"test_rmse=(?P<rmse>[\d.eE+-]+),\s*"
    r"test_relative_error=(?P<rel>[\d.eE+-]+)"
)
TEST_PAIRS_RE = re.compile(r"wrote test pairs:\s*(?P<path>.+?)\s*\(\d+ pairs\)")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare a trained model and non-learning baselines on exactly the same test pairs."
    )
    parser.add_argument("--model_log", required=True)
    parser.add_argument("--off_file", required=True)
    parser.add_argument("--file_folder", default=".")
    parser.add_argument("--test_pairs_file", default="")
    parser.add_argument("--train_pairs_file", default="")
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--in_feat", type=int, default=64)
    parser.add_argument("--highway_k", type=int, default=3)
    parser.add_argument("--transit_k", type=int, default=0)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--skip_highway", action="store_true")
    return parser


def resolve_from_root(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def parse_model_log(path):
    metrics = None
    test_pairs_path = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = TEST_METRICS_RE.search(line)
            if match:
                metrics = {
                    "mae": float(match.group("mae")),
                    "rmse": float(match.group("rmse")),
                    "relative_error": float(match.group("rel")),
                }
            match = TEST_PAIRS_RE.search(line)
            if match:
                test_pairs_path = match.group("path")
    if metrics is None:
        raise ValueError(f"No final test metrics found in {path}")
    return metrics, test_pairs_path


def constant_predictions(value, count):
    return [value] * count


def metric_row(method, metrics, count, uses_learning, uses_test_labels, mean_bias, notes):
    return {
        "scope": "overall",
        "method": method,
        "count": count,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "relative_error": metrics["relative_error"],
        "uses_learning": uses_learning,
        "uses_test_labels": uses_test_labels,
        "mean_bias": mean_bias,
        "notes": notes,
    }


def main():
    args = build_parser().parse_args()
    model_log = resolve_from_root(args.model_log)
    model_metrics, logged_test_pairs = parse_model_log(model_log)

    test_pairs_file = args.test_pairs_file or logged_test_pairs
    if not test_pairs_file:
        raise ValueError("No test-pairs path found; provide --test_pairs_file explicitly.")
    test_pairs_file = resolve_from_root(test_pairs_file)
    if not os.path.exists(test_pairs_file):
        raise FileNotFoundError(
            f"Aligned test pairs not found: {test_pairs_file}. "
            "Pass the exported CSV with --test_pairs_file."
        )

    base_folder = resolve_from_root(args.file_folder)
    off_path = args.off_file if os.path.isabs(args.off_file) else os.path.join(base_folder, args.off_file)
    vertices, _ = load_off(off_path)
    pairs = load_test_pairs(test_pairs_file)
    if not pairs:
        raise ValueError(f"No test pairs found in {test_pairs_file}")

    y_true = [distance for _, _, distance in pairs]
    test_mean = sum(y_true) / len(y_true)
    model_mean_bias = float("nan")
    rows = [
        metric_row(
            "model_euclidean_residual",
            model_metrics,
            len(pairs),
            "yes",
            "no",
            model_mean_bias,
            "Metrics parsed from the completed model log.",
        )
    ]

    train_pairs_file = args.train_pairs_file
    if not train_pairs_file and test_pairs_file.endswith("_test_pairs.csv"):
        candidate = test_pairs_file.replace("_test_pairs.csv", "_train_pairs.csv")
        if os.path.exists(candidate):
            train_pairs_file = candidate
    if train_pairs_file:
        train_pairs_file = resolve_from_root(train_pairs_file)
        train_pairs = load_test_pairs(train_pairs_file)
        if train_pairs:
            train_mean = sum(distance for _, _, distance in train_pairs) / len(train_pairs)
            preds = constant_predictions(train_mean, len(y_true))
            rows.append(metric_row(
                "train_mean_constant",
                compute_metrics(y_true, preds),
                len(pairs),
                "no",
                "no",
                sum(preds) / len(preds) - test_mean,
                "Formal constant baseline; mean estimated from training labels only.",
            ))

    test_mean_preds = constant_predictions(test_mean, len(y_true))
    rows.append(metric_row(
        "test_mean_oracle",
        compute_metrics(y_true, test_mean_preds),
        len(pairs),
        "no",
        "yes",
        0.0,
        "Diagnostic lower bound for constant predictors; not a formal baseline.",
    ))

    for method, predictor in (("euclidean_2d", euclidean_2d), ("euclidean_3d", euclidean_3d)):
        preds = [predictor(vertices, s, t) for s, t, _ in pairs]
        rows.append(metric_row(
            method,
            compute_metrics(y_true, preds),
            len(pairs),
            "no",
            "no",
            sum(preds) / len(preds) - test_mean,
            "Geometry-only baseline on the exact exported test pairs.",
        ))

    if not args.skip_highway:
        preds = highway_decomposition_preds(off_path, args, pairs, ROOT)
        rows.append(metric_row(
            "highway_decomp",
            compute_metrics(y_true, preds),
            len(pairs),
            "no",
            "no",
            sum(preds) / len(preds) - test_mean,
            "Non-learning highway decomposition baseline.",
        ))

    out_file = resolve_from_root(args.out_file)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fields = [
        "scope", "method", "count", "mae", "rmse", "relative_error",
        "uses_learning", "uses_test_labels", "mean_bias", "notes",
    ]
    with open(out_file, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aligned-baseline] test_pairs={test_pairs_file} count={len(pairs)}")
    for row in rows:
        print(
            f"[aligned-baseline] {row['method']:<28} "
            f"mae={row['mae']:.6f} rmse={row['rmse']:.6f} "
            f"relative_error={row['relative_error']:.6f}"
        )
    print(f"[aligned-baseline] wrote {out_file}")


if __name__ == "__main__":
    main()
