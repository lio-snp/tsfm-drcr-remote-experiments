#!/usr/bin/env python
"""Build a lightweight ex-ante failure predictor report for a GIFT-Eval rerun."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.features import feature_vector
from low_snr_tsfm.gift_eval_windowing import (
    forward_fill_nan,
    iter_univariate_targets,
    prediction_length,
    test_windows,
    window_count,
)


OUT_DIR = ROOT / "results" / "failure_mining"


def read_metrics(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["series_id"], int(row["window_index"])): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    tpr = float(np.mean(preds[positives] == 1)) if positives.any() else 0.0
    tnr = float(np.mean(preds[negatives] == 0)) if negatives.any() else 0.0
    return 0.5 * (tpr + tnr)


def best_threshold(rows: list[dict[str, object]], feature_names: list[str]) -> dict[str, object]:
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows])
    best: dict[str, object] = {
        "feature": "",
        "threshold": 0.0,
        "direction": "greater_equal",
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
    }
    for feature in feature_names:
        values = np.asarray([float(row[feature]) for row in rows])
        unique = np.unique(values[np.isfinite(values)])
        if unique.size == 0:
            continue
        thresholds = unique
        for threshold in thresholds:
            for direction in ["greater_equal", "less_equal"]:
                preds = (values >= threshold).astype(int)
                if direction == "less_equal":
                    preds = (values <= threshold).astype(int)
                accuracy = float(np.mean(preds == labels))
                bal = balanced_accuracy(labels, preds)
                if (bal, accuracy) > (
                    float(best["balanced_accuracy"]),
                    float(best["accuracy"]),
                ):
                    best = {
                        "feature": feature,
                        "threshold": float(threshold),
                        "direction": direction,
                        "accuracy": accuracy,
                        "balanced_accuracy": bal,
                    }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/gift-eval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="bizitobs_application")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--metrics", default="results/window_metrics/chronos_bolt_tiny_bizitobs_application_short_metrics.csv")
    parser.add_argument("--output-prefix", default="chronos_bolt_tiny_bizitobs_application_short")
    parser.add_argument("--max-series", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=15)
    parser.add_argument("--context-cap", type=int, default=2048)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency report
        raise SystemExit(f"datasets is required; run scripts/setup_chronos_env.sh first: {exc}") from exc

    metric_rows = read_metrics(ROOT / args.metrics)
    data_path = ROOT / args.data_path if args.data_path is not None else ROOT / args.data_root / args.dataset_name
    hf_dataset = load_from_disk(str(data_path))
    first = hf_dataset[0]
    freq = str(first["freq"])
    horizon = prediction_length(args.dataset_name, freq, args.term)
    raw_lengths = [np.asarray(item["target"], dtype=float).shape[-1] for item in hf_dataset]
    gift_windows = window_count(min(raw_lengths), horizon)
    windows_to_use = min(gift_windows, max(1, args.max_windows))

    feature_rows: list[dict[str, object]] = []
    processed_series = 0
    for item_idx, item in enumerate(hf_dataset):
        item_id = str(item.get("item_id", f"item_{item_idx}"))
        for series_id, values in iter_univariate_targets(item["target"], item_id):
            if processed_series >= args.max_series:
                break
            processed_series += 1
            for window in test_windows(values, horizon, gift_windows)[:windows_to_use]:
                key = (series_id, window.window_index)
                if key not in metric_rows:
                    continue
                context = forward_fill_nan(window.context)[-args.context_cap :]
                fv = feature_vector(context, horizon=horizon, context_length=context.size, period=horizon)
                metric = metric_rows[key]
                feature_rows.append(
                    {
                        "dataset": f"{args.dataset_name}/{freq}/{args.term}",
                        "series_id": series_id,
                        "window_index": window.window_index,
                        "origin": window.origin,
                        "context_length": int(context.size),
                        "failure_delta_005": int(float(metric["failure_delta_005"])),
                        "relative_error_ratio": float(metric["relative_error_ratio"]),
                        "flatness_score": float(metric["flatness_score"]),
                        "empirical_coverage_90": float(metric["empirical_coverage_90"]),
                        **fv,
                    }
                )
        if processed_series >= args.max_series:
            break

    if not feature_rows:
        raise SystemExit("No feature rows matched the rerun metrics")

    feature_names = [
        key
        for key in feature_rows[0]
        if key
        not in {
            "dataset",
            "series_id",
            "window_index",
            "origin",
            "context_length",
            "failure_delta_005",
            "relative_error_ratio",
            "flatness_score",
            "empirical_coverage_90",
        }
    ]
    labels = np.asarray([int(row["failure_delta_005"]) for row in feature_rows])
    report = {
        "status": "ok",
        "timestamp": int(time.time()),
        "dataset": f"{args.dataset_name}/{freq}/{args.term}",
        "n_windows": len(feature_rows),
        "failure_rate_delta_005": float(np.mean(labels)),
        "majority_class_accuracy": float(max(np.mean(labels), 1.0 - np.mean(labels))),
        "best_threshold": best_threshold(feature_rows, feature_names),
        "limitations": [
            "single-slice threshold predictor",
            "not cross-domain validated",
            "uses supplied rerun failure labels",
        ],
    }

    feature_path = OUT_DIR / f"{args.output_prefix}_predictor_features.csv"
    report_path = OUT_DIR / f"{args.output_prefix}_predictor_report.json"
    write_csv(feature_path, feature_rows)
    report["feature_table"] = str(feature_path)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
