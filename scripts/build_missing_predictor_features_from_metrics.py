#!/usr/bin/env python3
"""Backfill predictor feature files from existing feature banks and metrics.

The ex-ante local-structure features are dataset/window properties, not model
properties.  When a rerun has window metrics but lacks the corresponding
`*_predictor_features.csv`, this script copies matching ex-ante features from
another source with the same dataset/series/window key and overwrites the
model-dependent labels/metrics from the rerun's own window-metrics file.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "results" / "failure_mining"
METRIC_DIR = ROOT / "results" / "window_metrics"

MODEL_DEPENDENT_KEYS = {
    "failure_delta_005",
    "relative_error_ratio",
    "flatness_score",
    "empirical_coverage_90",
    "context_length",
    "origin",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def norm_dataset(value: str) -> str:
    return value.strip().lower()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (norm_dataset(row.get("dataset", "")), str(row.get("series_id", "")), str(row.get("window_index", "")))


def build_feature_bank() -> dict[tuple[str, str, str], dict[str, str]]:
    bank: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in sorted(FEATURE_DIR.glob("*_predictor_features.csv")):
        for row in read_csv(path):
            row_key = key(row)
            if row_key[0] and row_key[1] and row_key not in bank:
                bank[row_key] = row
    return bank


def backfill_slug(slug: str, bank: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, object]:
    metric_path = METRIC_DIR / f"{slug}_metrics.csv"
    if not metric_path.exists():
        return {"slug": slug, "status": "missing_metrics", "rows": 0, "matched": 0}
    output_path = FEATURE_DIR / f"{slug}_predictor_features.csv"
    rows: list[dict[str, object]] = []
    missing = 0
    for metric in read_csv(metric_path):
        template = bank.get(key(metric))
        if template is None:
            missing += 1
            continue
        row = dict(template)
        row.update(
            {
                "dataset": metric.get("dataset", row.get("dataset", "")),
                "series_id": metric.get("series_id", row.get("series_id", "")),
                "window_index": metric.get("window_index", row.get("window_index", "")),
                "origin": metric.get("origin", row.get("origin", "")),
                "context_length": metric.get("context_length", row.get("context_length", "")),
                "failure_delta_005": metric.get("failure_delta_005", row.get("failure_delta_005", "")),
                "relative_error_ratio": metric.get("relative_error_ratio", row.get("relative_error_ratio", "")),
                "flatness_score": metric.get("flatness_score", row.get("flatness_score", "")),
                "empirical_coverage_90": metric.get("empirical_coverage_90", row.get("empirical_coverage_90", "")),
                "feature_source": "backfilled_from_existing_feature_bank",
            }
        )
        for metric_key in [
            "baseline_mae",
            "baseline_mase",
            "baseline_mode",
            "domain",
            "forecast_variance_ratio",
            "horizon",
            "mae",
            "mase",
            "rmse",
            "spike_recall",
        ]:
            if metric_key in metric:
                row[metric_key] = metric[metric_key]
        rows.append(row)
    if rows:
        write_csv(output_path, rows)
    return {
        "slug": slug,
        "status": "ok" if rows else "no_matches",
        "rows": len(read_csv(metric_path)),
        "matched": len(rows),
        "missing": missing,
        "output": str(output_path.relative_to(ROOT)) if rows else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="+")
    args = parser.parse_args()
    bank = build_feature_bank()
    results = [backfill_slug(slug, bank) for slug in args.slugs]
    report = {
        "status": "ok" if all(item["status"] == "ok" for item in results) else "partial",
        "timestamp": int(time.time()),
        "feature_bank_size": len(bank),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
