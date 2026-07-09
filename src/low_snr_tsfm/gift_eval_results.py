"""Helpers for mining public GIFT-Eval aggregate result CSVs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


MASE = "eval_metrics/MASE[0.5]"
WQL = "eval_metrics/mean_weighted_sum_quantile_loss"
MAE = "eval_metrics/MAE[0.5]"
RMSE = "eval_metrics/RMSE[mean]"


def read_result_dir(result_dir: Path, local_model_key: str, model_family: str) -> list[dict[str, str]]:
    path = result_dir / "all_results.csv"
    if not path.exists():
        path = result_dir / "all-results.csv"
    if not path.exists():
        raise FileNotFoundError(result_dir / "all_results.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["local_model_key"] = local_model_key
        row["model_family"] = model_family
        row["source_result_dir"] = result_dir.name
    return rows


def metric_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def best_baseline_by_dataset(
    rows: list[dict[str, str]],
    baseline_keys: set[str],
    metric: str = MASE,
) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("local_model_key") not in baseline_keys:
            continue
        score = metric_float(row, metric)
        if not math.isfinite(score):
            continue
        dataset = row["dataset"]
        current = best.get(dataset)
        if current is None or score < metric_float(current, metric):
            best[dataset] = row
    return best


def build_failure_rows(
    rows: list[dict[str, str]],
    model_keys: set[str],
    baseline_keys: set[str],
    dataset_to_regime: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    dataset_to_regime = dataset_to_regime or {}
    baselines = best_baseline_by_dataset(rows, baseline_keys)
    failures: list[dict[str, object]] = []
    for row in rows:
        model_key = row.get("local_model_key", "")
        if model_key not in model_keys:
            continue
        baseline = baselines.get(row["dataset"])
        if baseline is None:
            continue
        model_mase = metric_float(row, MASE)
        baseline_mase = metric_float(baseline, MASE)
        model_wql = metric_float(row, WQL)
        baseline_wql = metric_float(baseline, WQL)
        if not math.isfinite(model_mase) or not math.isfinite(baseline_mase):
            continue
        mase_ratio = model_mase / max(baseline_mase, 1e-12)
        wql_ratio = (
            model_wql / max(baseline_wql, 1e-12)
            if math.isfinite(model_wql) and math.isfinite(baseline_wql)
            else float("nan")
        )
        dataset = row["dataset"]
        failures.append(
            {
                "dataset": dataset,
                "domain": row.get("domain", ""),
                "regime": dataset_to_regime.get(dataset, "outside_locked_slice"),
                "locked_slice": int(dataset in dataset_to_regime),
                "model": row.get("model", model_key),
                "model_family": row.get("model_family", ""),
                "local_model_key": model_key,
                "best_baseline": baseline["local_model_key"],
                "baseline_model": baseline.get("model", baseline["local_model_key"]),
                "model_mase": model_mase,
                "baseline_mase": baseline_mase,
                "mase_relative_error_ratio": mase_ratio,
                "model_wql": model_wql,
                "baseline_wql": baseline_wql,
                "wql_relative_error_ratio": wql_ratio,
                "failure_delta_0": int(mase_ratio > 1.0),
                "failure_delta_005": int(mase_ratio > 1.05),
                "failure_delta_010": int(mase_ratio > 1.10),
                "severe_failure_125": int(mase_ratio > 1.25),
                "aggregate_only": 1,
            }
        )
    return failures


def summarize_failures(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...] = ("model_family", "local_model_key"),
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)

    summary: list[dict[str, object]] = []
    for group, items in sorted(groups.items()):
        n = len(items)
        entry = {key: value for key, value in zip(group_keys, group)}
        entry.update(
            {
                "n_datasets": n,
                "n_locked_datasets": sum(int(item.get("locked_slice", 0)) for item in items),
                "mean_mase_relative_error_ratio": sum(
                    float(item["mase_relative_error_ratio"]) for item in items
                )
                / n,
                "max_mase_relative_error_ratio": max(
                    float(item["mase_relative_error_ratio"]) for item in items
                ),
                "failure_rate_delta_0": sum(int(item["failure_delta_0"]) for item in items) / n,
                "failure_rate_delta_005": sum(int(item["failure_delta_005"]) for item in items) / n,
                "failure_rate_delta_010": sum(int(item["failure_delta_010"]) for item in items) / n,
                "severe_failure_rate_125": sum(int(item["severe_failure_125"]) for item in items) / n,
            }
        )
        summary.append(entry)
    return summary


def shared_failure_rows(
    rows: list[dict[str, object]],
    min_failed_families: int = 2,
    failure_key: str = "failure_delta_005",
) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row.get(failure_key, 0)) == 1:
            by_dataset[str(row["dataset"])].append(row)

    shared: list[dict[str, object]] = []
    for dataset, items in by_dataset.items():
        families = sorted({str(item.get("model_family", "")) for item in items})
        if len(families) < min_failed_families:
            continue
        ratios = [float(item["mase_relative_error_ratio"]) for item in items]
        models = sorted({str(item.get("local_model_key", "")) for item in items})
        first = items[0]
        shared.append(
            {
                "dataset": dataset,
                "domain": first.get("domain", ""),
                "regime": first.get("regime", ""),
                "locked_slice": first.get("locked_slice", 0),
                "failed_families": ",".join(families),
                "failed_models": ",".join(models),
                "n_failed_families": len(families),
                "n_failed_models": len(models),
                "max_mase_relative_error_ratio": max(ratios),
                "mean_mase_relative_error_ratio": sum(ratios) / len(ratios),
                "best_baseline": first.get("best_baseline", ""),
                "aggregate_only": 1,
            }
        )
    return sorted(
        shared,
        key=lambda row: (
            int(row["locked_slice"]),
            int(row["n_failed_families"]),
            float(row["max_mase_relative_error_ratio"]),
        ),
        reverse=True,
    )
