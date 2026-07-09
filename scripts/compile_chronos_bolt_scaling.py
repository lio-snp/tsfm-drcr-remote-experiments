#!/usr/bin/env python
"""Compile Chronos-Bolt paired scaling metrics and bootstrap interactions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.scaling import (
    bootstrap_interactions,
    finite_float,
    log1p_nonnegative,
    model_rate_slope,
    trimmed_mean,
)


MODEL_META = {
    "chronos_bolt_tiny": {"size_label": "tiny", "params_m": 9.0, "size_rank": 0},
    "chronos_bolt_mini": {"size_label": "mini", "params_m": 21.0, "size_rank": 1},
    "chronos_bolt_small": {"size_label": "small", "params_m": 48.0, "size_rank": 2},
    "chronos_bolt_base": {"size_label": "base", "params_m": 205.0, "size_rank": 3},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def manifest_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row["dataset_name"]).lower(), str(row["series_id"]), int(row["window_index"]))
        lookup[key] = row
    return lookup


def annotate_metric_rows(metric_paths: list[Path], manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    manifest = manifest_lookup(manifest_rows)
    combined: list[dict[str, object]] = []
    missing_manifest_keys: list[dict[str, object]] = []
    for path in metric_paths:
        for row in read_csv(path):
            model = str(row["model"])
            meta = MODEL_META.get(model)
            if meta is None:
                continue
            dataset_name = str(row["dataset"]).split("/")[0].lower()
            key = (dataset_name, str(row["series_id"]), int(row["window_index"]))
            manifest_row = manifest.get(key)
            if manifest_row is None:
                missing_manifest_keys.append({"path": str(path), "key": key})
                continue
            enriched = dict(row)
            enriched.update(meta)
            enriched["params_m"] = float(meta["params_m"])
            enriched["log10_params_m"] = math.log10(float(meta["params_m"]))
            enriched["target_key"] = manifest_row["target_key"]
            enriched["target_role"] = manifest_row["target_role"]
            enriched["paired_unit_id"] = manifest_row["paired_unit_id"]
            enriched["selection_order"] = int(manifest_row["selection_order"])
            enriched["metric_path"] = str(path)
            enriched["log1p_forecast_variance_ratio"] = log1p_nonnegative(row.get("forecast_variance_ratio"))
            enriched["log1p_prediction_amplitude_ratio"] = log1p_nonnegative(row.get("prediction_amplitude_ratio"))
            combined.append(enriched)
    if missing_manifest_keys:
        preview = missing_manifest_keys[:5]
        raise ValueError(f"{len(missing_manifest_keys)} metric rows were not in manifest; preview={preview}")
    return combined


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = [finite_float(row.get(key)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_key"]), str(row["model"]))].append(row)

    summary: list[dict[str, object]] = []
    for (target_key, model), group in sorted(grouped.items()):
        rer = finite_values(group, "relative_error_ratio")
        nonfragile = [row for row in group if finite_float(row.get("baseline_mae")) >= 1e-6]
        nonfragile_rer = finite_values(nonfragile, "relative_error_ratio")
        denominator_fragile = len(group) - len(nonfragile)
        log_fvr = finite_values(group, "log1p_forecast_variance_ratio")
        log_par = finite_values(group, "log1p_prediction_amplitude_ratio")
        summary.append(
            {
                "target_key": target_key,
                "target_role": group[0]["target_role"],
                "dataset": group[0]["dataset"],
                "model": model,
                "size_label": group[0]["size_label"],
                "size_rank": group[0]["size_rank"],
                "params_m": group[0]["params_m"],
                "log10_params_m": group[0]["log10_params_m"],
                "n_windows": len(group),
                "n_series": len({row["series_id"] for row in group}),
                "failure_rate_delta_005": statistics.mean(finite_values(group, "failure_delta_005")),
                "excess_variance_rate": statistics.mean(finite_values(group, "excess_variance")),
                "over_smoothing_rate": statistics.mean(finite_values(group, "over_smoothing")),
                "mean_relative_error_ratio": statistics.mean(rer) if rer else float("nan"),
                "median_relative_error_ratio": median(rer),
                "nonfragile_median_relative_error_ratio": median(nonfragile_rer),
                "denominator_fragile_windows": denominator_fragile,
                "denominator_fragile_rate": denominator_fragile / len(group),
                "mean_empirical_coverage_90": statistics.mean(finite_values(group, "empirical_coverage_90")),
                "mean_forecast_variance_ratio": statistics.mean(finite_values(group, "forecast_variance_ratio")),
                "mean_prediction_amplitude_ratio": statistics.mean(finite_values(group, "prediction_amplitude_ratio")),
                "median_log1p_forecast_variance_ratio": median(log_fvr),
                "trimmed_mean_log1p_forecast_variance_ratio": trimmed_mean(log_fvr),
                "median_log1p_prediction_amplitude_ratio": median(log_par),
                "trimmed_mean_log1p_prediction_amplitude_ratio": trimmed_mean(log_par),
            }
        )
    return summary


def slope_rows(rows: list[dict[str, object]], outcomes: list[str]) -> list[dict[str, object]]:
    targets = sorted({str(row["target_key"]) for row in rows})
    out: list[dict[str, object]] = []
    for target_key in targets:
        target_rows = [row for row in rows if row["target_key"] == target_key]
        for outcome in outcomes:
            out.append(
                {
                    "target_key": target_key,
                    "target_role": target_rows[0]["target_role"],
                    "outcome": outcome,
                    "x_key": "log10_params_m",
                    "slope": model_rate_slope(target_rows, outcome=outcome),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/scaling/chronos_bolt_scaling_manifest.csv")
    parser.add_argument("--metrics-glob", default="results/window_metrics/chronos_bolt_*_scaling_*_metrics.csv")
    parser.add_argument("--output-dir", default="results/scaling")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir
    metric_paths = [Path(path) for path in sorted(glob.glob(str(ROOT / args.metrics_glob)))]
    if not metric_paths:
        raise SystemExit(f"No metric files matched {args.metrics_glob}")

    manifest_rows = read_csv(manifest_path)
    combined = annotate_metric_rows(metric_paths, manifest_rows)
    summary = summarize(combined)
    slopes = slope_rows(
        combined,
        outcomes=[
            "excess_variance",
            "relative_error_ratio",
            "over_smoothing",
            "empirical_coverage_90",
            "log1p_forecast_variance_ratio",
            "log1p_prediction_amplitude_ratio",
        ],
    )
    target_keys = sorted({str(row["target_key"]) for row in combined})
    failure_targets = sorted({str(row["target_key"]) for row in combined if row["target_role"] == "failure"})
    control_targets = sorted({str(row["target_key"]) for row in combined if row["target_role"] == "positive_control"})
    if failure_targets != ["covid_deaths_d_short"]:
        raise ValueError(f"Expected covid_deaths_d_short as the single failure target, got {failure_targets}")
    if not control_targets:
        raise ValueError("At least one positive-control target is required")

    bootstrap = bootstrap_interactions(
        combined,
        failure_target_key="covid_deaths_d_short",
        control_target_keys=control_targets,
        outcome="excess_variance",
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    robust_bootstrap = bootstrap_interactions(
        combined,
        failure_target_key="covid_deaths_d_short",
        control_target_keys=control_targets,
        outcome="log1p_forecast_variance_ratio",
        n_bootstrap=args.bootstrap,
        seed=args.seed + 100,
    )
    bootstrap.update(
        {
            "primary_outcome": "excess_variance_rate_log_param_slope_interaction",
            "robust_continuous_primary_sensitivity": robust_bootstrap,
            "framing": (
                "Tests whether an ERM-derived forecast-collapse mechanism extends to "
                "pretrained zero-shot Chronos-Bolt capacity scaling; it is not a direct "
                "replication of the theorem setting."
            ),
            "targets": target_keys,
            "models": sorted({str(row["model"]) for row in combined}),
            "generated_at": int(time.time()),
        }
    )

    write_csv(output_dir / "chronos_bolt_scaling_window_metrics.csv", combined)
    write_csv(output_dir / "chronos_bolt_scaling_summary.csv", summary)
    write_csv(output_dir / "chronos_bolt_scaling_slopes.csv", slopes)
    (output_dir / "chronos_bolt_scaling_bootstrap.json").write_text(json.dumps(bootstrap, indent=2))
    print(f"Wrote Chronos-Bolt scaling summary for {len(combined)} paired metric rows")


if __name__ == "__main__":
    main()
