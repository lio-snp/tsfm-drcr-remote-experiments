#!/usr/bin/env python
"""Create a locked GIFT-Eval aggregate reproduction slice.

This script ingests public GIFT-Eval result CSVs from the shallow-cloned
`external/gift-eval` repository and produces local aggregate failure-mining
artifacts. It deliberately does not claim window-level degeneration; that is
reserved for raw forecast artifacts defined in docs/forecast_artifact_contract.md.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
GIFT_RESULTS = EXTERNAL / "gift-eval" / "results"
REPRO_DIR = ROOT / "results" / "reproduction"
FAILURE_DIR = ROOT / "results" / "failure_mining"

MASE = "eval_metrics/MASE[0.5]"
WQL = "eval_metrics/mean_weighted_sum_quantile_loss"
MAE = "eval_metrics/MAE[0.5]"
RMSE = "eval_metrics/RMSE[mean]"


REPOS = {
    "gift_eval": EXTERNAL / "gift-eval",
    "chronos": EXTERNAL / "chronos-forecasting",
    "timesfm": EXTERNAL / "timesfm",
    "uni2ts": EXTERNAL / "uni2ts",
}

MODEL_DIRS = {
    "chronos_bolt_small": "chronos_bolt_small",
    "timesfm_2_0_500m": "timesfm_2_0_500m",
    "timesfm_2_5": "TimesFM-2.5",
    "moirai_small": "moirai_small",
    "moirai2": "Moirai2",
}

BASELINE_DIRS = {
    "naive": "naive",
    "seasonal_naive": "seasonal_naive",
    "auto_ets": "auto_ets",
    "auto_arima": "auto_arima",
}

LOCKED_DATASETS = {
    "seasonal_high_structure": [
        "electricity/W/short",
        "solar/10T/short",
        "ett1/H/short",
    ],
    "intermittent_bursty": [
        "car_parts/M/short",
        "bizitobs_application/10S/short",
        "bitbrains_rnd/5T/short",
    ],
    "noisy_low_signal": [
        "m4_weekly/W/short",
        "m4_hourly/H/short",
        "m4_monthly/M/short",
    ],
    "medium_snr_persistent": [
        "hospital/M/short",
        "covid_deaths/D/short",
        "loop_seattle/H/short",
    ],
}


def git_metadata(path: Path) -> dict[str, str]:
    if not (path / ".git").exists():
        return {"path": str(path), "present": "false"}
    return {
        "path": str(path),
        "present": "true",
        "remote": subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip(),
        "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(path), "branch", "--show-current"], text=True).strip(),
    }


def read_result_dir(result_dir: Path, label: str) -> list[dict[str, str]]:
    path = result_dir / "all_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["local_model_key"] = label
        row["source_result_dir"] = result_dir.name
    return rows


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


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    repo_meta = {name: git_metadata(path) for name, path in REPOS.items()}
    (REPRO_DIR / "upstream_repos.json").write_text(json.dumps(repo_meta, indent=2))

    all_rows: list[dict[str, str]] = []
    for label, dirname in {**MODEL_DIRS, **BASELINE_DIRS}.items():
        all_rows.extend(read_result_dir(GIFT_RESULTS / dirname, label))

    dataset_to_regime = {
        dataset: regime for regime, datasets in LOCKED_DATASETS.items() for dataset in datasets
    }
    locked = []
    for row in all_rows:
        regime = dataset_to_regime.get(row["dataset"])
        if regime is None:
            continue
        locked_row = dict(row)
        locked_row["regime"] = regime
        locked.append(locked_row)

    write_csv(REPRO_DIR / "gift_eval_locked_slice_scores.csv", locked)

    baseline_by_dataset: dict[str, dict[str, str]] = {}
    for dataset in dataset_to_regime:
        candidates = [
            row
            for row in locked
            if row["dataset"] == dataset and row["local_model_key"] in BASELINE_DIRS
        ]
        if not candidates:
            continue
        baseline_by_dataset[dataset] = min(candidates, key=lambda row: f(row, MASE))

    failures = []
    for row in locked:
        model_key = row["local_model_key"]
        if model_key in BASELINE_DIRS:
            continue
        baseline = baseline_by_dataset[row["dataset"]]
        mase_ratio = f(row, MASE) / max(f(baseline, MASE), 1e-12)
        wql_ratio = f(row, WQL) / max(f(baseline, WQL), 1e-12)
        failures.append(
            {
                "dataset": row["dataset"],
                "domain": row["domain"],
                "regime": row["regime"],
                "model": row["model"],
                "local_model_key": model_key,
                "best_baseline": baseline["local_model_key"],
                "model_mase": f(row, MASE),
                "baseline_mase": f(baseline, MASE),
                "mase_relative_error_ratio": mase_ratio,
                "model_wql": f(row, WQL),
                "baseline_wql": f(baseline, WQL),
                "wql_relative_error_ratio": wql_ratio,
                "failure_delta_0": int(mase_ratio > 1.0),
                "failure_delta_005": int(mase_ratio > 1.05),
                "failure_delta_010": int(mase_ratio > 1.10),
                "severe_failure_125": int(mase_ratio > 1.25),
                "aggregate_only": 1,
            }
        )
    write_csv(FAILURE_DIR / "gift_eval_aggregate_failures.csv", failures)

    summary_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in failures:
        summary_groups[(str(row["local_model_key"]), str(row["regime"]))].append(row)
    summary = []
    for (model, regime), rows in sorted(summary_groups.items()):
        n = len(rows)
        summary.append(
            {
                "local_model_key": model,
                "regime": regime,
                "n_datasets": n,
                "mean_mase_relative_error_ratio": sum(float(r["mase_relative_error_ratio"]) for r in rows) / n,
                "failure_rate_delta_0": sum(int(r["failure_delta_0"]) for r in rows) / n,
                "failure_rate_delta_005": sum(int(r["failure_delta_005"]) for r in rows) / n,
                "failure_rate_delta_010": sum(int(r["failure_delta_010"]) for r in rows) / n,
                "severe_failure_rate_125": sum(int(r["severe_failure_125"]) for r in rows) / n,
            }
        )
    write_csv(FAILURE_DIR / "gift_eval_aggregate_failure_summary.csv", summary)

    manifest = {
        "benchmark": "GIFT-Eval",
        "artifact_layer": "aggregate_results",
        "models": MODEL_DIRS,
        "baselines": BASELINE_DIRS,
        "locked_datasets": LOCKED_DATASETS,
        "outputs": {
            "scores": str(REPRO_DIR / "gift_eval_locked_slice_scores.csv"),
            "failures": str(FAILURE_DIR / "gift_eval_aggregate_failures.csv"),
            "summary": str(FAILURE_DIR / "gift_eval_aggregate_failure_summary.csv"),
        },
        "limitations": [
            "Aggregate GIFT-Eval results do not contain raw forecast paths.",
            "Degeneration metrics requiring forecast trajectories need window-level reruns.",
            "This slice is an anchor for reproduction and target selection, not final evidence.",
        ],
    }
    (REPRO_DIR / "gift_eval_locked_slice_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(locked)} locked score rows")
    print(f"Wrote {len(failures)} aggregate TSFM-vs-baseline comparisons")
    print(f"Wrote {len(summary)} model/regime summary rows")


if __name__ == "__main__":
    main()
