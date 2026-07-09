#!/usr/bin/env python
"""Mine second-family TSFM aggregate failures from public GIFT-Eval results."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.gift_eval_results import (  # noqa: E402
    build_failure_rows,
    read_result_dir,
    shared_failure_rows,
    summarize_failures,
)


GIFT_RESULTS = ROOT / "external" / "gift-eval" / "results"
REPRO_DIR = ROOT / "results" / "reproduction"
FAILURE_DIR = ROOT / "results" / "failure_mining"

MODEL_DIRS = {
    "chronos_bolt_small": ("chronos_bolt_small", "Chronos"),
    "timesfm_2_0_500m": ("timesfm_2_0_500m", "TimesFM"),
    "timesfm_2_5": ("TimesFM-2.5", "TimesFM"),
    "moirai_small": ("moirai_small", "Moirai"),
    "moirai2": ("Moirai2", "Moirai"),
}

SECOND_FAMILY_KEYS = {
    "timesfm_2_0_500m",
    "timesfm_2_5",
    "moirai_small",
    "moirai2",
}

SHARED_FAILURE_KEYS = SECOND_FAMILY_KEYS | {"chronos_bolt_small"}

BASELINE_DIRS = {
    "naive": ("naive", "Baseline"),
    "seasonal_naive": ("seasonal_naive", "Baseline"),
    "auto_ets": ("auto_ets", "Baseline"),
    "auto_arima": ("auto_arima", "Baseline"),
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


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, (dirname, family) in {**MODEL_DIRS, **BASELINE_DIRS}.items():
        rows.extend(read_result_dir(GIFT_RESULTS / dirname, key, family))
    return rows


def main() -> None:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_to_regime = {
        dataset: regime for regime, datasets in LOCKED_DATASETS.items() for dataset in datasets
    }
    rows = load_rows()
    baseline_keys = set(BASELINE_DIRS)

    second_family_failures = build_failure_rows(
        rows,
        model_keys=SECOND_FAMILY_KEYS,
        baseline_keys=baseline_keys,
        dataset_to_regime=dataset_to_regime,
    )
    second_family_failures.sort(
        key=lambda row: (
            int(row["locked_slice"]),
            int(row["failure_delta_005"]),
            float(row["mase_relative_error_ratio"]),
        ),
        reverse=True,
    )
    second_family_summary = summarize_failures(second_family_failures)

    shared_source = build_failure_rows(
        rows,
        model_keys=SHARED_FAILURE_KEYS,
        baseline_keys=baseline_keys,
        dataset_to_regime=dataset_to_regime,
    )
    shared_failures = shared_failure_rows(shared_source, min_failed_families=2)

    failures_path = FAILURE_DIR / "gift_eval_second_family_failures.csv"
    summary_path = FAILURE_DIR / "gift_eval_second_family_summary.csv"
    shared_path = FAILURE_DIR / "gift_eval_shared_family_failures.csv"
    manifest_path = REPRO_DIR / "second_family_aggregate_manifest.json"

    write_csv(failures_path, second_family_failures)
    write_csv(summary_path, second_family_summary)
    write_csv(shared_path, shared_failures)

    manifest = {
        "benchmark": "GIFT-Eval",
        "artifact_layer": "aggregate_results",
        "purpose": "Second-family TimesFM/Moirai failure mining and shared-failure target selection.",
        "models": MODEL_DIRS,
        "second_family_keys": sorted(SECOND_FAMILY_KEYS),
        "baselines": BASELINE_DIRS,
        "locked_datasets": LOCKED_DATASETS,
        "gift_eval_commit": git_commit(ROOT / "external" / "gift-eval"),
        "timesfm_commit": git_commit(ROOT / "external" / "timesfm"),
        "uni2ts_commit": git_commit(ROOT / "external" / "uni2ts"),
        "outputs": {
            "second_family_failures": str(failures_path),
            "second_family_summary": str(summary_path),
            "shared_family_failures": str(shared_path),
        },
        "counts": {
            "second_family_comparisons": len(second_family_failures),
            "second_family_failures_delta_005": sum(
                int(row["failure_delta_005"]) for row in second_family_failures
            ),
            "locked_second_family_comparisons": sum(
                int(row["locked_slice"]) for row in second_family_failures
            ),
            "shared_failure_datasets": len(shared_failures),
            "locked_shared_failure_datasets": sum(int(row["locked_slice"]) for row in shared_failures),
        },
        "limitations": [
            "Aggregate GIFT-Eval rows do not include raw forecast trajectories.",
            "Second-family aggregate failures can prioritize reruns but cannot prove degeneration type.",
            "Raw TimesFM/Moirai reruns remain pending when local memory or dependency constraints block safe loading.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
