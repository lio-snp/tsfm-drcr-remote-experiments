#!/usr/bin/env python
"""Build the fixed paired-window manifest for Chronos-Bolt scaling tests."""

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

from low_snr_tsfm.gift_eval_windowing import (
    iter_univariate_targets,
    prediction_length,
    test_windows,
    window_count,
)


SCALING_TARGETS = {
    "covid_deaths": {
        "target_key": "covid_deaths_d_short",
        "dataset_name": "covid_deaths",
        "data_path": "data/gift-eval/covid_deaths",
        "term": "short",
        "baseline_mode": "auto_ets",
        "baseline_context_cap": 512,
        "baseline_season_length": 1,
        "context_cap": 2048,
        "target_role": "failure",
        "domain": "Healthcare",
        "regime": "low_local_structure_count_decay",
    },
    "solar": {
        "target_key": "solar_10t_short",
        "dataset_name": "solar",
        "data_path": "data/gift-eval/solar/10T",
        "term": "short",
        "baseline_mode": "seasonal_naive",
        "baseline_context_cap": 2048,
        "baseline_season_length": 48,
        "context_cap": 2048,
        "target_role": "positive_control",
        "domain": "Energy",
        "regime": "seasonal_high_structure",
    },
    "loop_seattle": {
        "target_key": "loop_seattle_h_short",
        "dataset_name": "loop_seattle",
        "data_path": "data/gift-eval/LOOP_SEATTLE/H",
        "term": "short",
        "baseline_mode": "seasonal_naive",
        "baseline_context_cap": 2048,
        "baseline_season_length": 48,
        "context_cap": 2048,
        "target_role": "positive_control",
        "domain": "Transport",
        "regime": "medium_snr_persistent",
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_rows(config: dict[str, object], max_units: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    from datasets import load_from_disk

    data_path = ROOT / str(config["data_path"])
    hf_dataset = load_from_disk(str(data_path))
    first = hf_dataset[0]
    freq = str(first["freq"])
    horizon = prediction_length(str(config["dataset_name"]), freq, str(config["term"]))

    raw_lengths: list[int] = []
    for item in hf_dataset:
        target = np.asarray(item["target"], dtype=float)
        raw_lengths.append(target.shape[-1])
    gift_windows = window_count(min(raw_lengths), horizon)

    candidates: list[dict[str, object]] = []
    for item_idx, item in enumerate(hf_dataset):
        item_id = str(item.get("item_id", f"item_{item_idx}"))
        for series_id, values in iter_univariate_targets(item["target"], item_id):
            for window in test_windows(values, horizon, gift_windows):
                candidates.append(
                    {
                        **config,
                        "freq": freq,
                        "horizon": horizon,
                        "series_id": series_id,
                        "series_order": item_idx,
                        "window_index": window.window_index,
                        "origin": window.origin,
                        "full_context_length": len(window.context),
                        "paired_unit_id": f"{config['target_key']}:{series_id}:{window.window_index}",
                    }
                )

    candidates.sort(key=lambda row: (int(row["window_index"]), int(row["series_order"]), str(row["series_id"])))
    selected = candidates if max_units <= 0 else candidates[:max_units]
    for order, row in enumerate(selected):
        row["selection_order"] = order

    status = {
        "target_key": config["target_key"],
        "dataset_name": config["dataset_name"],
        "data_path": str(data_path),
        "freq": freq,
        "term": config["term"],
        "horizon": horizon,
        "gift_eval_windows_per_series": gift_windows,
        "available_units": len(candidates),
        "selected_units": len(selected),
        "selected_series": len({row["series_id"] for row in selected}),
        "selection": "deterministic round-robin by window_index then series_order",
    }
    return selected, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaling/chronos_bolt_scaling_manifest.csv")
    parser.add_argument("--status", default="results/scaling/chronos_bolt_scaling_manifest_status.json")
    parser.add_argument("--targets", default="covid_deaths,solar,loop_seattle")
    parser.add_argument("--max-units-per-target", type=int, default=32)
    args = parser.parse_args()

    selected_targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    rows: list[dict[str, object]] = []
    target_statuses: list[dict[str, object]] = []
    for target in selected_targets:
        if target not in SCALING_TARGETS:
            raise ValueError(f"Unknown target {target}; choose from {sorted(SCALING_TARGETS)}")
        target_manifest, status = target_rows(SCALING_TARGETS[target], args.max_units_per_target)
        rows.extend(target_manifest)
        target_statuses.append(status)

    output_path = ROOT / args.output
    status_path = ROOT / args.status
    write_csv(output_path, rows)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "timestamp": int(time.time()),
                "manifest": str(output_path),
                "targets": target_statuses,
                "max_units_per_target": args.max_units_per_target,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(rows)} paired window rows to {output_path}")


if __name__ == "__main__":
    main()
