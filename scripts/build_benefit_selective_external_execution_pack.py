#!/usr/bin/env python3
"""Build the frozen, strictly serial execution pack for external DRCR evidence."""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.gift_eval_windowing import prediction_length


PROTOCOL = ROOT / "configs" / "benefit_selective_drcr_external_protocol.json"
MANIFEST = ROOT / "results" / "aaai_stress" / "benefit_selective_external_manifest.csv"
OUT = ROOT / "results" / "aaai_stress" / "benefit_selective_external_execution_jobs.csv"
DOC = ROOT / "docs" / "benefit_selective_external_execution_pack.md"

HUB_DATASET_NAMES = {
    "car_parts": "car_parts_with_missing",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE",
    "temperature_rain": "temperature_rain_with_missing",
}
MODEL_CONTRACT = {
    "chronos": {
        "python": ".venv-chronos/bin/python",
        "runner": "scripts/run_chronos_bolt_gift_eval_raw.py",
        "model_id": "amazon/chronos-bolt-small",
        "model_name": "chronos_bolt_small",
    },
    "moirai": {
        "python": ".venv-moirai/bin/python",
        "runner": "scripts/run_moirai_gift_eval_raw.py",
        "model_id": "Salesforce/moirai-1.1-R-small",
        "model_name": "moirai_1_1_small",
    },
    "timesfm": {
        "python": ".venv-chronos/bin/python",
        "runner": "scripts/run_timesfm_gift_eval_raw.py",
        "model_id": "google/timesfm-2.5-200m-pytorch",
        "model_name": "timesfm_2_5",
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quote_command(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    evaluation = protocol["evaluation"]
    source_rows = list(csv.DictReader(MANIFEST.open()))
    jobs: list[dict[str, object]] = []

    for source in source_rows:
        base, freq, term = source["dataset"].split("/")
        family = source["family"]
        contract = MODEL_CONTRACT[family]
        hub_name = HUB_DATASET_NAMES.get(base, base)
        relative_data_path = Path("data/gift-eval") / hub_name / freq
        short_season = prediction_length(base, freq, "short")
        output_slug = f"external_v1_{family}_{base}_{freq}_{term}".lower().replace("/", "_")
        status_path = f"results/raw_forecasts/{output_slug}_status.json"

        command = [
            contract["python"],
            contract["runner"],
            "--dataset-name", hub_name,
            "--data-path", str(relative_data_path),
            "--term", term,
            "--model-id", contract["model_id"],
            "--model-name", contract["model_name"],
            "--max-series", str(evaluation["series_per_model_domain"]),
            "--max-windows", str(evaluation["windows_per_series"]),
            "--context-cap", str(evaluation["context_cap"]),
            "--baseline-mode", "rolling_pre_origin",
            "--baseline-context-cap", str(evaluation["classical_context_cap"]),
            "--baseline-season-length", str(short_season),
            "--domain", source["domain"],
            "--regime", "external_locked",
            "--output-slug", output_slug,
            "--export-history-sidecar",
        ]
        if family == "chronos":
            command.extend(["--quantile-levels", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"])
        elif family == "moirai":
            command.extend([
                "--model-family", "moirai1",
                "--num-samples", str(evaluation["moirai_samples"]),
                "--seed", "7",
            ])

        download = [
            ".venv-chronos/bin/python",
            "scripts/download_gift_eval_subset.py",
            "--dataset-name", hub_name,
        ]
        jobs.append({
            "job_id": output_slug,
            "domain": source["domain"],
            "dataset": source["dataset"],
            "dataset_name": base,
            "hub_dataset_name": hub_name,
            "frequency": freq,
            "term": term,
            "family": family,
            "model": source["model"],
            "model_id": contract["model_id"],
            "windows_requested": evaluation["windows_per_model_domain"],
            "series_requested": evaluation["series_per_model_domain"],
            "windows_per_series": evaluation["windows_per_series"],
            "context_cap": evaluation["context_cap"],
            "baseline_context_cap": evaluation["classical_context_cap"],
            "baseline_validation_folds": evaluation["classical_validation_folds"],
            "baseline_season_length": short_season,
            "moirai_samples": evaluation["moirai_samples"] if family == "moirai" else "",
            "quantile_grid": source["quantile_grid"],
            "data_path": str(relative_data_path),
            "data_root_path": str(Path("data/gift-eval") / hub_name),
            "output_slug": output_slug,
            "status_path": status_path,
            "download_command": quote_command(download),
            "command": quote_command(command),
            "status": "ready_not_run",
        })

    if len(jobs) != 42:
        raise ValueError(f"Expected 42 frozen jobs, found {len(jobs)}")
    write_csv(OUT, jobs)

    family_counts = {family: sum(row["family"] == family for row in jobs) for family in MODEL_CONTRACT}
    DOC.write_text(
        "# Benefit-Selective DRCR External Execution Pack\n\n"
        "This pack operationalizes the outcome-blind external manifest. It was frozen before any external forecast outcome was inspected.\n\n"
        "## Inventory\n\n"
        f"- Jobs: {len(jobs)} ({family_counts}).\n"
        f"- Distinct datasets: {len({row['dataset'] for row in jobs})}.\n"
        f"- Requested source-specific windows: {sum(int(row['windows_requested']) for row in jobs)}.\n"
        "- Execution: one model process at a time; no parallel model loading.\n"
        "- Quantiles: q10 through q90 for every family.\n"
        "- Context cap: 512 for every model; classical context cap: 2048.\n"
        "- Classical expert: three-fold pre-origin rolling validation among naive, seasonal naive, AutoETS, and AutoARIMA.\n\n"
        "## Run\n\n"
        "Dry-run and inspect resource gating:\n\n"
        "```bash\npython3 scripts/run_benefit_selective_external_queue.py --dry-run\n```\n\n"
        "Run only Chronos, downloading missing datasets, strictly serially:\n\n"
        "```bash\npython3 scripts/run_benefit_selective_external_queue.py --family chronos --download-missing\n```\n\n"
        "Run one job by ID:\n\n"
        "```bash\npython3 scripts/run_benefit_selective_external_queue.py --job-id external_v1_chronos_m4_yearly_a_short --download-missing\n```\n\n"
        "The queue refuses to start model jobs when swap use exceeds 85%, disk free space is below 10 GiB, or family-specific available-RAM floors fail. Overrides exist only for a known larger machine and are recorded in the execution status.\n\n"
        "## Artifacts\n\n"
        "- Job table: `results/aaai_stress/benefit_selective_external_execution_jobs.csv`.\n"
        "- Runtime status: `results/aaai_stress/benefit_selective_external_execution_status.csv`.\n"
        "- Logs: `results/aaai_stress/external_logs/`.\n"
        "- Raw forecasts and sidecars: `results/raw_forecasts/external_v1_*`.\n"
    )
    print(f"wrote {len(jobs)} jobs to {OUT}")


if __name__ == "__main__":
    main()
