#!/usr/bin/env python
"""Run the paired Moirai-1.1 capacity scaling experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MODELS = {
    "small": {"model_id": "Salesforce/moirai-1.1-R-small", "model_name": "moirai_1_1_small"},
    "base": {"model_id": "Salesforce/moirai-1.1-R-base", "model_name": "moirai_1_1_base"},
    "large": {"model_id": "Salesforce/moirai-1.1-R-large", "model_name": "moirai_1_1_large"},
}

TARGETS = {
    "covid_deaths": {
        "target_key": "covid_deaths_d_short",
        "dataset_name": "covid_deaths",
        "data_path": "data/gift-eval/covid_deaths",
        "term": "short",
        "baseline_mode": "auto_ets",
        "baseline_context_cap": "512",
        "baseline_season_length": "1",
        "context_cap": "1680",
        "domain": "Healthcare",
        "regime": "low_local_structure_count_decay",
    },
    "loop_seattle": {
        "target_key": "loop_seattle_h_short",
        "dataset_name": "loop_seattle",
        "data_path": "data/gift-eval/LOOP_SEATTLE/H",
        "term": "short",
        "baseline_mode": "seasonal_naive",
        "baseline_context_cap": "1680",
        "baseline_season_length": "48",
        "context_cap": "1680",
        "domain": "Transport",
        "regime": "medium_snr_persistent",
    },
}


def python_bin() -> str:
    candidate = ROOT / ".venv-moirai" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def run(command: list[str], *, env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def status_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="covid_deaths,loop_seattle")
    parser.add_argument("--models", default="small,base,large")
    parser.add_argument("--max-units-per-target", type=int, default=8)
    parser.add_argument("--manifest", default="results/scaling/moirai_1_1_scaling_manifest.csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--patch-size", default="16")
    parser.add_argument("--min-available-ram-gb", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected_targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    selected_models = [model.strip() for model in args.models.split(",") if model.strip()]
    for target in selected_targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target}; choose from {sorted(TARGETS)}")
    for model in selected_models:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model}; choose from {sorted(MODELS)}")

    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    py = python_bin()
    run(
        [
            py,
            "scripts/build_chronos_scaling_manifest.py",
            "--targets",
            ",".join(selected_targets),
            "--max-units-per-target",
            str(args.max_units_per_target),
            "--output",
            args.manifest,
            "--status",
            "results/scaling/moirai_1_1_scaling_manifest_status.json",
        ],
        env=env,
    )

    executed: list[dict[str, str]] = []
    for model_key in selected_models:
        model = MODELS[model_key]
        for target_key in selected_targets:
            target = TARGETS[target_key]
            output_slug = f"{model['model_name']}_scaling_{target['target_key']}_{target['baseline_mode']}"
            status_path = ROOT / "results" / "raw_forecasts" / f"{output_slug}_status.json"
            if args.skip_existing and status_ok(status_path):
                print(f"Skipping existing ok artifact: {status_path}", flush=True)
                executed.append({"model": model_key, "target": target_key, "status": "skipped_existing"})
                continue
            run(
                [
                    py,
                    "scripts/run_moirai_gift_eval_raw.py",
                    "--dataset-name",
                    str(target["dataset_name"]),
                    "--data-path",
                    str(target["data_path"]),
                    "--term",
                    str(target["term"]),
                    "--model-id",
                    str(model["model_id"]),
                    "--model-name",
                    str(model["model_name"]),
                    "--model-family",
                    "moirai1",
                    "--baseline-mode",
                    str(target["baseline_mode"]),
                    "--baseline-context-cap",
                    str(target["baseline_context_cap"]),
                    "--baseline-season-length",
                    str(target["baseline_season_length"]),
                    "--context-cap",
                    str(target["context_cap"]),
                    "--domain",
                    str(target["domain"]),
                    "--regime",
                    str(target["regime"]),
                    "--patch-size",
                    str(args.patch_size),
                    "--num-samples",
                    str(args.num_samples),
                    "--min-available-ram-gb",
                    str(args.min_available_ram_gb),
                    "--window-manifest",
                    args.manifest,
                    "--output-slug",
                    output_slug,
                ],
                env=env,
            )
            executed.append({"model": model_key, "target": target_key, "status": "run"})

    run(
        [
            py,
            "scripts/compile_moirai_1_1_scaling.py",
            "--manifest",
            args.manifest,
            "--bootstrap",
            str(args.bootstrap),
        ],
        env=env,
    )

    status_path = ROOT / "results" / "scaling" / "moirai_1_1_scaling_run_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "timestamp": int(time.time()),
                "targets": selected_targets,
                "models": selected_models,
                "max_units_per_target": args.max_units_per_target,
                "num_samples": args.num_samples,
                "patch_size": args.patch_size,
                "manifest": str(ROOT / args.manifest),
                "executed": executed,
            },
            indent=2,
        )
    )
    print(f"Wrote Moirai-1.1 scaling run status to {status_path}")


if __name__ == "__main__":
    main()
