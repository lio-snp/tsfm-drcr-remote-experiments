# TSFM DRCR Remote q9 Rerun Handoff

This repository is a remote-execution handoff for the remaining memory-bound
q9/full-grid reruns in the DRCR/TSFM reliability study. The local 8GB machine is
the research lead and paper-integration environment; the remote/larger-memory
machine is responsible for executing the frozen rerun queue and returning
auditable artifacts.

## Current Scope

- P0 sources: `17`
- Forecast windows: `408`
- Families: `moirai`, `timesfm`
- Goal: upgrade the current Moirai/TimesFM proxy-level evidence to full q9 or
  full-grid forecast artifacts that can be ingested into the final-main paper
  tables.
- Current lead-side ingestion status: `0/17 ready`

This handoff does not change the scientific protocol. It only moves the
memory-heavy model inference to a larger machine.

## Hardware Target

- Minimum practical RAM: `16GB`
- Preferred RAM: `32GB+`
- Python: `3.10+`
- Disk: leave at least `20GB` free for model caches and raw forecast artifacts.
- GPU is useful but not required if runtime is acceptable; CPU-only runs should
  use long timeouts.

## Repository Layout Expected by the Runner

```text
.
├── configs/
├── docs/
├── results/
│   ├── aaai_stress/
│   │   ├── remote_q9_rerun_plan.csv
│   │   ├── oral_rerun_command_manifest.csv
│   │   └── rerun_manifests/
│   └── raw_forecasts/
├── scripts/
├── src/
└── tests/
```

The runner expects GIFT-Eval style data under `data/gift-eval/...`. Data is not
included in the public handoff by default. Put data in the same relative paths
used by the command manifest, or symlink those paths before running.

## Setup on the Remote Machine

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,tsfm]"
python -m pip install pandas numpy scipy scikit-learn matplotlib pyyaml
```

Family-specific dependencies may still be required by the model adapters. If a
runner fails because a model package is missing, install the package named in
the traceback and rerun the same source. Do not edit manifests to skip failures.

## Dry Run

```bash
python3 scripts/run_oral_rerun_queue.py --priority P0 --dry-run
```

The dry run should list `17` P0 commands and should not execute model inference.

## Execute the Queue

Run families separately so failures are isolated.

```bash
python3 scripts/run_oral_rerun_queue.py --priority P0 --family moirai --timeout-seconds 7200
python3 scripts/run_oral_rerun_queue.py --priority P0 --family timesfm --timeout-seconds 7200
```

If the remote machine has enough RAM but the preflight check is conservative,
rerun the exact same command with:

```bash
--allow-low-memory
```

Do not use that flag on a genuinely small machine; it only bypasses the check,
it does not reduce memory use.

## Validate Before Returning Results

```bash
python3 scripts/build_remote_q9_rerun_completion_audit.py
bash scripts/critic_remote_q9_rerun_completion_audit.sh
python3 scripts/build_remote_q9_ingestion_manifest.py
bash scripts/critic_remote_q9_ingestion_manifest.sh
python3 scripts/build_oral_evidence_gap_matrix.py
python3 scripts/build_oral_rerun_command_manifest.py
python3 scripts/build_aaai_oral_goal_status.py
python3 scripts/build_drcr_paper_readiness_reports.py
bash scripts/critic_oral_evidence_gap_matrix.sh
bash scripts/critic_oral_rerun_command_manifest.sh
bash scripts/critic_drcr_paper_readiness_reports.sh
```

Expected successful outcome:

- `results/aaai_stress/remote_q9_rerun_completion_audit.csv` marks all P0
  sources as complete for ingestion.
- `results/aaai_stress/remote_q9_ingestion_manifest.csv` marks all P0 sources
  ready.
- Each expected raw forecast file exists under `results/raw_forecasts/`.
- Each expected history/context sidecar exists under `results/raw_forecasts/`.
- Runner status JSON files report success.

## Return Protocol

After running, commit or archive only these classes of artifacts:

- `results/raw_forecasts/*_oral_sidecar_rerun.csv`
- `results/raw_forecasts/*_oral_sidecar_rerun_history_context.csv`
- runner status JSON files created by the queue
- `results/aaai_stress/remote_q9_rerun_completion_audit.csv`
- `results/aaai_stress/remote_q9_ingestion_manifest.csv`
- updated `docs/remote_q9_rerun_completion_audit.md`
- updated `docs/remote_q9_ingestion_manifest.md`
- one filled run log based on `remote_q9_handoff/RUN_LOG_TEMPLATE.md`

Do not overwrite locked final-main paper tables manually. The lead machine will
ingest returned artifacts, refresh the source inventory, and rebuild the paper
tables.

## What Not To Change

- Do not edit `results/aaai_stress/rerun_manifests/*.csv`.
- Do not change `output_slug` values.
- Do not remove timeout/failure rows from manifests.
- Do not tune DRCR candidate policies on the remote results.
- Do not report q3 interval proxy outputs as q9/full-grid outputs.

## Scientific Boundary

This remote run is an evidence-breadth upgrade, not a new method search. Its job
is to answer whether the current DRCR claims remain stable when Moirai and
TimesFM sources are upgraded from proxy-level evidence to full q9/full-grid
artifacts.
