# Benefit-Selective DRCR External Execution Pack

This pack operationalizes the outcome-blind external manifest. It was frozen before any external forecast outcome was inspected.

## Inventory

- Jobs: 42 ({'chronos': 14, 'moirai': 14, 'timesfm': 14}).
- Distinct datasets: 14.
- Requested source-specific windows: 672.
- Execution: one model process at a time; no parallel model loading.
- Quantiles: q10 through q90 for every family.
- Context cap: 512 for every model; classical context cap: 2048.
- Classical expert: three-fold pre-origin rolling validation among naive, seasonal naive, AutoETS, and AutoARIMA.

## Run

Dry-run and inspect resource gating:

```bash
python3 scripts/run_benefit_selective_external_queue.py --dry-run
```

Run only Chronos, downloading missing datasets, strictly serially:

```bash
python3 scripts/run_benefit_selective_external_queue.py --family chronos --download-missing
```

Run one job by ID:

```bash
python3 scripts/run_benefit_selective_external_queue.py --job-id external_v1_chronos_m4_yearly_a_short --download-missing
```

The queue refuses to start model jobs when swap use exceeds 85%, disk free space is below 10 GiB, or family-specific available-RAM floors fail. Overrides exist only for a known larger machine and are recorded in the execution status.

Before the first external outcome is inspected, freeze the protocol and evaluator hashes:

```bash
PYTHONPATH=src python3 scripts/freeze_benefit_selective_external_protocol.py
```

The evaluator remains fail-closed until all 42 exact jobs are ready:

```bash
PYTHONPATH=src python3 scripts/evaluate_benefit_selective_external.py --preflight-only
```

## Artifacts

- Job table: `results/aaai_stress/benefit_selective_external_execution_jobs.csv`.
- Runtime status: `results/aaai_stress/benefit_selective_external_execution_status.csv`.
- Logs: `results/aaai_stress/external_logs/`.
- Raw forecasts and sidecars: `results/raw_forecasts/external_v1_*`.
- Frozen file contract: `results/aaai_stress/benefit_selective_external_freeze_hashes.json`.
- Final standard metrics: q9-WQL, RelMAE, RelRMSE, MASE, q10-q90 coverage/gap, interval width, and intervention rate.
- Family sensitivity and forest-ready domain deltas are exported by the locked evaluator.
