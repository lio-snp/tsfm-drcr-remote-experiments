# Remote q9 Results Contract

This contract defines what a remote machine must return after running the P0
q9/full-grid queue. The lead paper environment will reject incomplete or
ambiguous outputs.

## Required Artifact Classes

For every row in `results/aaai_stress/remote_q9_rerun_plan.csv`, return:

1. Raw forecast CSV
   - Path must match `expected_raw_path`.
   - Must be non-empty.
   - Must contain one row per requested manifest window.
   - Must contain quantile forecast columns appropriate for q9/full-grid
     ingestion, e.g. `forecast_q10`, `forecast_q20`, ..., `forecast_q90`.

2. History/context sidecar CSV
   - Path must match `expected_sidecar_path`.
   - Must be non-empty.
   - Must align one-to-one with the window manifest rows.
   - Must preserve enough history/context information for downstream baseline
     reconstruction and audit.

3. Runner status JSON
   - Must report the source slug, command, start/end timestamps, exit status,
     and error message if any.
   - Successful rows should have an explicit success/ok flag.

4. Audit tables
   - `results/aaai_stress/remote_q9_rerun_completion_audit.csv`
   - `results/aaai_stress/remote_q9_ingestion_manifest.csv`

5. Human run log
   - Fill `remote_q9_handoff/RUN_LOG_TEMPLATE.md`.
   - Include hardware, environment, exact command sequence, failures, retries,
     and known deviations.

## Accepted Completion States

Use these exact meanings in the run log:

- `complete`: raw, sidecar, and status outputs exist and pass critic scripts.
- `partial`: some outputs exist, but at least one required artifact is missing
  or failed validation.
- `failed`: command executed and failed without usable artifacts.
- `not_run`: command was not attempted.

Do not mark a row `complete` only because the command exited with code 0. The
audit scripts are the authority.

## Preferred Return Mechanisms

Small enough for Git:

```bash
git add results/raw_forecasts results/aaai_stress docs remote_q9_handoff
git commit -m "Return remote q9 rerun artifacts"
git push
```

Too large for Git:

- Put raw forecast files into a GitHub Release artifact or external archive.
- Commit the audit CSVs, run log, and a manifest with SHA256 checksums.
- Preserve the relative paths so the lead machine can restore them into
  `results/raw_forecasts/`.

## Minimal Review Checklist

Before handoff back to the lead machine:

- [ ] `python3 scripts/run_oral_rerun_queue.py --priority P0 --dry-run` still
      lists the same P0 sources after execution.
- [ ] Completion audit reports all expected raw files.
- [ ] Completion audit reports all expected sidecar files.
- [ ] Ingestion manifest marks completed sources as ready.
- [ ] Critic scripts pass.
- [ ] No locked window manifest was edited.
- [ ] No DRCR thresholds or candidate policies were changed.
- [ ] The run log documents any failures or retries.
