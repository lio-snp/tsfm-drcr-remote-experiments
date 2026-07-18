# Prompt for the Remote Codex: E-GAP-000R

Paste the following prompt into Codex on the larger machine after cloning this
repository.

```text
You are the remote execution owner for E-GAP-000R in the TSFM reliability
project. Work autonomously until every recoverable frozen job has been retried,
the fail-closed validation has been run, and all auditable artifacts have been
committed and pushed. Do not tune the research method.

Repository:
https://github.com/lio-snp/tsfm-drcr-remote-experiments

Scientific objective:
Recover the 33 outcome-invalid or incomplete jobs in the frozen 42-job external
Benefit-Selective DRCR endpoint. Preserve the original 14 datasets, three TSFM
families, 16 selected series per job, one origin per series, q10-q90 grid,
classical reference, model IDs, output slugs, candidate policy, metrics, and all
28 frozen hashes. This is execution recovery, not method search.

Current authoritative state:
- 42 jobs / 672 requested windows.
- 19 runner-complete.
- 10 exact-contract.
- 9 outcome-valid.
- 8 balanced analysis jobs / 128 windows / four datasets / Chronos and Moirai.
- 33 recovery jobs: 15 timeout, 3 baseline infeasible, 5 TimesFM non-finite,
  9 runner-complete contract mismatch, and 1 outcome-invalid.
- Recovery inventory:
  results/aaai_stress/benefit_selective_external_recovery_jobs.csv
- Lead reconciliation:
  results/aaai_stress/benefit_selective_external_endpoint_reconciliation.csv

Rules:
1. Do not edit any hash-locked file or any scientific selection.
2. Do not change manifests, window identities, model IDs, context caps,
   quantiles, baseline definition, output slugs, actions, policy, or metrics.
3. Environment/package/cache changes are allowed. Source-code changes to
   hash-locked runners or evaluators are not allowed.
4. Rerun every recovery job with --force. Existing status=ok is not sufficient:
   nine such jobs violate the exact artifact contract and one has an invalid
   outcome.
5. Continue after individual failures. Preserve logs and status JSON for every
   attempt. Do not delete failed rows.
6. Do not inspect results to tune the method. Run the locked final evaluator
   only if its preflight becomes ready.

Execution procedure:
1. Inspect README.md, docs/benefit_selective_external_confirmation_protocol.md,
   and docs/benefit_selective_external_execution_pack.md.
2. Detect OS, RAM, disk, Python, available GPU, and existing virtual
   environments. Reuse model caches where possible.
3. Install dependencies without modifying frozen source files. On Windows,
   preserve the separate .venv-chronos and .venv-moirai environments when they
   already exist.
4. Run:
   python scripts/build_benefit_selective_external_recovery_manifest.py
   python tests/test_benefit_selective_external_recovery.py
5. Dry-run and verify exactly 33 jobs are listed:
   python scripts/run_benefit_selective_external_recovery.py --dry-run
6. Execute serially, preferably family by family, with an eight-hour per-job
   ceiling unless the machine has a justified stricter scheduler limit:
   python scripts/run_benefit_selective_external_recovery.py --family chronos --timeout-seconds 28800
   python scripts/run_benefit_selective_external_recovery.py --family moirai --timeout-seconds 28800
   python scripts/run_benefit_selective_external_recovery.py --family timesfm --timeout-seconds 28800
   The wrapper continues after failed jobs. If a resource preflight is overly
   conservative on a genuinely larger machine, use --ignore-swap-preflight and
   record why in the run log.
7. Rebuild the frozen preflight:
   PYTHONPATH=src python scripts/evaluate_benefit_selective_external.py --preflight-only
8. Run all critics:
   bash scripts/critic_benefit_selective_external_protocol.sh
   bash scripts/critic_benefit_selective_external_execution_pack.sh
   bash scripts/critic_benefit_selective_external_evaluator.sh
   If Bash is unavailable on Windows, execute the embedded Python checks and
   record that substitution exactly.
9. Only if preflight reports the complete locked endpoint ready, run once:
   PYTHONPATH=src python scripts/evaluate_benefit_selective_external.py
10. Create a dated run log based on remote_q9_handoff/RUN_LOG_TEMPLATE.md. It
    must list hardware, environment, attempted jobs, elapsed times, failures,
    exact preflight counts, hash result, and whether outcomes were inspected.
11. Run the repository tests relevant to changed or generated artifacts.
12. Commit and push only generated forecasts, histories, metrics, failure
    summaries, status JSON/CSV, logs, confirmation artifacts, and the dated run
    log. Do not commit caches, datasets, model weights, virtual environments, or
    modified frozen files.

Success criteria:
- Preferred: 42/42 exact jobs, 672/672 windows, 28/28 hashes, final evaluator
  completes once.
- Honest fallback: every one of the 33 jobs was attempted under the unchanged
  contract; all remaining technical failures are fully logged; preflight stays
  fail-closed; no scientific claim is upgraded.

Final response must report:
- commit hash and pushed branch;
- attempted/succeeded/failed counts by family and failure category;
- exact preflight jobs/windows ready;
- 28-file hash result;
- whether the final evaluator ran;
- paths to the run log, recovery status CSV, and confirmation artifacts;
- any remaining blocker without proposing method retuning.
```
