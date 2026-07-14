# Benefit-Selective External Confirmation Partial Run Log

Run date: 2026-07-12 to 2026-07-14 Asia/Shanghai

Repository: `E:\DRCR-helper\tsfm-drcr-remote-experiments-win-exec`

Purpose: execute the frozen `benefit_selective_external_v1` external confirmation queue after the earlier 17-source q9 queue was completed elsewhere. This package intentionally returns partial external artifacts from the current 24 GB Windows CPU machine. It does not modify frozen protocol files, manifests, model parameters, DRCR method parameters, metrics, or output slugs.

## Machine

- OS: Windows 11 10.0.26200 (`platform.platform()`)
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- Logical CPUs: 20
- RAM: 23.73 GiB total; 11.31 GiB available at final audit
- GPU: not used
- Disk: E: 183.92 GiB free at final audit
- Python: 3.12.3

System commands `systeminfo`, `wmic`, WSL bash, and Git Bash critic launch were blocked by local Windows access-control errors. Hardware fields above were collected through Python/psutil.

## Environment

Two independent local virtual environments were used:

- `.venv-chronos`: Chronos and TimesFM runs
- `.venv-moirai`: Moirai runs

Relevant packages:

- `torch==2.4.1`
- `chronos-forecasting==2.3.1`
- `timesfm==2.0.0`
- `uni2ts==2.0.0`
- `pandas==2.1.4`
- `numpy==1.26.4`
- `scipy==1.11.4`
- `scikit-learn==1.9.0`

Cache and offline settings used for model runs:

- `HF_HOME=E:\DRCR-helper\tsfm-drcr-cache\huggingface`
- `HF_HUB_CACHE=E:\DRCR-helper\tsfm-drcr-cache\huggingface\hub`
- `TRANSFORMERS_CACHE=E:\DRCR-helper\tsfm-drcr-cache\huggingface\transformers`
- `TORCH_HOME=E:\DRCR-helper\tsfm-drcr-cache\torch`
- `PIP_CACHE_DIR=E:\DRCR-helper\tsfm-drcr-cache\pip`
- `TEMP=E:\DRCR-helper\tsfm-drcr-cache\temp`
- `TMP=E:\DRCR-helper\tsfm-drcr-cache\temp`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1`

## Data

The 14 frozen GIFT-Eval external datasets were downloaded/prepared under `data/gift-eval`. Windows case-insensitive lookup was sufficient for the `SZ_TAXI`/`sz_taxi` naming mismatch; no manifest or data-path slug was edited. Some datasets use subdirectory frequency layouts and some use a root layout; the queue resolver handled both.

## Preflight and Protocol Guards

Before model execution:

- External queue dry run: 42/42 jobs were `dry_run_ready`.
- Fail-closed evaluator preflight: 0/42 ready, 0/672 windows, `outcomes_inspected=false`.
- 28 frozen SHA256 files matched the freeze contract.

After partial execution:

- Runner-level status: 19 `ok`, 23 `failed_artifact_contract`, 0 `not_run`.
- Fail-closed evaluator preflight: `pending_external_artifacts`, 10/42 jobs ready, 160/672 windows ready, `outcomes_inspected=false`.
- The final evaluator was not run because the fail-closed preflight correctly reports incomplete external artifacts.

## Completed Runner Artifacts

Runner-level complete jobs:

- Chronos: `external_v1_chronos_bizitobs_l2c_5t_medium`, `external_v1_chronos_ett1_w_short`, `external_v1_chronos_hospital_m_short`, `external_v1_chronos_m4_monthly_m_short`, `external_v1_chronos_m4_yearly_a_short`, `external_v1_chronos_restaurant_d_short`, `external_v1_chronos_temperature_rain_d_short`, `external_v1_chronos_us_births_m_short`
- Moirai: `external_v1_moirai_bizitobs_l2c_5t_medium`, `external_v1_moirai_ett1_w_short`, `external_v1_moirai_hospital_m_short`, `external_v1_moirai_m4_monthly_m_short`, `external_v1_moirai_m4_yearly_a_short`, `external_v1_moirai_restaurant_d_short`, `external_v1_moirai_temperature_rain_d_short`, `external_v1_moirai_us_births_m_short`
- TimesFM: `external_v1_timesfm_bizitobs_l2c_5t_medium`, `external_v1_timesfm_m4_monthly_m_short`, `external_v1_timesfm_temperature_rain_d_short`

Each complete job has:

- raw forecast CSV in `results/raw_forecasts/external_v1_*.csv`
- history sidecar CSV in `results/raw_forecasts/external_v1_*_history_context.csv`
- status JSON in `results/raw_forecasts/external_v1_*_status.json`
- window metrics in `results/window_metrics/external_v1_*_metrics.csv`
- failure-mining summary/window CSVs in `results/failure_mining/external_v1_*`

Largest raw/history files are about 1.75 MB, so no external release artifact or compression was needed.

## Failed Jobs

The following jobs have preserved status/log evidence:

- Timeout at 7200 seconds: `external_v1_chronos_bitbrains_fast_storage_5t_short`, `external_v1_chronos_m_dense_h_short`, `external_v1_chronos_sz_taxi_15t_short`, `external_v1_moirai_bitbrains_fast_storage_5t_short`, `external_v1_moirai_electricity_15t_short`, `external_v1_moirai_kdd_cup_2018_h_medium`, `external_v1_moirai_m_dense_h_short`, `external_v1_moirai_sz_taxi_15t_short`, `external_v1_timesfm_bitbrains_fast_storage_5t_short`, `external_v1_timesfm_electricity_15t_short`, `external_v1_timesfm_kdd_cup_2018_h_medium`, `external_v1_timesfm_m_dense_h_short`, `external_v1_timesfm_sz_taxi_15t_short`
- Timeout at 14400 seconds after retry: `external_v1_chronos_electricity_15t_short`, `external_v1_chronos_kdd_cup_2018_h_medium`
- Baseline feasibility failure: `external_v1_chronos_car_parts_m_short`, `external_v1_moirai_car_parts_m_short`, `external_v1_timesfm_car_parts_m_short`
- TimesFM non-finite/no-row failure: `external_v1_timesfm_ett1_w_short`, `external_v1_timesfm_hospital_m_short`, `external_v1_timesfm_m4_yearly_a_short`, `external_v1_timesfm_restaurant_d_short`, `external_v1_timesfm_us_births_m_short`

Representative traceback/status summaries:

- `car_parts`: `ValueError: Rolling-selected baseline produced no finite comparison points`.
- TimesFM short-context failures: runner wrote `blocked_no_rows`; diagnostic probes showed finite input context/target but all-non-finite TimesFM point/quantile output on affected short-context examples.
- Timeout failures: no model traceback; watchdog killed the job after 7200 seconds or, for two Chronos retry probes, 14400 seconds.

## Retries

Before 14400-second retry, the status CSV and timeout logs were snapshotted under:

- `results/aaai_stress/external_logs/retry_timeout14400_snapshot_20260714_110234/`

Attempt-2 logs preserved:

- `results/aaai_stress/external_logs/external_v1_chronos_electricity_15t_short_attempt2_timeout14400.log`
- `results/aaai_stress/external_logs/external_v1_chronos_kdd_cup_2018_h_medium_attempt2_timeout14400.log`

Both 14400-second retry probes still timed out. Remaining timeout jobs were not retried further on this machine because the representative 4-hour probes indicate the current CPU-only machine is unlikely to finish the slow sources efficiently.

## Validation Output Summary

Direct Bash execution failed due local Windows access-control errors:

- WSL bash: `Bash/Service/CreateInstance/E_ACCESSDENIED`
- Git Bash: `couldn't create signal pipe, Win32 error 5` / `CreateFileMapping ..., Win32 error 5`

The embedded Python logic from each critic script was executed directly with `.venv-chronos\Scripts\python.exe`:

- `[external-protocol critic] PASS: frozen whole-domain protocol is outcome-blind and fail-closed`
- `[external-execution-pack critic] PASS: 42 locked serial jobs use q9 and non-leaky pre-origin baseline selection`
- `[external-evaluator critic] PASS: fail-closed evaluator exports standard metrics, controls, family sensitivity, and domain inference`
- `[external-freeze-hash] PASS`: 28/28 frozen files checked, 0 missing, 0 mismatched

Fail-closed evaluator preflight after partial execution:

- `status=pending_external_artifacts`
- `jobs_ready=10`
- `jobs_expected=42`
- `windows_ready=160`
- `windows_expected=672`
- `outcomes_inspected=false`

## Interpretation for Lead Machine

This package should be ingested as a partial external confirmation execution package, not as a completed external confirmation. The lead machine can use the 19 complete raw/history/status artifacts and the 23 failure logs/status records for audit. It should not upgrade any Benefit-Selective external-confirmation claim from this package alone because the frozen fail-closed evaluator still reports incomplete artifacts.

Recommended next step on a stronger machine: rerun only the preserved timeout jobs with a longer wall-clock budget or faster CPU/GPU where supported, while keeping all frozen job commands, output slugs, protocol files, and metrics unchanged. Baseline feasibility and TimesFM non-finite/no-row failures should remain as recorded unless the team explicitly approves a protocol-compatible runner fix.
