# Remote q9/Full-Grid Rerun Execution Pack

This pack is the execution contract for the remaining P0 probabilistic-evidence gaps. It is meant for a remote or larger-memory machine; local 8GB attempts already timed out and should not be treated as failed science.

## Scope

- P0 sources to rerun: `17`.
- Total forecast windows: `408`.
- Families: `moirai, timesfm`.
- Current local timeout blockers among these sources: `2`.
- Required evidence per source: raw forecast CSV, history/context sidecar CSV, and runner status JSON.
- Success criterion: every P0 row in `results/aaai_stress/remote_q9_rerun_plan.csv` has non-empty expected raw and sidecar outputs, then the gap matrix and paper-readiness reports are rebuilt and critics pass.

## Hardware / Runtime Recommendation

- Minimum practical target: 16GB RAM.
- Preferred target: 32GB RAM or more, especially for TimesFM 2.5 and Moirai 1.1 large/base runs.
- Keep the same repository checkout and data paths; commands use source-specific window manifests to avoid mixing windows.
- Use family-specific environments if present: `.venv-moirai` for Moirai and `.venv-chronos` for TimesFM/Chronos. The queue runner handles this automatically.

## Family Summary

| Family | Sources | Windows | Failure targets | Positive controls | Stress targets |
| --- | --- | --- | --- | --- | --- |
| moirai | 7 | 56 | 3 | 4 | 0 |
| timesfm | 10 | 352 | 4 | 3 | 3 |

## Role Summary

| Role | Sources | Windows |
| --- | --- | --- |
| failure_target | 7 | 240 |
| positive_control | 7 | 80 |
| stress_target | 3 | 88 |

## Recommended Execution

Run Moirai and TimesFM separately so failures are isolated and resumable.

### Dry Run

```bash
python3 scripts/run_oral_rerun_queue.py --priority P0 --dry-run
```

### Moirai

```bash
python3 scripts/run_oral_rerun_queue.py --priority P0 --family moirai --timeout-seconds 7200
```

### TimesFM

```bash
python3 scripts/run_oral_rerun_queue.py --priority P0 --family timesfm --timeout-seconds 7200
```

If the remote environment has enough RAM, do not use `--allow-low-memory`; that flag only bypasses local preflight checks and does not make the inference cheaper. If a preflight check is too conservative on a known large machine, rerun the same command with `--allow-low-memory` and a longer timeout.

## Post-Run Validation

After the queue finishes, first audit whether the expected raw/sidecar/status artifacts actually arrived, then rebuild the current dashboard reports:

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

Expected validation outcome after a complete successful remote run:

- `results/aaai_stress/remote_q9_rerun_completion_audit.csv` reports all 17 P0 sources as `complete_for_ingestion=1`.
- Raw files have non-empty `forecast_q10..forecast_q90` columns; sidecar files have one row per manifest window; runner status JSON files report `ok`.
- The current gap matrix may still show q9/full-grid gaps until the final-main source inventory is refreshed to point at the `_oral_sidecar_rerun` slugs. New raw files alone do not update the locked final-main manifests.
- After the ingestion/refresh step, `docs/aaai_oral_goal_status.md` and `docs/final_drcr_paper_readiness_summary.md` should stop describing these Moirai/TimesFM sources as only proxy-level evidence.
- The critic scripts above pass.

## P0 Source Plan

| # | Family | Dataset | Role | Source | Windows | Current status | Expected raw | Expected sidecar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | moirai | LOOP_SEATTLE/H/short | positive_control | moirai2_loop_m8_loop_seattle_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 2 | moirai | covid_deaths/D/short | failure_target | moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets | 8 | not_attempted | results/raw_forecasts/moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 3 | moirai | covid_deaths/D/short | failure_target | moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets | 8 | not_attempted | results/raw_forecasts/moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 4 | moirai | covid_deaths/D/short | failure_target | moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets | 8 | timeout | results/raw_forecasts/moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 5 | moirai | loop_seattle/H/short | positive_control | moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 6 | moirai | loop_seattle/H/short | positive_control | moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 7 | moirai | loop_seattle/H/short | positive_control | moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 8 | timesfm | FRED finance stress seed | stress_target | timesfm_2_5_finance_fred_finance_fred_stress | 56 | not_attempted | results/raw_forecasts/timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun_history_context.csv |
| 9 | timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive | 32 | not_attempted | results/raw_forecasts/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 10 | timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |
| 11 | timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima | 30 | not_attempted | results/raw_forecasts/timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun_history_context.csv |
| 12 | timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima | 2 | timeout | results/raw_forecasts/timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun_history_context.csv |
| 13 | timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m128_covid_deaths_short_auto_ets | 128 | not_attempted | results/raw_forecasts/timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 14 | timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m16_covid_deaths_short_auto_ets | 16 | not_attempted | results/raw_forecasts/timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 15 | timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m64_covid_deaths_short_auto_ets | 64 | not_attempted | results/raw_forecasts/timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 16 | timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m8_covid_deaths_short_auto_ets | 8 | not_attempted | results/raw_forecasts/timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv |
| 17 | timesfm | solar/10T/short | positive_control | timesfm_2_5_solar_m8_solar_short_seasonal_naive | 8 | not_attempted | results/raw_forecasts/timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun.csv | results/raw_forecasts/timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun_history_context.csv |

## Exact Commands

### 1. moirai2_loop_m8_loop_seattle_short_seasonal_naive

- Family: `moirai`; dataset: `LOOP_SEATTLE/H/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/LOOP_SEATTLE/H --term short --model-id Salesforce/moirai-2.0-R-small --model-name moirai2_loop_m8 --model-family moirai2 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/moirai2_loop_m8_loop_seattle_short_seasonal_naive_windows.csv --output-slug moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 1680 --baseline-context-cap 1680 --baseline-season-length 48
```

### 2. moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets

- Family: `moirai`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id Salesforce/moirai-1.1-R-base --model-name moirai_1_1_base --model-family moirai1 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_windows.csv --output-slug moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 182 --num-samples 50 --baseline-context-cap 182 --baseline-season-length 1
```

### 3. moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets

- Family: `moirai`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id Salesforce/moirai-1.1-R-large --model-name moirai_1_1_large --model-family moirai1 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_windows.csv --output-slug moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 182 --num-samples 50 --baseline-context-cap 182 --baseline-season-length 1
```

### 4. moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets

- Family: `moirai`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id Salesforce/moirai-1.1-R-small --model-name moirai_1_1_small --model-family moirai1 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_windows.csv --output-slug moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 182 --num-samples 50 --baseline-context-cap 182 --baseline-season-length 1
```

### 5. moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive

- Family: `moirai`; dataset: `loop_seattle/H/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/loop_seattle/H --term short --model-id Salesforce/moirai-1.1-R-base --model-name moirai_1_1_base --model-family moirai1 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_windows.csv --output-slug moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 1680 --num-samples 50 --baseline-context-cap 1680 --baseline-season-length 48
```

### 6. moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive

- Family: `moirai`; dataset: `loop_seattle/H/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/loop_seattle/H --term short --model-id Salesforce/moirai-1.1-R-large --model-name moirai_1_1_large --model-family moirai1 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_windows.csv --output-slug moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 1680 --num-samples 50 --baseline-context-cap 1680 --baseline-season-length 48
```

### 7. moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive

- Family: `moirai`; dataset: `loop_seattle/H/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_moirai_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/loop_seattle/H --term short --model-id Salesforce/moirai-1.1-R-small --model-name moirai_1_1_small --model-family moirai1 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_windows.csv --output-slug moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 1680 --num-samples 50 --baseline-context-cap 1680 --baseline-season-length 48
```

### 8. timesfm_2_5_finance_fred_finance_fred_stress

- Family: `timesfm`; dataset: `FRED finance stress seed`; role: `stress_target`; windows: `56`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_finance_fred_stress.py --backend timesfm --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_finance_fred --start-date 2016-01-01 --end-date 2026-07-04 --context-length 256 --horizon 10 --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_finance_fred_finance_fred_stress_windows.csv --output-slug timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun --export-history-sidecar
```

### 9. timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive

- Family: `timesfm`; dataset: `LOOP_SEATTLE/H/short`; role: `positive_control`; windows: `32`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/LOOP_SEATTLE/H --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_loop_m32 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_windows.csv --output-slug timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 10. timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive

- Family: `timesfm`; dataset: `LOOP_SEATTLE/H/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name loop_seattle --data-path data/gift-eval/LOOP_SEATTLE/H --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_loop_m8 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_windows.csv --output-slug timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 11. timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima

- Family: `timesfm`; dataset: `bizitobs_application/10S/short`; role: `stress_target`; windows: `30`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name bizitobs_application --data-path data/gift-eval/bizitobs_application --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_bizitobs_m30 --baseline-mode auto_arima --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_windows.csv --output-slug timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 12. timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima

- Family: `timesfm`; dataset: `bizitobs_application/10S/short`; role: `stress_target`; windows: `2`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name bizitobs_application --data-path data/gift-eval/bizitobs_application --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_bizitobs_m8 --baseline-mode auto_arima --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_windows.csv --output-slug timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 13. timesfm_2_5_m128_covid_deaths_short_auto_ets

- Family: `timesfm`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `128`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_m128 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_m128_covid_deaths_short_auto_ets_windows.csv --output-slug timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 14. timesfm_2_5_m16_covid_deaths_short_auto_ets

- Family: `timesfm`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `16`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_m16 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_m16_covid_deaths_short_auto_ets_windows.csv --output-slug timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 15. timesfm_2_5_m64_covid_deaths_short_auto_ets

- Family: `timesfm`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `64`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_m64 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_m64_covid_deaths_short_auto_ets_windows.csv --output-slug timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 16. timesfm_2_5_m8_covid_deaths_short_auto_ets

- Family: `timesfm`; dataset: `covid_deaths/D/short`; role: `failure_target`; windows: `8`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name covid_deaths --data-path data/gift-eval/covid_deaths --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_m8 --baseline-mode auto_ets --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_m8_covid_deaths_short_auto_ets_windows.csv --output-slug timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

### 17. timesfm_2_5_solar_m8_solar_short_seasonal_naive

- Family: `timesfm`; dataset: `solar/10T/short`; role: `positive_control`; windows: `8`.
- Expected raw: `results/raw_forecasts/timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun.csv`.
- Expected sidecar: `results/raw_forecasts/timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun_history_context.csv`.

```bash
python3 scripts/run_timesfm_gift_eval_raw.py --dataset-name solar --data-path data/gift-eval/solar/10T --term short --model-id google/timesfm-2.5-200m-pytorch --model-name timesfm_2_5_solar_m8 --baseline-mode seasonal_naive --window-manifest results/aaai_stress/rerun_manifests/timesfm_2_5_solar_m8_solar_short_seasonal_naive_windows.csv --output-slug timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun --export-history-sidecar --context-cap 128
```

## Artifacts

- `results/aaai_stress/remote_q9_rerun_plan.csv`
- `docs/remote_q9_rerun_execution_pack.md`
- Source command manifest: `results/aaai_stress/oral_rerun_command_manifest.csv`
- Gap matrix: `results/aaai_stress/oral_evidence_source_gap_matrix.csv`
