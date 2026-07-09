# Remote q9/Full-Grid Rerun Completion Audit

This audit answers a narrower question than the main paper dashboard: have the remote q9/full-grid rerun artifacts actually arrived and are they complete enough to be ingested into the final-main suite?

## Summary

- Planned P0 sources: `17`.
- Complete for ingestion: `0`.
- Sources with raw CSV present: `0`.
- Sources with sidecar CSV present: `0`.
- Sources with all q10..q90 columns non-empty: `0`.
- Interpretation: a source is complete only when raw forecasts, exact history/context sidecar, runner status, and nine quantile columns are all present. This audit does not run model inference.

## Incomplete / Pending Sources

| Family | Dataset | Role | Source | Windows | Raw | Raw windows | Sidecar | Sidecar rows | Status | Missing q cols |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moirai | LOOP_SEATTLE/H/short | positive_control | moirai2_loop_m8_loop_seattle_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets | 8 | 0 | 0 | 0 | 0 | blocked_insufficient_memory | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | FRED finance stress seed | stress_target | timesfm_2_5_finance_fred_finance_fred_stress | 56 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive | 32 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima | 30 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima | 2 | 0 | 0 | 0 | 0 | blocked_insufficient_memory | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m128_covid_deaths_short_auto_ets | 128 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m16_covid_deaths_short_auto_ets | 16 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m64_covid_deaths_short_auto_ets | 64 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m8_covid_deaths_short_auto_ets | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |
| timesfm | solar/10T/short | positive_control | timesfm_2_5_solar_m8_solar_short_seasonal_naive | 8 | 0 | 0 | 0 | 0 | missing | forecast_q10;forecast_q20;forecast_q30;forecast_q40;forecast_q50;forecast_q60;forecast_q70;forecast_q80;forecast_q90 |

## Post-Completion Ingestion Steps

After all 17 sources are complete, backfill predictor features for the rerun slugs, then refresh the final-main source inventory so the `_oral_sidecar_rerun` slugs replace their q3/proxy predecessors as q9/full-grid evidence.

```bash
# If no slugs are complete yet, rerun this audit after remote outputs arrive.
python3 scripts/build_missing_predictor_features_from_metrics.py <complete_rerun_slug_1> <complete_rerun_slug_2> ...
```

Important boundary: `build_oral_evidence_gap_matrix.py` is still based on the current final-main manifests. It will not clear these gaps merely because new raw files exist. The final-main source inventory must be refreshed or explicitly pointed at the rerun slugs before the main dashboard can claim the q9/full-grid gap is closed.

## Artifacts

- `results/aaai_stress/remote_q9_rerun_completion_audit.csv`
- `docs/remote_q9_rerun_completion_audit.md`
- Source plan: `results/aaai_stress/remote_q9_rerun_plan.csv`
