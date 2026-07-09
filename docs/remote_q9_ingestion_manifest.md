# Remote q9 Ingestion Manifest

This manifest is the bridge between successful remote q9/full-grid reruns and a future final-main refresh. It does not claim new evidence is available; it records which locked q3/proxy sources should be replaced once their rerun raw, metrics, and predictor-feature artifacts are present.

## Summary

- Replacement sources: `17`.
- Raw/sidecar/status complete by completion audit: `0`.
- Metrics files present: `0`.
- Predictor features present: `0`.
- Ready for final-main refresh: `0`.
- Interpretation: a row is eligible for the next final-main rebuild only when `ready_for_final_main_refresh=1`.
- Boundary: do not count as q9/fullgrid evidence until ready_for_final_main_refresh=1.

## Replacement Map

| Family | Dataset | Role | Original source | Rerun slug | Old tier | New tier | Raw complete | Metrics | Features | Ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moirai | LOOP_SEATTLE/H/short | positive_control | moirai2_loop_m8_loop_seattle_short_seasonal_naive | moirai2_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets | moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets | moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | covid_deaths/D/short | failure_target | moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets | moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive | moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive | moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| moirai | loop_seattle/H/short | positive_control | moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive | moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | FRED finance stress seed | stress_target | timesfm_2_5_finance_fred_finance_fred_stress | timesfm_2_5_finance_fred_finance_fred_stress_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive | timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | LOOP_SEATTLE/H/short | positive_control | timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive | timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima | timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | bizitobs_application/10S/short | stress_target | timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima | timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m128_covid_deaths_short_auto_ets | timesfm_2_5_m128_covid_deaths_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m16_covid_deaths_short_auto_ets | timesfm_2_5_m16_covid_deaths_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m64_covid_deaths_short_auto_ets | timesfm_2_5_m64_covid_deaths_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | covid_deaths/D/short | failure_target | timesfm_2_5_m8_covid_deaths_short_auto_ets | timesfm_2_5_m8_covid_deaths_short_auto_ets_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |
| timesfm | solar/10T/short | positive_control | timesfm_2_5_solar_m8_solar_short_seasonal_naive | timesfm_2_5_solar_m8_solar_short_seasonal_naive_oral_sidecar_rerun | q3_interval_proxy | q9_fullgrid | 0 | 0 | 0 | 0 |

## Refresh Protocol

Refresh action: `replace_original_source_with_rerun_slug`.

1. Run remote q9 reruns until `docs/remote_q9_rerun_completion_audit.md` reports all required sources complete.
2. Backfill predictor features for complete rerun slugs using `scripts/build_missing_predictor_features_from_metrics.py`.
3. Rebuild this ingestion manifest and confirm `ready_for_final_main_refresh=1` for the replacement rows.
4. Refresh the final-main source inventory so each `original_source` is replaced by its `rerun_slug` with `new_evidence_tier=q9_fullgrid`.
5. Rerun final-main figures, paper-readiness reports, and critics before upgrading any q9/full-grid claim.

## Artifacts

- `results/aaai_stress/remote_q9_ingestion_manifest.csv`
- `docs/remote_q9_ingestion_manifest.md`
