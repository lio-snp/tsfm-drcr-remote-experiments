# Remote q9 Main-Table-Style Results

This table summarizes the 17 completed P0 Moirai / TimesFM q9 full-grid reruns. One row is one frozen rerun source.

Reading guide: `MAE-RER = TSFM MAE / baseline MAE`; values below 1 mean TSFM is better than the baseline, values above 1 mean TSFM is worse. `Clipped mean` caps each window RER at 5 so near-zero baseline denominators do not dominate the display table. Raw means are preserved in the CSV.

- Sources complete for ingestion: 17 / 17.
- Forecast windows covered: 408.
- Evidence tier after ingestion: q9_fullgrid for all rows.
- Nominal interval target: 90 percent empirical coverage.

| Family | Model | Dataset | Role | Windows | Median MAE-RER | Clipped Mean MAE-RER | Worse >5% | TSFM <= Baseline | Mean 90% Cov | OverSmooth | ExcessVar | Takeaway |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| moirai | moirai_1_1_base | covid_deaths/D/short | failure_target | 8 | 1.32 | 1.9 | 75% | 25% | 52% | 0% | 62% | failure target confirmed; TSFM risk persists |
| moirai | moirai_1_1_large | covid_deaths/D/short | failure_target | 8 | 1.26 | 1.94 | 88% | 12% | 61% | 0% | 75% | failure target confirmed; TSFM risk persists |
| moirai | moirai_1_1_small | covid_deaths/D/short | failure_target | 8 | 1.52 | 2.22 | 75% | 25% | 70% | 0% | 75% | failure target confirmed; TSFM risk persists |
| timesfm | timesfm_2_5_m128 | covid_deaths/D/short | failure_target | 128 | 1 | 1.36 | 44% | 50% | 78% | 7% | 29% | failure target confirmed; TSFM risk persists |
| timesfm | timesfm_2_5_m16 | covid_deaths/D/short | failure_target | 16 | 1.31 | 2.38 | 69% | 19% | 70% | 12% | 44% | failure target confirmed; TSFM risk persists |
| timesfm | timesfm_2_5_m64 | covid_deaths/D/short | failure_target | 64 | 1.16 | 1.71 | 53% | 39% | 77% | 6% | 39% | failure target confirmed; TSFM risk persists |
| timesfm | timesfm_2_5_m8 | covid_deaths/D/short | failure_target | 8 | 1.31 | 1.96 | 75% | 12% | 60% | 12% | 38% | failure target confirmed; TSFM risk persists |
| moirai | moirai_1_1_base | loop_seattle/H/short | positive_control | 8 | 0.52 | 0.56 | 0% | 100% | 84% | 0% | 0% | positive control stable; TSFM beats baseline |
| moirai | moirai_1_1_large | loop_seattle/H/short | positive_control | 8 | 0.42 | 0.41 | 0% | 100% | 77% | 0% | 0% | positive control stable; TSFM beats baseline |
| moirai | moirai_1_1_small | loop_seattle/H/short | positive_control | 8 | 0.77 | 0.77 | 0% | 88% | 79% | 0% | 0% | positive control stable; TSFM beats baseline |
| moirai | moirai2_loop_m8 | LOOP_SEATTLE/H/short | positive_control | 8 | 0.33 | 0.35 | 0% | 100% | 66% | 0% | 0% | positive control stable; TSFM beats baseline |
| timesfm | timesfm_2_5_loop_m32 | LOOP_SEATTLE/H/short | positive_control | 32 | 0.43 | 0.44 | 0% | 100% | 82% | 0% | 0% | positive control stable; TSFM beats baseline |
| timesfm | timesfm_2_5_loop_m8 | LOOP_SEATTLE/H/short | positive_control | 8 | 0.46 | 0.49 | 0% | 100% | 80% | 0% | 0% | positive control stable; TSFM beats baseline |
| timesfm | timesfm_2_5_solar_m8 | solar/10T/short | positive_control | 8 | 0.85 | 1.21 | 38% | 62% | 64% | 12% | 0% | coverage weakness |
| timesfm | timesfm_2_5_bizitobs_m30 | bizitobs_application/10S/short | stress_target | 30 | 0.57 | 0.68 | 13% | 70% | 76% | 7% | 0% | mostly TSFM advantage |
| timesfm | timesfm_2_5_bizitobs_m8 | bizitobs_application/10S/short | stress_target | 2 | 0.76 | 0.76 | 50% | 50% | 95% | 50% | 0% | stress risk; over-smoothing dominates |
| timesfm | timesfm_2_5_finance_fred | FRED finance stress seed | stress_target | 56 | 1.07 | 1.17 | 62% | 20% | 84% | 59% | 0% | stress risk; over-smoothing dominates |

CSV with audit columns and raw mean RER: `results/aaai_stress/remote_q9_main_table.csv`.

Interpretation boundary: this table supports ingestion and result inspection for the remote q9 reruns. It is not a replacement for the lead-machine final-main rebuild, which must refresh the source inventory to point at the `_oral_sidecar_rerun` slugs before changing final paper tables.
