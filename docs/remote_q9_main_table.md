# Remote q9 Literature-Aligned Main Table

This table summarizes the 17 completed P0 Moirai / TimesFM q9 full-grid reruns. It is generated from raw forecasts and history sidecars by `scripts/build_remote_q9_main_table.py`; it does not alter frozen manifests, forecasts, or DRCR policy settings.

## Result Summary

- Complete sources: `17 / 17`; forecast windows: `408`; forecast points: `13520`.
- Failure targets: `7 / 7` rows have both RelMAE and RelRMSE above 1, so the point-forecast failure signal is not specific to one loss function.
- Positive controls: `7 / 7` rows have both RelMAE and RelRMSE below 1.
- Stress targets are mixed: `2 / 3` rows beat their baseline on both pooled point metrics; the finance row loses to an oracle-selected comparator and must not be read as a deployable baseline comparison.
- Probabilistic calibration: `12 / 17` rows fall below the nominal 80% q10-q90 coverage. q9-WQL is reported beside coverage so wide intervals are not rewarded merely for covering more observations.
- AvgRelMAE is intentionally `NA` for `7` rows containing zero candidate or baseline series-level MAE. No epsilon, clipping, or hidden replacement value is used.
- Quantile audit: `1` forecast point(s) have an internal adjacent-quantile crossing; raw values are retained rather than silently sorted. No q10-q90 outer interval is reversed.

## Main Table

| Family | Model | Dataset | Role | Baseline | W/S | RelMAE | RelRMSE | AvgRelMAE [valid S] | MASE [valid W] | PB(MAE) | Sign BH q | q9-WQL | q10-q90 Cov |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moirai | moirai_1_1_base | covid_deaths/D/short | failure_target | AutoETS | 8/8 | 1.240 | 1.256 | NA [7/8] | 44.183 [8/8] | 25.0% | 0.351 | 0.113 | 52.5% |
| moirai | moirai_1_1_large | covid_deaths/D/short | failure_target | AutoETS | 8/8 | 1.586 | 1.633 | NA [7/8] | 41.847 [8/8] | 12.5% | 0.100 | 0.148 | 60.8% |
| moirai | moirai_1_1_small | covid_deaths/D/short | failure_target | AutoETS | 8/8 | 1.766 | 1.867 | NA [7/8] | 48.409 [8/8] | 25.0% | 0.351 | 0.167 | 69.6% |
| timesfm | timesfm_2_5_m128 | covid_deaths/D/short | failure_target | AutoETS | 128/128 | 1.358 | 1.498 | NA [77/128] | 29.945 [107/128] | 23.4% | 0.003 | 0.029 | 78.4% |
| timesfm | timesfm_2_5_m16 | covid_deaths/D/short | failure_target | AutoETS | 16/16 | 1.250 | 1.237 | NA [12/16] | 63.156 [15/16] | 12.5% | 0.015 | 0.130 | 69.6% |
| timesfm | timesfm_2_5_m64 | covid_deaths/D/short | failure_target | AutoETS | 64/64 | 1.693 | 1.915 | NA [43/64] | 27.543 [58/64] | 21.9% | 0.003 | 0.031 | 77.1% |
| timesfm | timesfm_2_5_m8 | covid_deaths/D/short | failure_target | AutoETS | 8/8 | 1.254 | 1.238 | NA [7/8] | 42.828 [8/8] | 12.5% | 0.100 | 0.117 | 60.0% |
| moirai | moirai2_loop_m8 | loop_seattle/H/short | positive_control | SeasonalNaive | 8/8 | 0.351 | 0.364 | 0.349 [8/8] | 0.509 [8/8] | 100.0% | 0.015 | 0.054 | 66.4% |
| moirai | moirai_1_1_base | loop_seattle/H/short | positive_control | SeasonalNaive | 8/8 | 0.561 | 0.607 | 0.540 [8/8] | 0.813 [8/8] | 100.0% | 0.015 | 0.086 | 84.4% |
| moirai | moirai_1_1_large | loop_seattle/H/short | positive_control | SeasonalNaive | 8/8 | 0.411 | 0.471 | 0.390 [8/8] | 0.592 [8/8] | 100.0% | 0.015 | 0.063 | 76.8% |
| moirai | moirai_1_1_small | loop_seattle/H/short | positive_control | SeasonalNaive | 8/8 | 0.780 | 0.823 | 0.761 [8/8] | 1.122 [8/8] | 87.5% | 0.100 | 0.124 | 79.2% |
| timesfm | timesfm_2_5_loop_m32 | loop_seattle/H/short | positive_control | SeasonalNaive | 32/32 | 0.450 | 0.481 | 0.430 [32/32] | 0.623 [32/32] | 100.0% | <0.001 | 0.058 | 82.2% |
| timesfm | timesfm_2_5_loop_m8 | loop_seattle/H/short | positive_control | SeasonalNaive | 8/8 | 0.487 | 0.489 | 0.477 [8/8] | 0.703 [8/8] | 100.0% | 0.015 | 0.074 | 80.2% |
| timesfm | timesfm_2_5_solar_m8 | solar/10T/short | positive_control | SeasonalNaive | 8/8 | 0.812 | 0.738 | 1.013 [8/8] | 0.555 [8/8] | 62.5% | 0.772 | 0.643 | 63.5% |
| timesfm | timesfm_2_5_bizitobs_m30 | bizitobs_application/10S/short | stress_target | AutoARIMA | 30/2 | 0.573 | 0.687 | 0.560 [2/2] | 9.319 [30/30] | 100.0% | 0.567 | 0.030 | 76.4% |
| timesfm | timesfm_2_5_bizitobs_m8 | bizitobs_application/10S/short | stress_target | AutoARIMA | 2/2 | 0.162 | 0.172 | 0.468 [2/2] | 1.508 [2/2] | 50.0% | 1.000 | 0.008 | 95.0% |
| timesfm | timesfm_2_5_finance_fred | FRED finance stress seed | stress_target | OracleBestSimple | 56/14 | 1.326 | 1.185 | 1.154 [14/14] | 0.788 [56/56] | 0.0% | 0.001 | 0.043 | 84.1% |

Rebuild command:

```bash
python3 scripts/build_remote_q9_main_table.py
```

## Column Meaning

| Column | Question answered | Exact computation | Direction |
| --- | --- | --- | --- |
| W/S | How much evidence is in the row? | Number of forecast windows / distinct series. | More evidence improves stability; it is not a quality score. |
| RelMAE | Does the model reduce ordinary absolute error against the frozen baseline? | Pooled model MAE divided by pooled baseline MAE over all forecast points. Raw MAEs remain in the CSV. | Below 1 is better. |
| RelRMSE | Does the conclusion survive a loss that penalizes large misses more strongly? | Pooled model RMSE divided by pooled baseline RMSE. Raw RMSEs remain in the CSV. | Below 1 is better. |
| AvgRelMAE | Is the relative MAE result robust across differently scaled series? | Davydenko-Fildes weighted geometric mean of per-series MAE ratios, weighted by available forecast errors. It is `NA` unless every series has positive model and baseline MAE. | Below 1 is better. |
| MASE | How large is model MAE relative to an in-sample seasonal-naive scale? | Mean absolute scaled error recomputed from each pre-origin `full_context_values`; the sidecar seasonal period is used, with 1 only when unspecified. Valid windows are shown. | Lower is better; below 1 beats the in-sample naive scale. |
| PB(MAE) | Is improvement broad or driven by a few large series? | `100 * mean(I[series MAE_model < series MAE_baseline])`; repeated windows are aggregated within series first. | Higher is better. |
| Sign BH q | Is the win/loss imbalance distinguishable from 50/50? | Two-sided exact sign test across non-tied series, followed by Benjamini-Hochberg correction across the 17 rows. | Smaller is stronger evidence; direction comes from PB(MAE). |
| q9-WQL | Are all nine forecast quantiles jointly accurate under a proper quantile loss? | Chronos-style WQL over q10,...,q90: average normalized pinball loss with `2 * sum(loss) / sum(abs(actual))`. | Lower is better. |
| q10-q90 Cov | Does the outer interval calibrate to its stated probability? | Point-level fraction satisfying `q10 <= actual <= q90`. The nominal target is 80%, not 90%. | Closer to 80% is better. |

## Literature Basis

- MAE, RMSE, relative MAE, Percent Better, and MASE follow the forecast-evaluation taxonomy and caveats in [Hyndman and Koehler (2006, International Journal of Forecasting)](https://doi.org/10.1016/j.ijforecast.2006.03.001).
- AvgRelMAE follows the weighted geometric aggregation in [Davydenko and Fildes (2013, International Journal of Forecasting)](https://doi.org/10.1016/j.ijforecast.2012.09.002).
- Percent Better is paired with a sign test rather than an invented 5% margin; the sign-test use in forecast comparison is discussed by [Flores (1986, International Journal of Forecasting)](https://doi.org/10.1016/0169-2070(86)90093-2).
- Multiple sign tests are controlled with the false-discovery-rate procedure of [Benjamini and Hochberg (1995, JRSS-B)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
- q9-WQL uses the exact nine-level definition used by [Chronos (2024, Transactions on Machine Learning Research)](https://arxiv.org/abs/2403.07815), grounded in proper scoring rules reviewed by [Gneiting and Raftery (2007, JASA)](https://doi.org/10.1198/016214506000001437).
- Coverage is interpreted jointly with sharpness/proper score, following [Gneiting, Balabdaoui, and Raftery (2007, JRSS-B)](https://doi.org/10.1111/j.1467-9868.2007.00587.x).

## Removed From The Formal Main Table

The former `Worse >5%`, `OverSmooth`, `ExcessVar`, and cap-at-5 mean RER columns are not confirmatory main-table metrics. Their exact thresholds are project diagnostics rather than established benchmark measures. Frozen window-level files retain them for exploratory mechanism analysis, but this generator neither reads them into the formal table nor rewrites them.

## Interpretation Boundaries

- RelMAE, RelRMSE, and q9-WQL pool absolute losses and are therefore scale-weighted. AvgRelMAE and PB(MAE) provide complementary series-level views.
- `OracleBestSimple` in the FRED finance row was selected by minimizing error on each target window. It is an oracle stress comparator, not a deployable or validation-selected baseline.
- Small rows, especially the two-series BizITObs m8 slice, have low inferential power even when effect sizes look large.
- This is a remote-rerun evidence table, not a direct edit to final paper tables. The lead machine must ingest the `_oral_sidecar_rerun` slugs and rebuild the locked final-main inventory before quoting final-paper aggregates.
- The required all-window DRCR endpoint and its mutually exclusive failure/non-failure decomposition are specified in `docs/remote_q9_drcr_repair_evaluation_addendum.md`.

Machine-readable table: `results/aaai_stress/remote_q9_main_table.csv`.
