# Forecast Artifact Contract

The reproduction-first study has two artifact layers.

## Layer 1: Benchmark Aggregate Results

Used for immediate anchoring to GIFT-Eval public leaderboard-style results.

Required columns:

- `dataset`
- `model`
- `domain`
- `num_variates`
- `eval_metrics/MASE[0.5]`
- `eval_metrics/mean_weighted_sum_quantile_loss`
- `eval_metrics/MAE[0.5]`
- `eval_metrics/RMSE[mean]`

This layer supports:

- reproduction-slice sanity checks,
- aggregate TSFM-vs-baseline failure ratios,
- domain/regime summaries.

It does **not** support:

- forecast variance ratio,
- flatness score,
- spike recall,
- calibration recovery time,
- raw forecast plotting.

## Layer 2: Window-Level Raw Forecasts

Required before claiming full low-signal forecast degeneration.

Required columns:

- `run_id`
- `dataset`
- `series_id`
- `domain`
- `regime`
- `model`
- `baseline_family`
- `baseline_forecast`, required for post-hoc repair probes and DM-style raw loss comparisons
- `origin`
- `horizon_index`
- `context_length`
- `horizon`
- `actual`
- `forecast_mean`
- `forecast_median`
- `forecast_q10`
- `forecast_q50`
- `forecast_q90`
- `forecast_sample_id`, optional for sample forecasts
- `source_commit`
- `model_id`
- `model_version`

This layer supports all degeneration metrics in `docs/metrics.md`.

Quantile artifact rule:

- `forecast_q10`, `forecast_q50`, and `forecast_q90` remain the backward-compatible minimum triplet. They support q10/q50/q90 WQL proxy and q10-q90 coverage checks.
- Richer quantile grids should be persisted as additional `forecast_qXX` columns, for example `forecast_q2p5`, `forecast_q05`, `forecast_q25`, `forecast_q75`, or `forecast_q95`. The robustness code discovers these columns dynamically and upgrades WQL from triplet proxy to available-grid WQL when more than three interior quantile levels are present.
- Chronos-Bolt raw reruns can request richer grids with `--quantile-levels`; TimesFM and Moirai2 exporters persist all quantile levels exposed by the model output path. Moirai1 currently derives q10/q50/q90 from samples but does not persist sample paths.
- CRPS is claimable only when sample paths or horizon-level sample forecasts are persisted and tied back to `forecast_sample_id`. Until then, CRPS remains an artifact gap, not a negative result.

## Layer 3: Window-Level Metrics

Derived from Layer 2.

Required columns:

- `run_id`
- `dataset`
- `series_id`
- `domain`
- `regime`
- `model`
- `baseline`
- `origin`
- `context_length`
- `horizon`
- `mae`
- `rmse`
- `mase`
- `baseline_mae`
- `relative_error_ratio`
- `forecast_variance_ratio`
- `prediction_amplitude_ratio`
- `flatness_score`
- `spike_recall`
- `empirical_coverage_90`, if probabilistic
- `pinball_loss`, if at least q10/q50/q90 are available
- `mean_weighted_quantile_loss`, exact available-grid WQL only when a richer quantile grid is present
- `crps`, only if sample paths or sample forecasts are persisted

## Contract Rule

Aggregate GIFT-Eval result files may be used to motivate where to rerun models, but the paper must not present aggregate rows as direct evidence for excess-variance, over-smoothing, or calibration degeneration. Those claims require Layer 2 and Layer 3 artifacts.

Paper-claim boundary:

- Unified MAE-RER remains the main cross-family failure/repair metric because it is available for every current raw rerun.
- MASE/RMSE/sMAPE/WAPE and q10/q50/q90 WQL/coverage are robustness metrics on the existing artifacts.
- Full-grid WQL can become a paper-faithful probabilistic robustness claim only after the affected slices are rerun with richer quantile columns.
- CRPS can become a main probabilistic claim only after sample artifacts are persisted.
