# Metric Definitions

Let `y_{t+1:t+h}` be the realized horizon, `f_{t+1:t+h}` be a point forecast, `b_{t+1:t+h}` be a baseline forecast, and `q_tau(t+k)` be a forecasted quantile.

All metrics are computed per rolling window, then aggregated by dataset, domain, model, horizon, and signal bin with confidence intervals.

## Accuracy

Mean absolute error:

```text
MAE = (1/h) * sum_k |y_{t+k} - f_{t+k}|
```

Root mean squared error:

```text
RMSE = sqrt((1/h) * sum_k (y_{t+k} - f_{t+k})^2)
```

Symmetric MAPE, with epsilon protection:

```text
sMAPE = (1/h) * sum_k 2|y_k - f_k| / (|y_k| + |f_k| + eps)
```

Weighted absolute percentage error:

```text
WAPE = sum_k |y_k - f_k| / (sum_k |y_k| + eps)
```

MASE with in-sample seasonal naive scale:

```text
MASE = MAE / mean_{i=m+1..T} |x_i - x_{i-m}|
```

## Relative Failure

Relative error ratio:

```text
RER = Error(TSFM) / (Error(baseline) + eps)
```

Failure at tolerance `delta`:

```text
failure_delta = 1[RER > 1 + delta]
```

Severe failure:

```text
severe_failure = 1[RER > 1.25]
```

For each window, the primary baseline comparator is the best simple baseline selected by validation-only protocol, never by test-window hindsight.

## Excess-Variance Degeneration

Use first differences over the forecast horizon:

```text
Delta y_k = y_{t+k} - y_{t+k-1}
Delta f_k = f_{t+k} - f_{t+k-1}
```

Forecast variance ratio:

```text
FVR = Var(Delta f) / (Var(Delta y) + eps)
```

Prediction amplitude ratio:

```text
PAR = mean_k |Delta f_k| / (mean_k |Delta y_k| + eps)
```

Excess movement score:

```text
EMS = max(0, PAR - 1)
```

An excess-variance degeneration window is flagged when `RER > 1 + delta` and either `FVR > tau_var` or `PAR > tau_amp`. Default sensitivity grid: `tau_var in {1.5, 2.0, 3.0}`, `tau_amp in {1.25, 1.5, 2.0}`.

## Over-Smoothing Degeneration

Flatness score:

```text
FS = 1 - min(1, mean_k |Delta f_k| / (mean_k |Delta y_k| + eps))
```

`FS` is near `1` when the forecast is almost flat relative to realized movement and near `0` when forecast movement matches or exceeds realized movement.

Spike set using absolute changes:

```text
S_y(K) = indices of top K values of |Delta y|
S_f(K) = indices of top K values of |Delta f|
SpikeRecall@K = |S_y(K) intersect S_f(K)| / K
```

Peak timing error:

```text
PTE = min_{i in S_f(K)} |i - argmax_j |Delta y_j||
```

An over-smoothing degeneration window is flagged when `RER > 1 + delta`, `FS > tau_flat`, and spike recall is below the locked threshold.

## Calibration Degeneration

For an interval `[q_alpha/2, q_1-alpha/2]`:

```text
Coverage = mean_k 1[q_low,k <= y_k <= q_high,k]
IntervalWidth = mean_k (q_high,k - q_low,k)
```

Coverage failure for a nominal 90% interval:

```text
Coverage not in [0.85, 0.95]
```

Pinball loss for quantile `tau`:

```text
L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
```

Weighted quantile loss:

```text
WQL = 2 * sum_{tau,k} L_tau(y_k, q_tau,k) / (sum_k |y_k| + eps)
```

CRPS:

```text
CRPS(F, y) = integral_{-inf}^{inf} (F(z) - 1[y <= z])^2 dz
```

For sample forecasts `x_1...x_M`, use:

```text
CRPS_sample = mean_i |x_i - y| - 0.5 * mean_{i,j} |x_i - x_j|
```

Financial VaR-style tests:

- Kupiec unconditional coverage tests whether exception frequency matches nominal alpha.
- Christoffersen independence/conditional coverage tests whether exceptions are independent and correctly covered.

## Horizon Degradation

For each horizon index `k`, compute a relative error ratio `RER_k`. Fit:

```text
RER_k = beta_0 + beta_1 * k + e_k
```

`beta_1` is the horizon degradation slope. Positive slope means relative performance deteriorates over the horizon.

## Diagnostic Predictor Target

Primary binary target:

```text
Y_fail = 1[Error(TSFM) > (1 + delta) * Error(best_simple_baseline)]
```

Secondary multiclass target:

```text
class in {no_failure, excess_variance, over_smoothing, calibration, mixed}
```

Where multiple degeneration flags fire, use `mixed` for the main classifier and keep one-vs-rest labels for diagnostic plots.
