#!/usr/bin/env python3
"""Build a literature-aligned summary of the completed remote q9 reruns."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "aaai_stress"
PLAN_PATH = OUT_DIR / "remote_q9_rerun_plan.csv"
CSV_OUT = OUT_DIR / "remote_q9_main_table.csv"
DOC_OUT = ROOT / "docs" / "remote_q9_main_table.md"

QUANTILES = tuple((level / 10.0, f"forecast_q{level}0") for level in range(1, 10))
NOMINAL_COVERAGE = 0.80
ROLE_ORDER = {"failure_target": 0, "positive_control": 1, "stress_target": 2}
FAMILY_ORDER = {"moirai": 0, "timesfm": 1}


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


raise_csv_field_limit()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty main table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row.get(key)!r}")
    return value


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def window_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row.get("series_id", ""), row.get("origin", ""), row.get("window_index", "")


def pooled_point_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Raw forecast rows are empty")
    model_abs = 0.0
    baseline_abs = 0.0
    model_sq = 0.0
    baseline_sq = 0.0
    for row in rows:
        actual = finite_float(row, "actual")
        model = finite_float(row, "forecast_mean")
        baseline = finite_float(row, "baseline_forecast")
        model_error = actual - model
        baseline_error = actual - baseline
        model_abs += abs(model_error)
        baseline_abs += abs(baseline_error)
        model_sq += model_error**2
        baseline_sq += baseline_error**2
    if baseline_abs <= 0.0 or baseline_sq <= 0.0:
        raise ValueError("The pooled baseline error is zero; RelMAE/RelRMSE are undefined")
    count = len(rows)
    model_mae = model_abs / count
    baseline_mae = baseline_abs / count
    model_rmse = math.sqrt(model_sq / count)
    baseline_rmse = math.sqrt(baseline_sq / count)
    return {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "model_rmse": model_rmse,
        "baseline_rmse": baseline_rmse,
        "relmae": model_mae / baseline_mae,
        "relrmse": model_rmse / baseline_rmse,
    }


def exact_sign_test_pvalue(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(trials, index) for index in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def series_comparison_metrics(rows: list[dict[str, str]]) -> dict[str, float | int | str | None]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)

    weighted_logs: list[tuple[int, float]] = []
    wins = 0
    losses = 0
    ties = 0
    zero_baseline = 0
    zero_model = 0
    for series_rows in grouped.values():
        count = len(series_rows)
        model_mae = sum(
            abs(finite_float(row, "actual") - finite_float(row, "forecast_mean"))
            for row in series_rows
        ) / count
        baseline_mae = sum(
            abs(finite_float(row, "actual") - finite_float(row, "baseline_forecast"))
            for row in series_rows
        ) / count

        if model_mae < baseline_mae:
            wins += 1
        elif model_mae > baseline_mae:
            losses += 1
        else:
            ties += 1

        if baseline_mae <= 0.0:
            zero_baseline += 1
        if model_mae <= 0.0:
            zero_model += 1
        if model_mae > 0.0 and baseline_mae > 0.0:
            weighted_logs.append((count, math.log(model_mae / baseline_mae)))

    total_series = len(grouped)
    valid_series = len(weighted_logs)
    if valid_series == total_series:
        total_weight = sum(weight for weight, _ in weighted_logs)
        avg_relmae = math.exp(
            sum(weight * log_ratio for weight, log_ratio in weighted_logs) / total_weight
        )
        status = "defined"
    else:
        avg_relmae = None
        status = "undefined_nonpositive_series_mae"

    return {
        "series": total_series,
        "avg_relmae": avg_relmae,
        "avg_relmae_status": status,
        "avg_relmae_valid_series": valid_series,
        "zero_baseline_mae_series": zero_baseline,
        "zero_model_mae_series": zero_model,
        "percent_better_mae": 100.0 * wins / total_series,
        "percent_tied_mae": 100.0 * ties / total_series,
        "series_wins": wins,
        "series_losses": losses,
        "series_ties": ties,
        "sign_test_pvalue": exact_sign_test_pvalue(wins, losses),
    }


def q9_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    target_magnitude = 0.0
    pinball_total = 0.0
    covered = 0
    crossings = 0
    outer_reversals = 0
    for row in rows:
        actual = finite_float(row, "actual")
        quantiles = [finite_float(row, column) for _, column in QUANTILES]
        target_magnitude += abs(actual)
        if any(left > right for left, right in zip(quantiles, quantiles[1:])):
            crossings += 1
        if quantiles[0] > quantiles[-1]:
            outer_reversals += 1
        if quantiles[0] <= actual <= quantiles[-1]:
            covered += 1
        for (tau, _), prediction in zip(QUANTILES, quantiles):
            error = actual - prediction
            pinball_total += max(tau * error, (tau - 1.0) * error)
    if target_magnitude <= 0.0:
        raise ValueError("q9-WQL is undefined because sum(abs(actual)) is zero")
    coverage = covered / len(rows)
    return {
        "q9_wql": 2.0 * pinball_total / (len(QUANTILES) * target_magnitude),
        "q10_q90_coverage": coverage,
        "q10_q90_coverage_abs_error": abs(coverage - NOMINAL_COVERAGE),
        "quantile_crossing_points": crossings,
        "outer_interval_reversal_points": outer_reversals,
    }


def seasonal_scale(values: list[float], season_length: int) -> float | None:
    if season_length < 1:
        raise ValueError(f"Invalid MASE season length: {season_length}")
    if len(values) <= season_length:
        return None
    scale = sum(
        abs(values[index] - values[index - season_length])
        for index in range(season_length, len(values))
    ) / (len(values) - season_length)
    return scale if scale > 0.0 else None


def mase_metrics(
    raw_rows: list[dict[str, str]],
    history_rows: list[dict[str, str]],
) -> dict[str, float | int | None]:
    raw_by_window: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_window[window_key(row)].append(row)

    scaled_error_sum = 0.0
    valid_points = 0
    valid_windows = 0
    undefined_windows = 0
    seen: set[tuple[str, str, str]] = set()
    for history in history_rows:
        key = window_key(history)
        if key in seen:
            raise ValueError(f"Duplicate history sidecar key: {key}")
        seen.add(key)
        if key not in raw_by_window:
            raise ValueError(f"History sidecar has no matching raw window: {key}")
        context = [float(value) for value in json.loads(history["full_context_values"])]
        if not all(math.isfinite(value) for value in context):
            raise ValueError(f"Non-finite full context values for window {key}")
        season_length = int(float(history.get("baseline_season_length") or 1))
        scale = seasonal_scale(context, season_length)
        if scale is None:
            undefined_windows += 1
            continue
        window_rows = raw_by_window[key]
        scaled_error_sum += sum(
            abs(finite_float(row, "actual") - finite_float(row, "forecast_mean")) / scale
            for row in window_rows
        )
        valid_points += len(window_rows)
        valid_windows += 1

    missing_history = set(raw_by_window) - seen
    if missing_history:
        raise ValueError(f"Raw windows missing from history sidecar: {sorted(missing_history)[:3]}")
    return {
        "mean_mase": scaled_error_sum / valid_points if valid_points else None,
        "mase_valid_windows": valid_windows,
        "mase_undefined_windows": undefined_windows,
    }


def benjamini_hochberg(values: list[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(values) if value is not None]
    if not indexed:
        return [None] * len(values)
    ranked = sorted(indexed, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 1.0
    tests = len(ranked)
    for rank_index in range(tests - 1, -1, -1):
        original_index, pvalue = ranked[rank_index]
        rank = rank_index + 1
        running = min(running, pvalue * tests / rank)
        adjusted[original_index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def baseline_label(mode: str) -> str:
    return {
        "auto_ets": "AutoETS",
        "seasonal_naive": "SeasonalNaive",
        "auto_arima": "AutoARIMA",
        "best_simple": "OracleBestSimple",
    }.get(mode, mode)


def summarize_source(plan_row: dict[str, str]) -> dict[str, object]:
    raw_path = ROOT / plan_row["expected_raw_path"]
    history_path = ROOT / plan_row["expected_history_sidecar_path"]
    slug = raw_path.stem
    metrics_path = ROOT / "results" / "window_metrics" / f"{slug}_metrics.csv"
    raw_rows = read_csv(raw_path)
    history_rows = read_csv(history_path)
    metric_rows = read_csv(metrics_path)

    required = {
        "actual",
        "forecast_mean",
        "baseline_forecast",
        "series_id",
        "origin",
        "window_index",
        *(column for _, column in QUANTILES),
    }
    missing = required - set(raw_rows[0]) if raw_rows else required
    if missing:
        raise ValueError(f"{raw_path.name} is missing columns: {sorted(missing)}")

    window_count = len({window_key(row) for row in raw_rows})
    manifest_windows = int(plan_row["manifest_windows"])
    if window_count != manifest_windows:
        raise ValueError(f"{slug}: raw windows {window_count} != manifest windows {manifest_windows}")
    if len(history_rows) != manifest_windows or len(metric_rows) != manifest_windows:
        raise ValueError(
            f"{slug}: expected {manifest_windows} history/metric rows, got "
            f"{len(history_rows)}/{len(metric_rows)}"
        )

    models = {row["model"] for row in raw_rows}
    datasets = {row["dataset"] for row in raw_rows}
    baseline_modes = {row["baseline_mode"] for row in raw_rows}
    if len(models) != 1 or len(datasets) != 1 or len(baseline_modes) != 1:
        raise ValueError(f"{slug}: model, dataset, and baseline mode must each be unique")

    point = pooled_point_metrics(raw_rows)
    series = series_comparison_metrics(raw_rows)
    probabilistic = q9_metrics(raw_rows)
    scaled = mase_metrics(raw_rows, history_rows)
    baseline_mode = next(iter(baseline_modes))
    return {
        "family": plan_row["family"],
        "model": next(iter(models)),
        "dataset": next(iter(datasets)),
        "role": plan_row["role"],
        "evidence_tier": "q9_fullgrid",
        "baseline_mode": baseline_mode,
        "baseline_protocol": (
            "target_window_oracle_selection" if baseline_mode == "best_simple" else "frozen_command_baseline"
        ),
        "windows": window_count,
        "series": series["series"],
        "forecast_points": len(raw_rows),
        "model_mae": fmt(point["model_mae"]),
        "baseline_mae": fmt(point["baseline_mae"]),
        "model_rmse": fmt(point["model_rmse"]),
        "baseline_rmse": fmt(point["baseline_rmse"]),
        "relmae": fmt(point["relmae"]),
        "relrmse": fmt(point["relrmse"]),
        "avg_relmae": fmt(series["avg_relmae"]),
        "avg_relmae_status": series["avg_relmae_status"],
        "avg_relmae_valid_series": series["avg_relmae_valid_series"],
        "zero_baseline_mae_series": series["zero_baseline_mae_series"],
        "zero_model_mae_series": series["zero_model_mae_series"],
        "mean_mase": fmt(scaled["mean_mase"]),
        "mase_valid_windows": scaled["mase_valid_windows"],
        "mase_undefined_windows": scaled["mase_undefined_windows"],
        "percent_better_mae": fmt(series["percent_better_mae"]),
        "percent_tied_mae": fmt(series["percent_tied_mae"]),
        "series_wins": series["series_wins"],
        "series_losses": series["series_losses"],
        "series_ties": series["series_ties"],
        "sign_test_pvalue": fmt(series["sign_test_pvalue"], digits=8),
        "sign_test_bh_qvalue": "",
        "q9_wql": fmt(probabilistic["q9_wql"]),
        "q10_q90_coverage": fmt(probabilistic["q10_q90_coverage"]),
        "q10_q90_coverage_abs_error": fmt(probabilistic["q10_q90_coverage_abs_error"]),
        "quantile_crossing_points": probabilistic["quantile_crossing_points"],
        "outer_interval_reversal_points": probabilistic["outer_interval_reversal_points"],
        "raw_forecast_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "history_sidecar_path": str(history_path.relative_to(ROOT)).replace("\\", "/"),
        "window_metrics_path": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
        "rerun_slug": slug,
    }


def row_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        ROLE_ORDER.get(str(row["role"]), 99),
        str(row["dataset"]).lower(),
        FAMILY_ORDER.get(str(row["family"]), 99),
        str(row["model"]),
    )


def md_number(value: object, digits: int = 3) -> str:
    text = str(value)
    if text == "":
        return "NA"
    return f"{float(text):.{digits}f}"


def md_percent(value: object) -> str:
    return f"{float(str(value)):.1f}%"


def md_pvalue(value: object) -> str:
    if str(value) == "":
        return "NA"
    number = float(str(value))
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def markdown_table(rows: Iterable[dict[str, str]]) -> str:
    lines = [
        "| Family | Model | Dataset | Role | Baseline | W/S | RelMAE | RelRMSE | AvgRelMAE [valid S] | MASE [valid W] | PB(MAE) | Sign BH q | q9-WQL | q10-q90 Cov |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        avg_relmae = (
            f"{md_number(row['avg_relmae'])} [{row['avg_relmae_valid_series']}/{row['series']}]"
            if row["avg_relmae"]
            else f"NA [{row['avg_relmae_valid_series']}/{row['series']}]"
        )
        mase = f"{md_number(row['mean_mase'])} [{row['mase_valid_windows']}/{row['windows']}]"
        values = [
            row["family"],
            row["model"],
            row["dataset"],
            row["role"],
            baseline_label(row["baseline_mode"]),
            f"{row['windows']}/{row['series']}",
            md_number(row["relmae"]),
            md_number(row["relrmse"]),
            avg_relmae,
            mase,
            md_percent(row["percent_better_mae"]),
            md_pvalue(row["sign_test_bh_qvalue"]),
            md_number(row["q9_wql"]),
            md_percent(100.0 * float(row["q10_q90_coverage"])),
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def build_doc(rows: list[dict[str, str]]) -> str:
    total_windows = sum(int(row["windows"]) for row in rows)
    total_points = sum(int(row["forecast_points"]) for row in rows)
    failure_rows = [row for row in rows if row["role"] == "failure_target"]
    positive_rows = [row for row in rows if row["role"] == "positive_control"]
    stress_rows = [row for row in rows if row["role"] == "stress_target"]
    failure_both_worse = sum(
        float(row["relmae"]) > 1.0 and float(row["relrmse"]) > 1.0 for row in failure_rows
    )
    positive_both_better = sum(
        float(row["relmae"]) < 1.0 and float(row["relrmse"]) < 1.0 for row in positive_rows
    )
    stress_both_better = sum(
        float(row["relmae"]) < 1.0 and float(row["relrmse"]) < 1.0 for row in stress_rows
    )
    undercovered = sum(float(row["q10_q90_coverage"]) < NOMINAL_COVERAGE for row in rows)
    avg_relmae_undefined = sum(not row["avg_relmae"] for row in rows)
    crossing_points = sum(int(row["quantile_crossing_points"]) for row in rows)

    return "\n".join(
        [
            "# Remote q9 Literature-Aligned Main Table",
            "",
            "This table summarizes the 17 completed P0 Moirai / TimesFM q9 full-grid reruns. It is generated from raw forecasts and history sidecars by `scripts/build_remote_q9_main_table.py`; it does not alter frozen manifests, forecasts, or DRCR policy settings.",
            "",
            "## Result Summary",
            "",
            f"- Complete sources: `{len(rows)} / 17`; forecast windows: `{total_windows}`; forecast points: `{total_points}`.",
            f"- Failure targets: `{failure_both_worse} / {len(failure_rows)}` rows have both RelMAE and RelRMSE above 1, so the point-forecast failure signal is not specific to one loss function.",
            f"- Positive controls: `{positive_both_better} / {len(positive_rows)}` rows have both RelMAE and RelRMSE below 1.",
            f"- Stress targets are mixed: `{stress_both_better} / {len(stress_rows)}` rows beat their baseline on both pooled point metrics; the finance row loses to an oracle-selected comparator and must not be read as a deployable baseline comparison.",
            f"- Probabilistic calibration: `{undercovered} / {len(rows)}` rows fall below the nominal 80% q10-q90 coverage. q9-WQL is reported beside coverage so wide intervals are not rewarded merely for covering more observations.",
            f"- AvgRelMAE is intentionally `NA` for `{avg_relmae_undefined}` rows containing zero candidate or baseline series-level MAE. No epsilon, clipping, or hidden replacement value is used.",
            f"- Quantile audit: `{crossing_points}` forecast point(s) have an internal adjacent-quantile crossing; raw values are retained rather than silently sorted. No q10-q90 outer interval is reversed.",
            "",
            "## Main Table",
            "",
            markdown_table(rows),
            "",
            "Rebuild command:",
            "",
            "```bash",
            "python3 scripts/build_remote_q9_main_table.py",
            "```",
            "",
            "## Column Meaning",
            "",
            "| Column | Question answered | Exact computation | Direction |",
            "| --- | --- | --- | --- |",
            "| W/S | How much evidence is in the row? | Number of forecast windows / distinct series. | More evidence improves stability; it is not a quality score. |",
            "| RelMAE | Does the model reduce ordinary absolute error against the frozen baseline? | Pooled model MAE divided by pooled baseline MAE over all forecast points. Raw MAEs remain in the CSV. | Below 1 is better. |",
            "| RelRMSE | Does the conclusion survive a loss that penalizes large misses more strongly? | Pooled model RMSE divided by pooled baseline RMSE. Raw RMSEs remain in the CSV. | Below 1 is better. |",
            "| AvgRelMAE | Is the relative MAE result robust across differently scaled series? | Davydenko-Fildes weighted geometric mean of per-series MAE ratios, weighted by available forecast errors. It is `NA` unless every series has positive model and baseline MAE. | Below 1 is better. |",
            "| MASE | How large is model MAE relative to an in-sample seasonal-naive scale? | Mean absolute scaled error recomputed from each pre-origin `full_context_values`; the sidecar seasonal period is used, with 1 only when unspecified. Valid windows are shown. | Lower is better; below 1 beats the in-sample naive scale. |",
            "| PB(MAE) | Is improvement broad or driven by a few large series? | `100 * mean(I[series MAE_model < series MAE_baseline])`; repeated windows are aggregated within series first. | Higher is better. |",
            "| Sign BH q | Is the win/loss imbalance distinguishable from 50/50? | Two-sided exact sign test across non-tied series, followed by Benjamini-Hochberg correction across the 17 rows. | Smaller is stronger evidence; direction comes from PB(MAE). |",
            "| q9-WQL | Are all nine forecast quantiles jointly accurate under a proper quantile loss? | Chronos-style WQL over q10,...,q90: average normalized pinball loss with `2 * sum(loss) / sum(abs(actual))`. | Lower is better. |",
            "| q10-q90 Cov | Does the outer interval calibrate to its stated probability? | Point-level fraction satisfying `q10 <= actual <= q90`. The nominal target is 80%, not 90%. | Closer to 80% is better. |",
            "",
            "## Literature Basis",
            "",
            "- MAE, RMSE, relative MAE, Percent Better, and MASE follow the forecast-evaluation taxonomy and caveats in [Hyndman and Koehler (2006, International Journal of Forecasting)](https://doi.org/10.1016/j.ijforecast.2006.03.001).",
            "- AvgRelMAE follows the weighted geometric aggregation in [Davydenko and Fildes (2013, International Journal of Forecasting)](https://doi.org/10.1016/j.ijforecast.2012.09.002).",
            "- Percent Better is paired with a sign test rather than an invented 5% margin; the sign-test use in forecast comparison is discussed by [Flores (1986, International Journal of Forecasting)](https://doi.org/10.1016/0169-2070(86)90093-2).",
            "- Multiple sign tests are controlled with the false-discovery-rate procedure of [Benjamini and Hochberg (1995, JRSS-B)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).",
            "- q9-WQL uses the exact nine-level definition used by [Chronos (2024, Transactions on Machine Learning Research)](https://arxiv.org/abs/2403.07815), grounded in proper scoring rules reviewed by [Gneiting and Raftery (2007, JASA)](https://doi.org/10.1198/016214506000001437).",
            "- Coverage is interpreted jointly with sharpness/proper score, following [Gneiting, Balabdaoui, and Raftery (2007, JRSS-B)](https://doi.org/10.1111/j.1467-9868.2007.00587.x).",
            "",
            "## Removed From The Formal Main Table",
            "",
            "The former `Worse >5%`, `OverSmooth`, `ExcessVar`, and cap-at-5 mean RER columns are not confirmatory main-table metrics. Their exact thresholds are project diagnostics rather than established benchmark measures. Frozen window-level files retain them for exploratory mechanism analysis, but this generator neither reads them into the formal table nor rewrites them.",
            "",
            "## Interpretation Boundaries",
            "",
            "- RelMAE, RelRMSE, and q9-WQL pool absolute losses and are therefore scale-weighted. AvgRelMAE and PB(MAE) provide complementary series-level views.",
            "- `OracleBestSimple` in the FRED finance row was selected by minimizing error on each target window. It is an oracle stress comparator, not a deployable or validation-selected baseline.",
            "- Small rows, especially the two-series BizITObs m8 slice, have low inferential power even when effect sizes look large.",
            "- This is a remote-rerun evidence table, not a direct edit to final paper tables. The lead machine must ingest the `_oral_sidecar_rerun` slugs and rebuild the locked final-main inventory before quoting final-paper aggregates.",
            "",
            f"Machine-readable table: `{CSV_OUT.relative_to(ROOT).as_posix()}`.",
            "",
        ]
    )


def main() -> None:
    plan_rows = [row for row in read_csv(PLAN_PATH) if row.get("priority") == "P0"]
    rows = [summarize_source(row) for row in plan_rows]
    if len(rows) != 17:
        raise ValueError(f"Expected 17 P0 sources, found {len(rows)}")
    if sum(int(row["windows"]) for row in rows) != 408:
        raise ValueError("Expected exactly 408 P0 forecast windows")
    if any(int(row["outer_interval_reversal_points"]) for row in rows):
        raise ValueError("q10-q90 outer interval reversal detected; coverage would be ambiguous")

    qvalues = benjamini_hochberg(
        [float(row["sign_test_pvalue"]) if row["sign_test_pvalue"] else None for row in rows]
    )
    for row, qvalue in zip(rows, qvalues):
        row["sign_test_bh_qvalue"] = fmt(qvalue, digits=8)
    rows.sort(key=row_sort_key)

    write_csv(CSV_OUT, rows)
    DOC_OUT.write_text(build_doc(rows), encoding="utf-8")
    print(
        {
            "status": "ok",
            "sources": len(rows),
            "windows": sum(int(row["windows"]) for row in rows),
            "forecast_points": sum(int(row["forecast_points"]) for row in rows),
            "csv": CSV_OUT.relative_to(ROOT).as_posix(),
            "doc": DOC_OUT.relative_to(ROOT).as_posix(),
        }
    )


if __name__ == "__main__":
    main()
