#!/usr/bin/env python
"""Build factorized failure-family evidence and a calibrated gate audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    prediction_amplitude_ratio,
    relative_error_ratio,
)
from low_snr_tsfm.repair import blended_interval, convex_mixture, hull_interval  # noqa: E402


OUT_DIR = ROOT / "results" / "failure_family"
REPAIR_DIR = ROOT / "results" / "repair"
DOC_PATH = ROOT / "docs" / "factorized_failure_family_report.md"

FEATURE_TABLE = OUT_DIR / "factor_feature_table.csv"
MULTIMETRIC_TABLE = OUT_DIR / "multimetric_failure_table.csv"
TSFM_SYNTHETIC_STATUS = OUT_DIR / "tsfm_synthetic_ablation_status.json"
TSFM_SYNTHETIC_SUMMARY = OUT_DIR / "tsfm_synthetic_ablation_summary.csv"

FACTOR_IDS = [
    "information_insufficiency",
    "weak_reusable_structure",
    "pathological_dynamics",
    "sparse_count_zero_inflation",
    "noise_dominant_heavy_tail",
    "denominator_fragility",
]

EX_ANTE_FACTOR_IDS = FACTOR_IDS[:-1]

FIXED_POLICY = {
    "profile": "fixed_taxonomy",
    "hcr": 0.10,
    "seasonality": 0.15,
    "trend": 0.10,
    "spike": 0.022,
    "change": 0.08,
    "zero": 0.10,
    "cv": 1.50,
    "kurt": 8.0,
    "entropy": 0.85,
    "min_active": 2,
}

CALIBRATION_PROFILES = [
    {
        "profile": "loose",
        "hcr": 0.05,
        "seasonality": 0.20,
        "trend": 0.20,
        "spike": 0.018,
        "change": 0.05,
        "zero": 0.05,
        "cv": 1.00,
        "kurt": 5.0,
        "entropy": 0.80,
    },
    {
        "profile": "balanced",
        "hcr": 0.10,
        "seasonality": 0.15,
        "trend": 0.10,
        "spike": 0.022,
        "change": 0.08,
        "zero": 0.10,
        "cv": 1.50,
        "kurt": 8.0,
        "entropy": 0.85,
    },
    {
        "profile": "strict",
        "hcr": 0.15,
        "seasonality": 0.05,
        "trend": 0.05,
        "spike": 0.030,
        "change": 0.12,
        "zero": 0.20,
        "cv": 3.00,
        "kurt": 12.0,
        "entropy": 0.90,
    },
]

TAXONOMY_ROWS = [
    {
        "layer": "A",
        "factor_id": "information_insufficiency",
        "factor_name": "short context / high horizon-context ratio",
        "plain_language": "The available history is too short relative to the requested forecast path.",
        "operationalization": "horizon_context_ratio >= policy threshold; fixed threshold 0.10.",
        "expected_signature": "Higher MAE-RER and weaker coverage when context_length is reduced.",
        "gate_eligible": 1,
    },
    {
        "layer": "B",
        "factor_id": "weak_reusable_structure",
        "factor_name": "weak seasonality, weak reusable trend, weak autocorrelation proxy",
        "plain_language": "The local history has little repeatable pattern for a pretrained model to reuse.",
        "operationalization": "low seasonality plus low trend, with spectral entropy as a weak-structure proxy.",
        "expected_signature": "Failures concentrate where stable cycles or trends are absent.",
        "gate_eligible": 1,
    },
    {
        "layer": "C",
        "factor_id": "pathological_dynamics",
        "factor_name": "bursts, decays, changepoints, spikes",
        "plain_language": "The future path changes state or jumps in a way not implied by the short context.",
        "operationalization": "spike_frequency or changepoint_density above policy threshold.",
        "expected_signature": "Coverage and shape metrics degrade even when MAE is not the only signal.",
        "gate_eligible": 1,
    },
    {
        "layer": "D",
        "factor_id": "sparse_count_zero_inflation",
        "factor_name": "sparse/count or zero-inflated values",
        "plain_language": "Many near-zero observations make ratios and count-decay dynamics unstable.",
        "operationalization": "zero_ratio above policy threshold.",
        "expected_signature": "Higher denominator sensitivity and discontinuous RER behavior.",
        "gate_eligible": 1,
    },
    {
        "layer": "E",
        "factor_id": "noise_dominant_heavy_tail",
        "factor_name": "high CV, high entropy, heavy tails",
        "plain_language": "Noise dominates reusable signal, so large models can chase non-repeating movement.",
        "operationalization": "coefficient_of_variation, spectral_entropy, or kurtosis_excess above policy threshold.",
        "expected_signature": "More excess movement or unstable shape under low local structure.",
        "gate_eligible": 1,
    },
    {
        "layer": "F",
        "factor_id": "denominator_fragility",
        "factor_name": "baseline denominator fragility",
        "plain_language": "The comparison ratio explodes when the baseline error is extremely small.",
        "operationalization": "baseline_mae < 1e-6 in the realized window-level metric table.",
        "expected_signature": "RER becomes fragile; use as diagnostic evidence, not as an ex-ante gate feature.",
        "gate_eligible": 0,
    },
]

REPAIR_INPUTS = [
    {
        "source": "bizitobs_auto_arima",
        "raw": "results/raw_forecasts/chronos_bolt_small_bizitobs_application_short_auto_arima.csv",
        "features": "results/failure_mining/chronos_bolt_small_bizitobs_application_short_auto_arima_predictor_features.csv",
        "role": "failure_target",
    },
    {
        "source": "chronos_covid_auto_ets",
        "raw": "results/raw_forecasts/chronos_bolt_small_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/chronos_bolt_small_covid_deaths_short_auto_ets_predictor_features.csv",
        "role": "failure_target",
    },
    {
        "source": "chronos_finance_fred",
        "raw": "results/raw_forecasts/chronos_bolt_small_finance_fred_stress.csv",
        "features": "results/failure_mining/chronos_bolt_small_finance_fred_stress_predictor_features.csv",
        "role": "stress_target",
    },
    {
        "source": "chronos_solar_seasonal_naive",
        "raw": "results/raw_forecasts/chronos_bolt_small_solar_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_solar_short_seasonal_naive_predictor_features.csv",
        "role": "positive_control",
    },
    {
        "source": "chronos_loop_seattle",
        "raw": "results/raw_forecasts/chronos_bolt_small_loop_seattle_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_loop_seattle_short_seasonal_naive_predictor_features.csv",
        "role": "positive_control",
    },
]


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def source_role(source: str) -> str:
    if "solar" in source or "loop_seattle" in source:
        return "positive_control"
    if "finance" in source:
        return "stress_target"
    if "traffic" in source:
        return "traffic_transition"
    return "failure_target"


def feature_lookup_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("source", ""),
        row.get("dataset", ""),
        row.get("series_id", ""),
        str(row.get("window_index", "")),
    )


def dataset_window_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("series_id", ""),
        str(row.get("window_index", "")),
    )


def infer_source_for_multimetric(row: dict[str, str]) -> str:
    dataset = row.get("dataset", "")
    model = row.get("model", "")
    if dataset.startswith("bizitobs"):
        return "bizitobs_auto_arima"
    if dataset.startswith("FRED"):
        return "chronos_finance_fred"
    if dataset.startswith("loop_seattle"):
        return "chronos_loop_seattle"
    if dataset.startswith("metr-la"):
        return "metr_la_traffic"
    if dataset.startswith("pems-bay"):
        return "pems_bay_traffic"
    if dataset.startswith("solar"):
        return "moirai_solar_seasonal_naive" if model.startswith("moirai") else "chronos_solar_seasonal_naive"
    if dataset.startswith("covid"):
        if model.startswith("timesfm"):
            return "timesfm_covid_auto_ets"
        if model.startswith("moirai"):
            return "moirai_covid_auto_ets"
        return "chronos_covid_auto_ets"
    return dataset


def denominator_map(multimetric_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], int]:
    mapped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in multimetric_rows:
        key = dataset_window_key(row)
        mapped[key] = max(mapped[key], int(finite_float(row.get("denominator_fragile")) > 0.5))
    return dict(mapped)


def flags_for(
    row: dict[str, object],
    denominator_fragile: int = 0,
    policy: dict[str, object] | None = None,
) -> dict[str, bool]:
    selected = policy or FIXED_POLICY
    hcr = finite_float(row.get("horizon_context_ratio"))
    season = finite_float(row.get("seasonality_strength"))
    trend = finite_float(row.get("trend_strength"))
    spike = finite_float(row.get("spike_frequency"))
    zero = finite_float(row.get("zero_ratio"))
    change = finite_float(row.get("changepoint_density"))
    cv = finite_float(row.get("coefficient_of_variation"))
    entropy = finite_float(row.get("spectral_entropy"))
    kurt = finite_float(row.get("kurtosis_excess"))

    weak_structure = (season <= finite_float(selected["seasonality"]) and trend <= finite_float(selected["trend"])) or (
        season <= finite_float(selected["seasonality"]) / 2.0 and entropy >= finite_float(selected["entropy"])
    )
    noise_dominant = (
        cv >= finite_float(selected["cv"])
        or kurt >= finite_float(selected["kurt"])
        or (entropy >= finite_float(selected["entropy"]) and season <= finite_float(selected["seasonality"]))
    )
    return {
        "information_insufficiency": hcr >= finite_float(selected["hcr"]),
        "weak_reusable_structure": weak_structure,
        "pathological_dynamics": spike >= finite_float(selected["spike"]) or change >= finite_float(selected["change"]),
        "sparse_count_zero_inflation": zero >= finite_float(selected["zero"]),
        "noise_dominant_heavy_tail": noise_dominant,
        "denominator_fragility": bool(denominator_fragile),
    }


def active_factors(flags: dict[str, bool], include_denominator: bool = True) -> list[str]:
    allowed = FACTOR_IDS if include_denominator else EX_ANTE_FACTOR_IDS
    return [factor for factor in allowed if flags.get(factor)]


def combo_label(factors: list[str]) -> str:
    return "+".join(factors) if factors else "none"


def load_feature_rows() -> list[dict[str, object]]:
    if not FEATURE_TABLE.exists():
        raise SystemExit(f"Missing prerequisite: {FEATURE_TABLE.relative_to(ROOT)}")
    rows: list[dict[str, object]] = []
    for row in read_csv(FEATURE_TABLE):
        normalized: dict[str, object] = dict(row)
        normalized["role"] = source_role(row.get("source", ""))
        normalized["failure_delta_005"] = int(finite_float(row.get("failure_delta_005")) > 0.5)
        normalized["relative_error_ratio"] = finite_float(row.get("relative_error_ratio"))
        normalized["empirical_coverage_90"] = optional_float(row.get("empirical_coverage_90"))
        normalized["flatness_score"] = optional_float(row.get("flatness_score"))
        rows.append(normalized)
    return rows


def enrich_feature_rows(feature_rows: list[dict[str, object]], denom_by_window: dict[tuple[str, str, str], int]) -> list[dict[str, object]]:
    enriched = []
    for row in feature_rows:
        denom = denom_by_window.get((str(row.get("dataset", "")), str(row.get("series_id", "")), str(row.get("window_index", ""))), 0)
        flags = flags_for(row, denominator_fragile=denom)
        ex_ante = active_factors(flags, include_denominator=False)
        all_factors = active_factors(flags, include_denominator=True)
        out = dict(row)
        for factor in FACTOR_IDS:
            out[factor] = int(flags[factor])
        out["active_ex_ante_factor_count"] = len(ex_ante)
        out["active_factor_count"] = len(all_factors)
        out["factor_combo"] = combo_label(all_factors)
        out["ex_ante_factor_combo"] = combo_label(ex_ante)
        out["factorized_regime"] = int(len(ex_ante) >= 2)
        enriched.append(out)
    return enriched


def summarize_feature_group(group_name: str, rows: list[dict[str, object]], extra: dict[str, object] | None = None) -> dict[str, object]:
    coverage_values = [
        float(row["empirical_coverage_90"])
        for row in rows
        if row.get("empirical_coverage_90") is not None and math.isfinite(float(row["empirical_coverage_90"]))
    ]
    flatness_values = [
        float(row["flatness_score"])
        for row in rows
        if row.get("flatness_score") is not None and math.isfinite(float(row["flatness_score"]))
    ]
    output = {
        "group": group_name,
        "n_windows": len(rows),
        "n_sources": len({str(row.get("source", "")) for row in rows}),
        "sources": ",".join(sorted({str(row.get("source", "")) for row in rows})),
        "failure_rate_delta_005": rate([int(row["failure_delta_005"]) for row in rows]),
        "median_relative_error_ratio": median([float(row["relative_error_ratio"]) for row in rows]),
        "mean_relative_error_ratio": mean([float(row["relative_error_ratio"]) for row in rows]),
        "mean_empirical_coverage_90": mean(coverage_values),
        "bad_coverage_rate_lt_070": rate([int(value < 0.70) for value in coverage_values]),
        "mean_flatness_score": mean(flatness_values),
        "over_smoothing_rate_flatness_ge_060": rate([int(value >= 0.60) for value in flatness_values]),
        "positive_control_share": rate([int(row.get("role") == "positive_control") for row in rows]),
        "denominator_fragile_rate": rate([int(row.get("denominator_fragility", 0)) for row in rows]),
    }
    if extra:
        output.update(extra)
    return output


def build_factor_interactions(enriched_features: list[dict[str, object]], out_dir: Path) -> dict[str, object]:
    write_csv(out_dir / "factor_taxonomy.csv", TAXONOMY_ROWS)
    write_csv(out_dir / "factorized_feature_table.csv", enriched_features)

    combo_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in enriched_features:
        combo_groups[str(row["factor_combo"])].append(row)
    combo_rows = []
    for combo, rows in combo_groups.items():
        factors = [] if combo == "none" else combo.split("+")
        combo_rows.append(
            summarize_feature_group(
                combo,
                rows,
                {
                    "active_factor_count": len(factors),
                    "active_factors": combo,
                    "interaction_type": "specific_combo",
                },
            )
        )
    combo_rows.sort(key=lambda row: (-int(row["active_factor_count"]), -float(row["failure_rate_delta_005"]), str(row["group"])))
    write_csv(out_dir / "factor_interaction_table.csv", combo_rows)

    ladder_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in enriched_features:
        ladder_groups[int(row["active_ex_ante_factor_count"])].append(row)
    ladder_rows = [
        summarize_feature_group(
            f"{count}_ex_ante_factors",
            rows,
            {"active_ex_ante_factor_count": count, "interaction_type": "active_factor_count"},
        )
        for count, rows in sorted(ladder_groups.items())
    ]
    write_csv(out_dir / "factor_interaction_ladder.csv", ladder_rows)

    return {
        "taxonomy": str((out_dir / "factor_taxonomy.csv").relative_to(ROOT)),
        "factorized_feature_table": str((out_dir / "factorized_feature_table.csv").relative_to(ROOT)),
        "combo_table": str((out_dir / "factor_interaction_table.csv").relative_to(ROOT)),
        "ladder_table": str((out_dir / "factor_interaction_ladder.csv").relative_to(ROOT)),
        "n_feature_rows": len(enriched_features),
        "n_combos": len(combo_rows),
        "n_ladder_rows": len(ladder_rows),
    }


def feature_lookup(enriched_features: list[dict[str, object]]) -> tuple[dict[tuple[str, str, str, str], dict[str, object]], dict[tuple[str, str, str], dict[str, object]]]:
    by_source = {}
    by_dataset = {}
    for row in enriched_features:
        source_key = (
            str(row.get("source", "")),
            str(row.get("dataset", "")),
            str(row.get("series_id", "")),
            str(row.get("window_index", "")),
        )
        dataset_key = (
            str(row.get("dataset", "")),
            str(row.get("series_id", "")),
            str(row.get("window_index", "")),
        )
        by_source[source_key] = row
        by_dataset.setdefault(dataset_key, row)
    return by_source, by_dataset


def enrich_multimetric_rows(multimetric_rows: list[dict[str, str]], enriched_features: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source, by_dataset = feature_lookup(enriched_features)
    enriched = []
    for row in multimetric_rows:
        source = infer_source_for_multimetric(row)
        source_key = (
            source,
            row.get("dataset", ""),
            row.get("series_id", ""),
            str(row.get("window_index", "")),
        )
        dataset_key = dataset_window_key(row)
        feature = by_source.get(source_key) or by_dataset.get(dataset_key, {})
        out: dict[str, object] = dict(row)
        out["source"] = source
        out["role"] = source_role(source)
        out["active_ex_ante_factor_count"] = int(feature.get("active_ex_ante_factor_count", 0))
        out["active_factor_count"] = int(feature.get("active_factor_count", 0))
        out["factor_combo"] = str(feature.get("factor_combo", "none"))
        out["factorized_regime"] = int(int(feature.get("active_ex_ante_factor_count", 0)) >= 2)
        for factor in FACTOR_IDS:
            out[factor] = int(feature.get(factor, 0))
        enriched.append(out)
    return enriched


def metric_failure(row: dict[str, object], key: str, threshold: float = 1.05) -> int | None:
    value = optional_float(row.get(key))
    if value is None:
        return None
    return int(value > threshold)


def summarize_multimetric_group(group: str, rows: list[dict[str, object]]) -> dict[str, object]:
    mae_failures = [int(finite_float(row.get("relative_error_ratio")) > 1.05) for row in rows]
    rmse_failures = [
        metric_failure(row, "rmse_relative_error_ratio")
        for row in rows
        if metric_failure(row, "rmse_relative_error_ratio") is not None
    ]
    mase_failures = [
        metric_failure(row, "mase_relative_error_ratio")
        for row in rows
        if metric_failure(row, "mase_relative_error_ratio") is not None
    ]
    coverage_values = [
        float(row["empirical_coverage_90"])
        for row in rows
        if optional_float(row.get("empirical_coverage_90")) is not None
    ]
    fvr_values = [finite_float(row.get("forecast_variance_ratio")) for row in rows]
    par_values = [finite_float(row.get("prediction_amplitude_ratio")) for row in rows]
    flat_values = [finite_float(row.get("flatness_score")) for row in rows]
    spike_values = [finite_float(row.get("spike_recall")) for row in rows]
    shape_bad = [
        int(fvr > 2.0 or par > 1.5 or flat > 0.60 or spike < 0.34)
        for fvr, par, flat, spike in zip(fvr_values, par_values, flat_values, spike_values)
    ]
    return {
        "group": group,
        "n_windows": len(rows),
        "mae_rer_failure_rate": rate(mae_failures),
        "median_mae_rer": median([finite_float(row.get("relative_error_ratio")) for row in rows]),
        "rmse_rer_available_n": len(rmse_failures),
        "rmse_rer_failure_rate": rate([int(value) for value in rmse_failures]),
        "median_rmse_rer": median([finite_float(row.get("rmse_relative_error_ratio")) for row in rows]),
        "mase_rer_available_n": len(mase_failures),
        "mase_rer_failure_rate": rate([int(value) for value in mase_failures]),
        "median_mase_rer": median(
            [
                finite_float(row.get("mase_relative_error_ratio"))
                for row in rows
                if optional_float(row.get("mase_relative_error_ratio")) is not None
            ]
        ),
        "coverage_bad_rate_lt_070": rate([int(value < 0.70) for value in coverage_values]),
        "mean_empirical_coverage_90": mean(coverage_values),
        "shape_bad_rate": rate(shape_bad),
        "excess_variance_rate_fvr_gt_2": rate([int(value > 2.0) for value in fvr_values]),
        "over_smoothing_rate_flatness_ge_060": rate([int(value >= 0.60) for value in flat_values]),
        "denominator_fragile_rate": rate([int(finite_float(row.get("denominator_fragile")) > 0.5) for row in rows]),
        "positive_control_share": rate([int(row.get("role") == "positive_control") for row in rows]),
    }


def build_multimetric_robustness(enriched_multi: list[dict[str, object]], out_dir: Path) -> dict[str, object]:
    groups = {
        "all_windows": enriched_multi,
        "factorized_regime_n_ge_2": [row for row in enriched_multi if int(row.get("factorized_regime", 0)) == 1],
        "outside_factorized_regime": [row for row in enriched_multi if int(row.get("factorized_regime", 0)) == 0],
        "positive_controls": [row for row in enriched_multi if row.get("role") == "positive_control"],
        "noncontrol_failure_or_stress": [row for row in enriched_multi if row.get("role") != "positive_control"],
        "denominator_fragile": [row for row in enriched_multi if int(finite_float(row.get("denominator_fragile"))) == 1],
    }
    rows = [summarize_multimetric_group(group, subset) for group, subset in groups.items() if subset]
    path = out_dir / "multimetric_robustness_summary.csv"
    write_csv(path, rows)
    enriched_path = out_dir / "factorized_multimetric_table.csv"
    write_csv(enriched_path, enriched_multi)
    return {
        "summary": str(path.relative_to(ROOT)),
        "factorized_multimetric_table": str(enriched_path.relative_to(ROOT)),
        "n_groups": len(rows),
        "n_multimetric_rows": len(enriched_multi),
    }


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    positives = [idx for idx, label in enumerate(labels) if label == 1]
    negatives = [idx for idx, label in enumerate(labels) if label == 0]
    tpr = sum(predictions[idx] == 1 for idx in positives) / len(positives) if positives else 0.5
    tnr = sum(predictions[idx] == 0 for idx in negatives) / len(negatives) if negatives else 0.5
    return float((tpr + tnr) / 2.0)


def candidate_policies() -> list[dict[str, object]]:
    candidates = []
    for profile in CALIBRATION_PROFILES:
        for min_active in [2, 3]:
            candidate = dict(profile)
            candidate["min_active"] = min_active
            candidates.append(candidate)
    return candidates


def predict_gate(row: dict[str, object], policy: dict[str, object]) -> tuple[int, str, int]:
    flags = flags_for(row, denominator_fragile=0, policy=policy)
    factors = active_factors(flags, include_denominator=False)
    return int(len(factors) >= int(policy["min_active"])), combo_label(factors), len(factors)


def evaluate_policy(rows: list[dict[str, object]], policy: dict[str, object]) -> dict[str, float]:
    labels = [int(row["failure_delta_005"]) for row in rows]
    predictions = [predict_gate(row, policy)[0] for row in rows]
    positives = sum(labels)
    predicted = sum(predictions)
    true_positive = sum(int(label == 1 and pred == 1) for label, pred in zip(labels, predictions))
    false_positive = sum(int(label == 0 and pred == 1) for label, pred in zip(labels, predictions))
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / positives if positives else 0.0
    positive_control_rows = [idx for idx, row in enumerate(rows) if row.get("role") == "positive_control"]
    positive_control_gate_rate = (
        float(np.mean([predictions[idx] for idx in positive_control_rows])) if positive_control_rows else 0.0
    )
    return {
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "gate_rate": predicted / len(rows) if rows else 0.0,
        "failure_rate": positives / len(rows) if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "positive_control_gate_rate": positive_control_gate_rate,
        "false_positive_rate": false_positive / max(len(labels) - positives, 1),
    }


def select_policy(train_rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, float]]:
    best_policy: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -1e9
    for policy in candidate_policies():
        metrics = evaluate_policy(train_rows, policy)
        # The gate is used for repair, so recall matters more than a very low
        # intervention rate. Positive-control safety is audited downstream as
        # "does not worsen controls" rather than "never opens the gate".
        score = (
            0.30 * metrics["balanced_accuracy"]
            + 0.45 * metrics["recall"]
            + 0.20 * metrics["precision"]
            - 0.03 * metrics["positive_control_gate_rate"]
            - 0.02 * metrics["gate_rate"]
        )
        if score > best_score:
            best_score = score
            best_policy = policy
            best_metrics = metrics
    assert best_policy is not None and best_metrics is not None
    return dict(best_policy), dict(best_metrics)


def build_validation_calibrated_gate(enriched_features: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    sources = sorted({str(row.get("source", "")) for row in enriched_features})
    heldout_rows: list[dict[str, object]] = []
    policy_by_source: dict[str, dict[str, object]] = {}
    for source in sources:
        train = [row for row in enriched_features if row.get("source") != source]
        test = [row for row in enriched_features if row.get("source") == source]
        if not train or not test:
            continue
        policy, train_metrics = select_policy(train)
        test_metrics = evaluate_policy(test, policy)
        policy_by_source[source] = policy
        heldout_rows.append(
            {
                "heldout_source": source,
                "role": source_role(source),
                "train_n": len(train),
                "test_n": len(test),
                "selected_profile": policy["profile"],
                "selected_min_active": policy["min_active"],
                "threshold_hcr": policy["hcr"],
                "threshold_seasonality": policy["seasonality"],
                "threshold_trend": policy["trend"],
                "threshold_spike": policy["spike"],
                "threshold_changepoint": policy["change"],
                "threshold_zero": policy["zero"],
                "threshold_cv": policy["cv"],
                "threshold_kurtosis": policy["kurt"],
                "threshold_entropy": policy["entropy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_gate_rate": train_metrics["gate_rate"],
                "train_failure_rate": train_metrics["failure_rate"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_gate_rate": test_metrics["gate_rate"],
                "test_failure_rate": test_metrics["failure_rate"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_positive_control_gate_rate": test_metrics["positive_control_gate_rate"],
            }
        )
    path = out_dir / "validation_calibrated_gate_classification.csv"
    write_csv(path, heldout_rows)
    summary = {
        "classification": str(path.relative_to(ROOT)),
        "n_heldout_sources": len(heldout_rows),
        "mean_test_balanced_accuracy": mean([float(row["test_balanced_accuracy"]) for row in heldout_rows]),
        "median_test_balanced_accuracy": median([float(row["test_balanced_accuracy"]) for row in heldout_rows]),
        "mean_test_gate_rate": mean([float(row["test_gate_rate"]) for row in heldout_rows]),
        "mean_test_recall": mean([float(row["test_recall"]) for row in heldout_rows]),
        "mean_positive_control_gate_rate": mean(
            [float(row["test_gate_rate"]) for row in heldout_rows if row["role"] == "positive_control"]
        ),
    }
    return policy_by_source, summary


def raw_window_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("series_id", ""),
        row.get("origin", ""),
        str(row.get("window_index", "")),
    )


def feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("dataset", ""), row.get("series_id", ""), str(row.get("window_index", "")))


def raw_window_map(raw_path: Path) -> dict[tuple[str, str, str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(raw_path):
        groups[raw_window_key(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: finite_float(row.get("horizon_index")))
    return dict(groups)


def raw_baseline_values(rows: list[dict[str, str]]) -> np.ndarray:
    if rows and "baseline_forecast" in rows[0]:
        return np.asarray([finite_float(row.get("baseline_forecast")) for row in rows], dtype=float)
    if rows and "historical_mean" in rows[0]:
        return np.asarray([finite_float(row.get("historical_mean")) for row in rows], dtype=float)
    if rows and "bma_mean" in rows[0]:
        return np.asarray([finite_float(row.get("bma_mean")) for row in rows], dtype=float)
    raise ValueError("Raw rows do not contain a recognized baseline forecast column")


def summarize_repair(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    return {
        "group": group,
        "n_windows": len(rows),
        "gate_rate": rate([int(row["gate_active"]) for row in rows]),
        "model_failure_rate_delta_005": rate([int(row["model_failure_delta_005"]) for row in rows]),
        "repair_failure_rate_delta_005": rate([int(row["repair_failure_delta_005"]) for row in rows]),
        "model_median_relative_error_ratio": median([float(row["model_relative_error_ratio"]) for row in rows]),
        "repair_median_relative_error_ratio": median([float(row["repair_relative_error_ratio"]) for row in rows]),
        "repair_win_rate_vs_model": rate([int(row["repair_improves_model"]) for row in rows]),
        "model_mean_empirical_coverage_90": mean([float(row["model_empirical_coverage_90"]) for row in rows]),
        "repair_blend_mean_empirical_coverage_90": mean([float(row["repair_blend_empirical_coverage_90"]) for row in rows]),
        "repair_hull_mean_empirical_coverage_90": mean([float(row["repair_hull_empirical_coverage_90"]) for row in rows]),
    }


def run_heldout_calibrated_repair(policy_by_source: dict[str, dict[str, object]], out_dir: Path) -> dict[str, object]:
    metric_rows: list[dict[str, object]] = []
    for repair_input in REPAIR_INPUTS:
        source = repair_input["source"]
        policy = policy_by_source.get(source)
        if policy is None:
            continue
        raw_path = ROOT / str(repair_input["raw"])
        feature_path = ROOT / str(repair_input["features"])
        if not raw_path.exists() or not feature_path.exists():
            continue
        raw_groups = raw_window_map(raw_path)
        feature_rows = {feature_key(row): row for row in read_csv(feature_path)}
        for key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(("", series_id, window_index), {})
            gate_active, gate_reason, active_count = predict_gate(feature, policy)
            weight = 0.75 if gate_active else 0.0
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            repaired = convex_mixture(model, baseline, weight)
            blend_q10, blend_q90 = blended_interval(q10, q90, baseline, weight)
            hull_q10, hull_q90 = hull_interval(q10, q90, baseline)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            repair_mae = mae(actual, repaired)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            repair_rer = relative_error_ratio(repair_mae, baseline_mae)
            metric_rows.append(
                {
                    "heldout_source": source,
                    "role": repair_input["role"],
                    "dataset": dataset,
                    "model": model_name,
                    "series_id": series_id,
                    "origin": origin,
                    "window_index": window_index,
                    "selected_profile": policy["profile"],
                    "selected_min_active": policy["min_active"],
                    "active_ex_ante_factor_count": active_count,
                    "gate_active": gate_active,
                    "gate_reason": gate_reason,
                    "gate_weight": weight,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "repair_mae": repair_mae,
                    "model_relative_error_ratio": model_rer,
                    "repair_relative_error_ratio": repair_rer,
                    "model_failure_delta_005": int(model_rer > 1.05),
                    "repair_failure_delta_005": int(repair_rer > 1.05),
                    "repair_improves_model": int(repair_mae < model_mae),
                    "repair_beats_baseline": int(repair_mae < baseline_mae),
                    "model_empirical_coverage_90": empirical_coverage(actual, q10, q90),
                    "repair_blend_empirical_coverage_90": empirical_coverage(actual, blend_q10, blend_q90),
                    "repair_hull_empirical_coverage_90": empirical_coverage(actual, hull_q10, hull_q90),
                    "model_forecast_variance_ratio": forecast_variance_ratio(actual, model),
                    "repair_forecast_variance_ratio": forecast_variance_ratio(actual, repaired),
                    "model_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, model),
                    "repair_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, repaired),
                    "model_flatness_score": flatness_score(actual, model),
                    "repair_flatness_score": flatness_score(actual, repaired),
                }
            )
    metrics_path = out_dir / "validation_calibrated_gate_repair_metrics.csv"
    write_csv(metrics_path, metric_rows)
    summary_rows = []
    for role in sorted({str(row["role"]) for row in metric_rows}):
        summary_rows.append(summarize_repair([row for row in metric_rows if row["role"] == role], role))
    for source in sorted({str(row["heldout_source"]) for row in metric_rows}):
        summary_rows.append(summarize_repair([row for row in metric_rows if row["heldout_source"] == source], f"source:{source}"))
    summary_rows.append(summarize_repair(metric_rows, "overall"))
    summary_path = out_dir / "validation_calibrated_gate_repair_summary.csv"
    write_csv(summary_path, summary_rows)
    return {
        "metrics": str(metrics_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "n_windows": len(metric_rows),
    }


def load_tsfm_synthetic_status() -> dict[str, object]:
    if not TSFM_SYNTHETIC_STATUS.exists():
        return {
            "status": "missing",
            "message": "Run .venv-chronos/bin/python scripts/run_chronos_synthetic_factor_ablation.py --seeds 2",
        }
    return json.loads(TSFM_SYNTHETIC_STATUS.read_text())


def format_float(value: object) -> str:
    parsed = optional_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.3f}"


def markdown_table(rows: list[dict[str, object]], columns: list[str], max_rows: int | None = None) -> str:
    selected = rows[:max_rows] if max_rows is not None else rows
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in selected:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    *,
    interaction_ladder: list[dict[str, str]],
    robustness_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    repair_summary_rows: list[dict[str, str]],
    tsfm_status: dict[str, object],
) -> None:
    taxonomy_for_md = [
        {
            "Layer": row["layer"],
            "Factor": row["factor_id"],
            "Gate": row["gate_eligible"],
            "Operationalization": row["operationalization"],
        }
        for row in TAXONOMY_ROWS
    ]
    ladder_for_md = [
        {
            "Group": row["group"],
            "N": row["n_windows"],
            "Failure": format_float(row["failure_rate_delta_005"]),
            "Median RER": format_float(row["median_relative_error_ratio"]),
            "Coverage<0.70": format_float(row["bad_coverage_rate_lt_070"]),
            "Flat>=0.60": format_float(row["over_smoothing_rate_flatness_ge_060"]),
        }
        for row in interaction_ladder
    ]
    robustness_for_md = [
        {
            "Group": row["group"],
            "N": row["n_windows"],
            "MAE fail": format_float(row["mae_rer_failure_rate"]),
            "RMSE fail": format_float(row["rmse_rer_failure_rate"]),
            "MASE fail": format_float(row["mase_rer_failure_rate"]),
            "Coverage<0.70": format_float(row["coverage_bad_rate_lt_070"]),
            "Shape bad": format_float(row["shape_bad_rate"]),
        }
        for row in robustness_rows
    ]
    gate_for_md = [
        {
            "Heldout": row["heldout_source"],
            "Role": row["role"],
            "Profile": row["selected_profile"],
            "Min": row["selected_min_active"],
            "Test BA": format_float(row["test_balanced_accuracy"]),
            "Gate": format_float(row["test_gate_rate"]),
            "Recall": format_float(row["test_recall"]),
        }
        for row in gate_rows
    ]
    repair_for_md = [
        {
            "Group": row["group"],
            "N": row["n_windows"],
            "Gate": format_float(row["gate_rate"]),
            "Model fail": format_float(row["model_failure_rate_delta_005"]),
            "Repair fail": format_float(row["repair_failure_rate_delta_005"]),
            "Repair win": format_float(row["repair_win_rate_vs_model"]),
        }
        for row in repair_summary_rows
        if row["group"] in {"failure_target", "stress_target", "positive_control", "overall"}
    ]
    mean_gate_ba = mean([finite_float(row["test_balanced_accuracy"]) for row in gate_rows])
    mean_gate_recall = mean([finite_float(row["test_recall"]) for row in gate_rows])
    positive_control_summary = next((row for row in repair_summary_rows if row["group"] == "positive_control"), {})
    overall_summary = next((row for row in repair_summary_rows if row["group"] == "overall"), {})
    positive_control_note = (
        "Positive-control failure rate changes from "
        f"{format_float(positive_control_summary.get('model_failure_rate_delta_005'))} to "
        f"{format_float(positive_control_summary.get('repair_failure_rate_delta_005'))}; "
        "however, controls are still frequently gated, so this is a safety pass on failure rate, not a finished selective gate."
    )
    overall_note = (
        "Overall held-out repair failure rate changes from "
        f"{format_float(overall_summary.get('model_failure_rate_delta_005'))} to "
        f"{format_float(overall_summary.get('repair_failure_rate_delta_005'))}."
    )

    tsfm_block = f"Status: `{tsfm_status.get('status', 'missing')}`."
    if tsfm_status.get("status") == "ok" and TSFM_SYNTHETIC_SUMMARY.exists():
        synth_rows = read_csv(TSFM_SYNTHETIC_SUMMARY)
        synth_md_rows = [
            {
                "Factor": row["factor"],
                "Value": row["value"],
                "N": row["n_windows"],
                "Failure": format_float(row["failure_rate_delta_005"]),
                "Median RER": format_float(row["median_relative_error_ratio"]),
                "Coverage": format_float(row["mean_empirical_coverage_90"]),
            }
            for row in synth_rows
        ]
        tsfm_block += "\n\n" + markdown_table(
            synth_md_rows,
            ["Factor", "Value", "N", "Failure", "Median RER", "Coverage"],
            max_rows=12,
        )
    else:
        tsfm_block += f" Repro: `{tsfm_status.get('repro_command') or tsfm_status.get('message', '')}`."

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Factorized Failure Family Report",
                "",
                "## Claim",
                "",
                "This goal reframes the prior low-local-structure result as a failure regime, not a single root cause. The regime is defined by overlapping evidence of information insufficiency, weak reusable structure, pathological dynamics, sparse/count behavior, noise-dominant or heavy-tailed local history, and denominator-fragile evaluation.",
                "",
                "The shortest paper path is: figure case -> failure taxonomy -> factor interaction evidence -> controlled ablation -> validation-calibrated gated repair -> positive-control safety.",
                "",
                "## Six-Factor Taxonomy",
                "",
                markdown_table(taxonomy_for_md, ["Layer", "Factor", "Gate", "Operationalization"]),
                "",
                "Important design choice: denominator fragility is included in the taxonomy and interaction audit, but it is not used as an ex-ante gate feature because it depends on realized baseline error.",
                "",
                "## Factor Interaction Evidence",
                "",
                markdown_table(ladder_for_md, ["Group", "N", "Failure", "Median RER", "Coverage<0.70", "Flat>=0.60"]),
                "",
                "Full combination table: `results/failure_family/factor_interaction_table.csv`.",
                "",
                "## Multi-Metric Robustness",
                "",
                markdown_table(robustness_for_md, ["Group", "N", "MAE fail", "RMSE fail", "MASE fail", "Coverage<0.70", "Shape bad"]),
                "",
                "The main failure-rate entry remains MAE-RER, but the same family is audited under RMSE-RER, available MASE-RER, coverage, excess-variance, over-smoothing, and spike/shape signals.",
                "",
                "## TSFM-on-Synthetic Controlled Ablation",
                "",
                tsfm_block,
                "",
                "This is intentionally a compact Chronos-Bolt tiny zero-shot ablation. It tests whether controlled changes in context length, seasonality, spikes, or decay produce compatible signatures in a real TSFM runner rather than only in the proxy simulator.",
                "",
                "## Validation-Calibrated Gate",
                "",
                markdown_table(gate_for_md, ["Heldout", "Role", "Profile", "Min", "Test BA", "Gate", "Recall"], max_rows=12),
                "",
                "The gate thresholds are selected leave-one-source-out from a small policy family, with at least two ex-ante factors required before intervention. This avoids using the held-out dataset or model-family slice when choosing the gate and keeps the method aligned with the regime definition.",
                "",
                f"Interpretation: this is not yet a strong cross-domain classifier (mean held-out balanced accuracy {mean_gate_ba:.3f}), but it is a high-recall repair trigger (mean held-out recall {mean_gate_recall:.3f}). That distinction should be explicit in the paper framing.",
                "",
                "## Held-Out Gated Repair",
                "",
                markdown_table(repair_for_md, ["Group", "N", "Gate", "Model fail", "Repair fail", "Repair win"]),
                "",
                "The held-out repair uses the calibrated gate and a fixed conservative baseline-mixture weight. Positive controls are reported explicitly as a safety check.",
                "",
                overall_note,
                "",
                positive_control_note,
                "",
                "## Literature Positioning",
                "",
                "Using the literature map supplied in the project discussion, this report positions the contribution between benchmark-hidden regime failures, forecast-collapse theory, financial low-signal stress tests, and calibration audits. The novelty claim should remain modest: this is a cross-domain factorized failure family with controlled ablation and a gated repair, not proof that all TSFMs fail for one universal cause.",
                "",
                "## Artifacts",
                "",
                "- `results/failure_family/factor_taxonomy.csv`",
                "- `results/failure_family/factor_interaction_table.csv`",
                "- `results/failure_family/factor_interaction_ladder.csv`",
                "- `results/failure_family/multimetric_robustness_summary.csv`",
                "- `results/failure_family/tsfm_synthetic_ablation_status.json`",
                "- `results/failure_family/validation_calibrated_gate_classification.csv`",
                "- `results/repair/validation_calibrated_gate_repair_summary.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/failure_family")
    parser.add_argument("--repair-dir", default="results/repair")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    repair_dir = ROOT / args.repair_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)

    if not MULTIMETRIC_TABLE.exists():
        raise SystemExit(f"Missing prerequisite: {MULTIMETRIC_TABLE.relative_to(ROOT)}")
    multimetric_rows = read_csv(MULTIMETRIC_TABLE)
    features = load_feature_rows()
    enriched_features = enrich_feature_rows(features, denominator_map(multimetric_rows))
    interaction_report = build_factor_interactions(enriched_features, out_dir)
    enriched_multi = enrich_multimetric_rows(multimetric_rows, enriched_features)
    robustness_report = build_multimetric_robustness(enriched_multi, out_dir)
    policy_by_source, gate_report = build_validation_calibrated_gate(enriched_features, out_dir)
    repair_report = run_heldout_calibrated_repair(policy_by_source, repair_dir)
    tsfm_status = load_tsfm_synthetic_status()

    interaction_ladder = read_csv(out_dir / "factor_interaction_ladder.csv")
    robustness_rows = read_csv(out_dir / "multimetric_robustness_summary.csv")
    gate_rows = read_csv(out_dir / "validation_calibrated_gate_classification.csv")
    repair_summary_rows = read_csv(repair_dir / "validation_calibrated_gate_repair_summary.csv")
    write_report(
        interaction_ladder=interaction_ladder,
        robustness_rows=robustness_rows,
        gate_rows=gate_rows,
        repair_summary_rows=repair_summary_rows,
        tsfm_status=tsfm_status,
    )

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "taxonomy": str((out_dir / "factor_taxonomy.csv").relative_to(ROOT)),
        "interaction": interaction_report,
        "multimetric_robustness": robustness_report,
        "validation_calibrated_gate": gate_report,
        "heldout_repair": repair_report,
        "tsfm_synthetic": tsfm_status,
        "report": str(DOC_PATH.relative_to(ROOT)),
        "required_outputs": [
            str((out_dir / "factor_taxonomy.csv").relative_to(ROOT)),
            str((out_dir / "factor_interaction_table.csv").relative_to(ROOT)),
            str((out_dir / "factor_interaction_ladder.csv").relative_to(ROOT)),
            str((out_dir / "multimetric_robustness_summary.csv").relative_to(ROOT)),
            str((out_dir / "validation_calibrated_gate_classification.csv").relative_to(ROOT)),
            str((repair_dir / "validation_calibrated_gate_repair_summary.csv").relative_to(ROOT)),
            str(DOC_PATH.relative_to(ROOT)),
        ],
    }
    status_path = out_dir / "factorized_failure_family_goal_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
