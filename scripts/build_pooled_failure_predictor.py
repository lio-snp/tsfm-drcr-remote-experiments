#!/usr/bin/env python
"""Pool context-only failure predictors across generated regimes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.predictor import (
    apply_threshold,
    balanced_accuracy,
    best_threshold,
    evaluate_threshold,
    fit_depth2_threshold_tree,
    fit_balanced_logistic_regression,
    majority_label,
    predict_depth2_threshold_tree,
    predict_logistic_regression,
)


OUT_DIR = ROOT / "results" / "failure_mining"
FEATURE_NAMES = [
    "autocorrelation_strength",
    "changepoint_density",
    "coefficient_of_variation",
    "horizon_context_ratio",
    "kurtosis_excess",
    "missingness",
    "seasonality_strength",
    "spectral_entropy",
    "spike_frequency",
    "trend_strength",
    "zero_ratio",
]
DEFAULT_INPUTS = [
    (
        "results/failure_mining/chronos_bolt_small_bizitobs_application_short_auto_arima_predictor_features.csv",
        "Web/CloudOps",
        "intermittent_bursty",
        "bizitobs_auto_arima",
    ),
    (
        "results/failure_mining/chronos_bolt_small_covid_deaths_short_auto_ets_predictor_features.csv",
        "Healthcare",
        "medium_snr_persistent",
        "covid_auto_ets",
    ),
    (
        "results/failure_mining/moirai2_ctx1680_m12_covid_deaths_short_auto_ets_predictor_features.csv",
        "Healthcare",
        "medium_snr_persistent",
        "moirai2_ctx1680_m12_covid_auto_ets",
    ),
    (
        "results/failure_mining/timesfm_2_5_m16_covid_deaths_short_auto_ets_predictor_features.csv",
        "Healthcare",
        "medium_snr_persistent",
        "timesfm_2_5_m16_covid_auto_ets",
    ),
    (
        "results/failure_mining/chronos_bolt_small_solar_short_seasonal_naive_predictor_features.csv",
        "Energy",
        "seasonal_high_structure",
        "solar_seasonal_naive",
    ),
    (
        "results/failure_mining/moirai2_ctx1680_solar_m16_solar_short_seasonal_naive_predictor_features.csv",
        "Energy",
        "seasonal_high_structure",
        "moirai2_ctx1680_solar_seasonal_naive",
    ),
    (
        "results/failure_mining/chronos_bolt_small_finance_fred_stress_predictor_features.csv",
        "Finance",
        "low_snr_finance",
        "finance_fred",
    ),
    (
        "results/failure_mining/chronos_bolt_small_loop_seattle_short_seasonal_naive_predictor_features.csv",
        "Transport",
        "medium_snr_persistent",
        "loop_seattle_seasonal_naive",
    ),
    (
        "results/failure_mining/chronos_bolt_small_metr_la_traffic_bma_predictor_features.csv",
        "Traffic",
        "traffic_regime_shift",
        "metr_la_traffic_bma",
    ),
    (
        "results/failure_mining/chronos_bolt_small_pems_bay_traffic_bma_predictor_features.csv",
        "Traffic",
        "traffic_regime_shift",
        "pems_bay_traffic_bma",
    ),
]


def read_rows(path: Path, domain: str, regime: str, source: str) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = {
                "source": source,
                "domain": domain,
                "original_domain": row.get("domain", domain),
                "regime": row.get("regime") or regime,
                "dataset": row.get("dataset", source),
                "series_id": row.get("series_id", ""),
                "window_index": row.get("window_index", ""),
                "failure_delta_005": int(float(row["failure_delta_005"])),
                "relative_error_ratio": float(row["relative_error_ratio"]),
            }
            for feature in FEATURE_NAMES:
                normalized[feature] = float(row.get(feature, 0.0) or 0.0)
            rows.append(normalized)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_sort_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    try:
        window_index = int(float(str(row.get("window_index", "0") or "0")))
    except ValueError:
        window_index = 0
    return (
        str(row.get("source", "")),
        str(row.get("dataset", "")),
        str(row.get("series_id", "")),
        window_index,
    )


def cap_rows_by_domain(rows: list[dict[str, object]], max_rows_per_domain: int) -> list[dict[str, object]]:
    if max_rows_per_domain <= 0:
        return rows
    capped: list[dict[str, object]] = []
    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        domain_rows = [row for row in rows if str(row["domain"]) == domain]
        if len(domain_rows) <= max_rows_per_domain:
            capped.extend(sorted(domain_rows, key=row_sort_key))
            continue
        by_source: dict[str, list[dict[str, object]]] = {}
        for row in sorted(domain_rows, key=row_sort_key):
            by_source.setdefault(str(row.get("source", "")), []).append(row)
        source_names = sorted(by_source)
        selected: list[dict[str, object]] = []
        cursor = {source: 0 for source in source_names}
        while len(selected) < max_rows_per_domain:
            added = False
            for source in source_names:
                index = cursor[source]
                source_rows = by_source[source]
                if index >= len(source_rows):
                    continue
                selected.append(source_rows[index])
                cursor[source] = index + 1
                added = True
                if len(selected) >= max_rows_per_domain:
                    break
            if not added:
                break
        capped.extend(selected)
    return capped


def majority_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows], dtype=int)
    majority = int(np.mean(labels) >= 0.5)
    preds = np.full(labels.shape, majority, dtype=int)
    positives = labels == 1
    negatives = labels == 0
    tpr = float(np.mean(preds[positives] == 1)) if positives.any() else 0.0
    tnr = float(np.mean(preds[negatives] == 0)) if negatives.any() else 0.0
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "predicted_failure_rate": float(np.mean(preds)),
    }


def logistic_leave_one_domain_out(pooled_rows: list[dict[str, object]], domains: list[str]) -> list[dict[str, object]]:
    rows = []
    for domain in domains:
        train = [row for row in pooled_rows if row["domain"] != domain]
        test = [row for row in pooled_rows if row["domain"] == domain]
        x_train = np.asarray([[float(row[feature]) for feature in FEATURE_NAMES] for row in train])
        y_train = np.asarray([int(row["failure_delta_005"]) for row in train], dtype=int)
        x_test = np.asarray([[float(row[feature]) for feature in FEATURE_NAMES] for row in test])
        y_test = np.asarray([int(row["failure_delta_005"]) for row in test], dtype=int)
        if np.unique(y_train).size < 2:
            rows.append(
                {
                    "status": "skipped_single_train_class",
                    "holdout_domain": domain,
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
            continue
        model = fit_balanced_logistic_regression(
            x_train,
            y_train,
            learning_rate=0.1,
            max_iter=2500,
            l2=0.1,
        )
        preds = predict_logistic_regression(model, x_test)
        train_preds = predict_logistic_regression(model, x_train)
        rows.append(
            {
                "status": "ok",
                "estimator": "numpy_balanced_logistic_l2",
                "holdout_domain": domain,
                "n_train": len(train),
                "n_test": len(test),
                "test_failure_rate": float(np.mean(y_test)),
                "test_accuracy": float(np.mean(preds == y_test)),
                "test_balanced_accuracy": balanced_accuracy(y_test, preds),
                "test_predicted_failure_rate": float(np.mean(preds)),
                "train_accuracy": float(np.mean(train_preds == y_train)),
                "train_balanced_accuracy": balanced_accuracy(y_train, train_preds),
            }
        )
    return rows


def tree_leave_one_domain_out(pooled_rows: list[dict[str, object]], domains: list[str]) -> list[dict[str, object]]:
    rows = []
    for domain in domains:
        train = [row for row in pooled_rows if row["domain"] != domain]
        test = [row for row in pooled_rows if row["domain"] == domain]
        y_train = np.asarray([int(row["failure_delta_005"]) for row in train], dtype=int)
        y_test = np.asarray([int(row["failure_delta_005"]) for row in test], dtype=int)
        if np.unique(y_train).size < 2:
            rows.append(
                {
                    "status": "skipped_single_train_class",
                    "holdout_domain": domain,
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
            continue
        model = fit_depth2_threshold_tree(train, FEATURE_NAMES)
        preds = predict_depth2_threshold_tree(model, test)
        train_preds = predict_depth2_threshold_tree(model, train)
        root = model["root"]
        rows.append(
            {
                "status": "ok",
                "estimator": "depth2_threshold_tree",
                "holdout_domain": domain,
                "n_train": len(train),
                "n_test": len(test),
                "test_failure_rate": float(np.mean(y_test)),
                "test_accuracy": float(np.mean(preds == y_test)),
                "test_balanced_accuracy": balanced_accuracy(y_test, preds),
                "test_predicted_failure_rate": float(np.mean(preds)),
                "train_accuracy": float(np.mean(train_preds == y_train)),
                "train_balanced_accuracy": balanced_accuracy(y_train, train_preds),
                "root_feature": root["feature"],
                "root_threshold": root["threshold"],
                "root_direction": root["direction"],
            }
        )
    return rows


def grouped_holdout_key(row: dict[str, object]) -> str:
    return "|".join(
        [
            str(row.get("source", "")),
            str(row.get("dataset", "")),
            str(row.get("series_id", "")),
        ]
    )


def threshold_predictions(rows: list[dict[str, object]], threshold: dict[str, object]) -> np.ndarray:
    values = np.asarray([float(row[str(threshold["feature"])]) for row in rows], dtype=float)
    return apply_threshold(values, float(threshold["threshold"]), str(threshold["direction"]))


def feature_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([[float(row[feature]) for feature in FEATURE_NAMES] for row in rows])


def within_domain_grouped_holdout(pooled_rows: list[dict[str, object]], domains: list[str]) -> list[dict[str, object]]:
    """Validate predictors inside each domain while holding out whole source/dataset/series groups."""
    rows = []
    for domain in domains:
        domain_rows = [row for row in pooled_rows if row["domain"] == domain]
        groups = sorted({grouped_holdout_key(row) for row in domain_rows})
        if len(groups) < 2:
            rows.append(
                {
                    "status": "skipped_too_few_groups",
                    "domain": domain,
                    "n_rows": len(domain_rows),
                    "n_groups": len(groups),
                }
            )
            continue

        labels: list[np.ndarray] = []
        threshold_preds: list[np.ndarray] = []
        logistic_preds: list[np.ndarray] = []
        tree_preds: list[np.ndarray] = []
        majority_preds: list[np.ndarray] = []
        fallback_folds = 0
        ok_folds = 0
        threshold_features: list[str] = []
        for group in groups:
            train = [row for row in domain_rows if grouped_holdout_key(row) != group]
            test = [row for row in domain_rows if grouped_holdout_key(row) == group]
            if not train or not test:
                continue
            y_train = np.asarray([int(row["failure_delta_005"]) for row in train], dtype=int)
            y_test = np.asarray([int(row["failure_delta_005"]) for row in test], dtype=int)
            labels.append(y_test)
            majority = majority_label(train)
            majority_preds.append(np.full(y_test.shape, majority, dtype=int))
            if np.unique(y_train).size < 2:
                fallback_folds += 1
                threshold_preds.append(np.full(y_test.shape, majority, dtype=int))
                logistic_preds.append(np.full(y_test.shape, majority, dtype=int))
                tree_preds.append(np.full(y_test.shape, majority, dtype=int))
                continue

            ok_folds += 1
            threshold = best_threshold(train, FEATURE_NAMES)
            threshold_features.append(str(threshold["feature"]))
            threshold_preds.append(threshold_predictions(test, threshold))

            model = fit_balanced_logistic_regression(
                feature_matrix(train),
                y_train,
                learning_rate=0.1,
                max_iter=2500,
                l2=0.1,
            )
            logistic_preds.append(predict_logistic_regression(model, feature_matrix(test)))

            tree = fit_depth2_threshold_tree(train, FEATURE_NAMES)
            tree_preds.append(predict_depth2_threshold_tree(tree, test))

        if not labels:
            rows.append(
                {
                    "status": "skipped_no_folds",
                    "domain": domain,
                    "n_rows": len(domain_rows),
                    "n_groups": len(groups),
                }
            )
            continue

        y_all = np.concatenate(labels)
        threshold_all = np.concatenate(threshold_preds)
        logistic_all = np.concatenate(logistic_preds)
        tree_all = np.concatenate(tree_preds)
        majority_all = np.concatenate(majority_preds)
        rows.append(
            {
                "status": "ok",
                "domain": domain,
                "n_rows": len(domain_rows),
                "n_groups": len(groups),
                "ok_folds": ok_folds,
                "fallback_single_class_folds": fallback_folds,
                "failure_rate": float(np.mean(y_all)),
                "threshold_balanced_accuracy": balanced_accuracy(y_all, threshold_all),
                "threshold_accuracy": float(np.mean(threshold_all == y_all)),
                "threshold_predicted_failure_rate": float(np.mean(threshold_all)),
                "threshold_features": ";".join(sorted(set(threshold_features))),
                "logistic_balanced_accuracy": balanced_accuracy(y_all, logistic_all),
                "logistic_accuracy": float(np.mean(logistic_all == y_all)),
                "logistic_predicted_failure_rate": float(np.mean(logistic_all)),
                "tree_balanced_accuracy": balanced_accuracy(y_all, tree_all),
                "tree_accuracy": float(np.mean(tree_all == y_all)),
                "tree_predicted_failure_rate": float(np.mean(tree_all)),
                "majority_balanced_accuracy": balanced_accuracy(y_all, majority_all),
                "majority_accuracy": float(np.mean(majority_all == y_all)),
                "majority_predicted_failure_rate": float(np.mean(majority_all)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="pooled_context_failure_predictor")
    parser.add_argument("--max-rows-per-domain", type=int, default=0)
    args = parser.parse_args()

    pooled_rows: list[dict[str, object]] = []
    missing_inputs = []
    for rel_path, domain, regime, source in DEFAULT_INPUTS:
        path = ROOT / rel_path
        if not path.exists():
            missing_inputs.append(rel_path)
            continue
        pooled_rows.extend(read_rows(path, domain, regime, source))
    if missing_inputs:
        raise SystemExit(f"Missing predictor feature inputs: {missing_inputs}")
    if not pooled_rows:
        raise SystemExit("No pooled predictor rows are available")
    original_n_rows = len(pooled_rows)
    pooled_rows = cap_rows_by_domain(pooled_rows, args.max_rows_per_domain)

    holdout_rows = []
    domains = sorted({str(row["domain"]) for row in pooled_rows})
    for domain in domains:
        train = [row for row in pooled_rows if row["domain"] != domain]
        test = [row for row in pooled_rows if row["domain"] == domain]
        threshold = best_threshold(train, FEATURE_NAMES)
        metrics = evaluate_threshold(test, threshold)
        oracle_threshold = best_threshold(test, FEATURE_NAMES)
        oracle_metrics = evaluate_threshold(test, oracle_threshold)
        majority = majority_metrics(test)
        holdout_rows.append(
            {
                "holdout_domain": domain,
                "n_train": len(train),
                "n_test": len(test),
                "test_failure_rate": metrics["failure_rate"],
                "feature": threshold["feature"],
                "threshold": threshold["threshold"],
                "direction": threshold["direction"],
                "train_balanced_accuracy": threshold["balanced_accuracy"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_predicted_failure_rate": metrics["predicted_failure_rate"],
                "oracle_feature": oracle_threshold["feature"],
                "oracle_threshold": oracle_threshold["threshold"],
                "oracle_direction": oracle_threshold["direction"],
                "oracle_balanced_accuracy": oracle_metrics["balanced_accuracy"],
                "oracle_accuracy": oracle_metrics["accuracy"],
                "transfer_gap_to_oracle_balanced_accuracy": float(
                    oracle_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]
                ),
                "majority_accuracy": majority["accuracy"],
                "majority_balanced_accuracy": majority["balanced_accuracy"],
            }
        )

    global_threshold = best_threshold(pooled_rows, FEATURE_NAMES)
    global_metrics = evaluate_threshold(pooled_rows, global_threshold)
    logistic_rows = logistic_leave_one_domain_out(pooled_rows, domains)
    tree_rows = tree_leave_one_domain_out(pooled_rows, domains)
    within_domain_rows = within_domain_grouped_holdout(pooled_rows, domains)
    feature_path = OUT_DIR / f"{args.output_prefix}_features.csv"
    holdout_path = OUT_DIR / f"{args.output_prefix}_leave_one_domain_out.csv"
    logistic_path = OUT_DIR / f"{args.output_prefix}_logistic_leave_one_domain_out.csv"
    tree_path = OUT_DIR / f"{args.output_prefix}_tree_leave_one_domain_out.csv"
    within_domain_path = OUT_DIR / f"{args.output_prefix}_within_domain_grouped.csv"
    report_path = OUT_DIR / f"{args.output_prefix}_report.json"
    write_csv(feature_path, pooled_rows)
    write_csv(holdout_path, holdout_rows)
    write_csv(logistic_path, logistic_rows)
    write_csv(tree_path, tree_rows)
    write_csv(within_domain_path, within_domain_rows)
    logistic_ok_rows = [row for row in logistic_rows if row.get("status") == "ok"]
    tree_ok_rows = [row for row in tree_rows if row.get("status") == "ok"]
    within_domain_ok_rows = [row for row in within_domain_rows if row.get("status") == "ok"]
    report = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_rows": len(pooled_rows),
        "original_n_rows": original_n_rows,
        "domains": domains,
        "max_rows_per_domain": args.max_rows_per_domain,
        "feature_table": str(feature_path),
        "leave_one_domain_out": str(holdout_path),
        "logistic_leave_one_domain_out": str(logistic_path),
        "tree_leave_one_domain_out": str(tree_path),
        "within_domain_grouped": str(within_domain_path),
        "mean_holdout_balanced_accuracy": float(np.mean([float(row["test_balanced_accuracy"]) for row in holdout_rows])),
        "mean_oracle_holdout_balanced_accuracy": float(
            np.mean([float(row["oracle_balanced_accuracy"]) for row in holdout_rows])
        ),
        "mean_transfer_gap_to_oracle_balanced_accuracy": float(
            np.mean([float(row["transfer_gap_to_oracle_balanced_accuracy"]) for row in holdout_rows])
        ),
        "mean_majority_balanced_accuracy": float(
            np.mean([float(row["majority_balanced_accuracy"]) for row in holdout_rows])
        ),
        "mean_logistic_holdout_balanced_accuracy": (
            float(np.mean([float(row["test_balanced_accuracy"]) for row in logistic_ok_rows]))
            if logistic_ok_rows
            else None
        ),
        "mean_tree_holdout_balanced_accuracy": (
            float(np.mean([float(row["test_balanced_accuracy"]) for row in tree_ok_rows]))
            if tree_ok_rows
            else None
        ),
        "mean_within_domain_threshold_balanced_accuracy": (
            float(np.mean([float(row["threshold_balanced_accuracy"]) for row in within_domain_ok_rows]))
            if within_domain_ok_rows
            else None
        ),
        "mean_within_domain_logistic_balanced_accuracy": (
            float(np.mean([float(row["logistic_balanced_accuracy"]) for row in within_domain_ok_rows]))
            if within_domain_ok_rows
            else None
        ),
        "mean_within_domain_tree_balanced_accuracy": (
            float(np.mean([float(row["tree_balanced_accuracy"]) for row in within_domain_ok_rows]))
            if within_domain_ok_rows
            else None
        ),
        "mean_within_domain_majority_balanced_accuracy": (
            float(np.mean([float(row["majority_balanced_accuracy"]) for row in within_domain_ok_rows]))
            if within_domain_ok_rows
            else None
        ),
        "global_threshold": global_threshold,
        "global_metrics": global_metrics,
        "limitations": [
            "small generated-regime pool",
            "single-threshold and small regularized logistic baselines only",
            "oracle thresholds are diagnostic upper bounds and must not be reported as deployed predictors",
            "domain labels are coarse and manually mapped for GIFT-Eval feature seeds",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
