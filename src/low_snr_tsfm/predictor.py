"""Lightweight threshold predictors for ex-ante failure labels."""

from __future__ import annotations

from typing import Any

import numpy as np


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(preds, dtype=int)
    if y.shape != p.shape:
        raise ValueError("labels and preds must have the same shape")
    positives = y == 1
    negatives = y == 0
    tpr = float(np.mean(p[positives] == 1)) if positives.any() else 0.0
    tnr = float(np.mean(p[negatives] == 0)) if negatives.any() else 0.0
    return 0.5 * (tpr + tnr)


def apply_threshold(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if direction == "greater_equal":
        return (x >= threshold).astype(int)
    if direction == "less_equal":
        return (x <= threshold).astype(int)
    raise ValueError(f"Unknown threshold direction: {direction}")


def best_threshold(rows: list[dict[str, object]], feature_names: list[str]) -> dict[str, object]:
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows], dtype=int)
    best: dict[str, object] = {
        "feature": "",
        "threshold": 0.0,
        "direction": "greater_equal",
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
    }
    for feature in feature_names:
        values = np.asarray([float(row[feature]) for row in rows], dtype=float)
        unique = np.unique(values[np.isfinite(values)])
        if unique.size == 0:
            continue
        for threshold in unique:
            for direction in ["greater_equal", "less_equal"]:
                preds = apply_threshold(values, float(threshold), direction)
                accuracy = float(np.mean(preds == labels))
                bal = balanced_accuracy(labels, preds)
                if (bal, accuracy) > (
                    float(best["balanced_accuracy"]),
                    float(best["accuracy"]),
                ):
                    best = {
                        "feature": feature,
                        "threshold": float(threshold),
                        "direction": direction,
                        "accuracy": accuracy,
                        "balanced_accuracy": bal,
                    }
    return best


def evaluate_threshold(rows: list[dict[str, object]], threshold: dict[str, object]) -> dict[str, float]:
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows], dtype=int)
    feature = str(threshold["feature"])
    values = np.asarray([float(row[feature]) for row in rows], dtype=float)
    preds = apply_threshold(values, float(threshold["threshold"]), str(threshold["direction"]))
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": balanced_accuracy(labels, preds),
        "failure_rate": float(np.mean(labels)),
        "predicted_failure_rate": float(np.mean(preds)),
    }


def majority_label(rows: list[dict[str, object]], fallback: int = 0) -> int:
    if not rows:
        return int(fallback)
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows], dtype=int)
    return int(np.mean(labels) >= 0.5)


def fit_depth2_threshold_tree(rows: list[dict[str, object]], feature_names: list[str]) -> dict[str, Any]:
    """Fit a tiny interpretable two-level threshold tree."""
    if not rows:
        raise ValueError("depth-2 threshold tree requires at least one row")
    root = best_threshold(rows, feature_names)
    labels = np.asarray([int(row["failure_delta_005"]) for row in rows], dtype=int)
    root_values = np.asarray([float(row[str(root["feature"])]) for row in rows], dtype=float)
    root_preds = apply_threshold(root_values, float(root["threshold"]), str(root["direction"]))
    fallback = majority_label(rows)
    leaves: dict[int, dict[str, object]] = {}
    for branch_value in [0, 1]:
        branch_rows = [row for row, pred in zip(rows, root_preds) if int(pred) == branch_value]
        branch_labels = {int(row["failure_delta_005"]) for row in branch_rows}
        if len(branch_rows) == 0 or len(branch_labels) < 2:
            leaves[branch_value] = {
                "type": "constant",
                "value": majority_label(branch_rows, fallback=fallback),
                "n_rows": len(branch_rows),
            }
        else:
            leaves[branch_value] = {
                "type": "threshold",
                "threshold": best_threshold(branch_rows, feature_names),
                "n_rows": len(branch_rows),
            }
    return {
        "type": "depth2_threshold_tree",
        "root": root,
        "leaves": leaves,
        "train_balanced_accuracy": balanced_accuracy(labels, predict_depth2_threshold_tree({"root": root, "leaves": leaves}, rows)),
    }


def predict_depth2_threshold_tree(model: dict[str, Any], rows: list[dict[str, object]]) -> np.ndarray:
    root = model["root"]
    root_values = np.asarray([float(row[str(root["feature"])]) for row in rows], dtype=float)
    root_preds = apply_threshold(root_values, float(root["threshold"]), str(root["direction"]))
    preds = []
    leaves = model["leaves"]
    for row, root_pred in zip(rows, root_preds):
        leaf = leaves[int(root_pred)]
        if leaf["type"] == "constant":
            preds.append(int(leaf["value"]))
            continue
        threshold = leaf["threshold"]
        value = np.asarray([float(row[str(threshold["feature"])])], dtype=float)
        preds.append(int(apply_threshold(value, float(threshold["threshold"]), str(threshold["direction"]))[0]))
    return np.asarray(preds, dtype=int)


def fit_balanced_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 2500,
    l2: float = 0.1,
) -> dict[str, Any]:
    """Fit a tiny dependency-free class-balanced logistic model."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must be a 1D array with one label per row")
    if np.unique(y.astype(int)).size < 2:
        raise ValueError("logistic regression requires both classes")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    xs = (x - mean) / scale
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)

    positives = y == 1.0
    negatives = y == 0.0
    sample_weight = np.ones_like(y, dtype=float)
    sample_weight[positives] = len(y) / (2.0 * float(np.sum(positives)))
    sample_weight[negatives] = len(y) / (2.0 * float(np.sum(negatives)))
    weight_sum = float(np.sum(sample_weight))

    coef = np.zeros(xs.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(max_iter):
        logits = np.clip(xs @ coef + intercept, -35.0, 35.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = (probs - y) * sample_weight
        grad_coef = (xs.T @ error) / weight_sum + l2 * coef
        grad_intercept = float(np.sum(error) / weight_sum)
        coef -= learning_rate * grad_coef
        intercept -= learning_rate * grad_intercept

    return {
        "coef": coef,
        "intercept": float(intercept),
        "mean": mean,
        "scale": scale,
    }


def predict_logistic_regression(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    xs = (x - np.asarray(model["mean"], dtype=float)) / np.asarray(model["scale"], dtype=float)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    logits = np.clip(xs @ np.asarray(model["coef"], dtype=float) + float(model["intercept"]), -35.0, 35.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= 0.5).astype(int)
