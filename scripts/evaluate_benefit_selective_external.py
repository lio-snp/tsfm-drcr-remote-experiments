#!/usr/bin/env python3
"""Fail-closed external confirmation for frozen Benefit-Selective DRCR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.benefit_selective import (  # noqa: E402
    frozen_action_grids,
    low_structure_count,
    pre_origin_feature_vector,
)


CONFIG = ROOT / "configs" / "benefit_selective_drcr_external_protocol.json"
JOBS = ROOT / "results" / "aaai_stress" / "benefit_selective_external_execution_jobs.csv"
FREEZE_HASHES = ROOT / "results" / "aaai_stress" / "benefit_selective_external_freeze_hashes.json"
DEV_CONFIG = ROOT / "configs" / "benefit_selective_drcr_protocol.json"
DEV_WINDOWS = ROOT / "results" / "aaai_stress" / "final_main_figure_windows.csv"
DEV_REPLACEMENTS = ROOT / "results" / "aaai_stress" / "remote_q9_final_main_replacements.csv"
OUT_DIR = ROOT / "results" / "aaai_stress"
STATUS_OUT = OUT_DIR / "benefit_selective_external_confirmation_status.json"
PREFLIGHT_OUT = OUT_DIR / "benefit_selective_external_confirmation_preflight.json"
WINDOW_OUT = OUT_DIR / "benefit_selective_external_confirmation_windows.csv"
SUMMARY_OUT = OUT_DIR / "benefit_selective_external_confirmation_summary.csv"
PAIRED_OUT = OUT_DIR / "benefit_selective_external_confirmation_paired_stats.csv"
DOMAIN_OUT = OUT_DIR / "benefit_selective_external_confirmation_domains.csv"
FOREST_OUT = OUT_DIR / "benefit_selective_external_confirmation_forest.csv"
REPORT_OUT = ROOT / "docs" / "benefit_selective_external_confirmation_report.md"
EXPECTED_QUANTILES = [level / 10 for level in range(1, 10)]
NOMINAL_COVERAGE = 0.80


def raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


raise_csv_limit()


@dataclass(frozen=True)
class ExternalWindow:
    key: tuple[str, str, str]
    source: str
    domain: str
    dataset: str
    family: str
    series_id: str
    window_index: str
    origin: int
    horizon: int
    features: dict[str, float]
    low_structure: bool
    diagnostics: dict[str, float | int]
    metrics: dict[str, dict[str, float]]
    context_sha256: str
    target_sha256: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    normalized = np.nan_to_num(array, nan=9.87654321e307, posinf=8.7654321e307, neginf=-8.7654321e307)
    return hashlib.sha256(str(array.shape).encode("ascii") + normalized.tobytes()).hexdigest()


def float_or_nan(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze_hashes(protocol: dict[str, Any]) -> None:
    if not FREEZE_HASHES.exists():
        raise FileNotFoundError(f"Frozen hash contract missing: {FREEZE_HASHES}")
    payload = json.loads(FREEZE_HASHES.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Frozen hash protocol id mismatch")
    if payload.get("external_candidate_id") != protocol["method"]["external_candidate_id"]:
        raise ValueError("Frozen hash candidate id mismatch")
    for relative, expected in payload.get("sha256", {}).items():
        path = ROOT / relative
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"Frozen file changed after protocol lock: {relative}")


def status_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("status") == "ok" else None


def artifact_inventory(jobs: list[dict[str, str]]) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for job in jobs:
        payload = status_payload(ROOT / job["status_path"])
        detail = "missing_or_non_ok_status"
        ready = False
        raw_path = ""
        history_path = ""
        windows_run = 0
        if payload is not None:
            raw_path = str(payload.get("raw_forecasts") or "")
            history_path = str(payload.get("history_context_sidecar") or "")
            windows_run = int(payload.get("windows_run") or 0)
            raw_exists = bool(raw_path) and artifact_path(raw_path).exists()
            history_exists = bool(history_path) and artifact_path(history_path).exists()
            exact_windows = windows_run == int(job["windows_requested"])
            exact_series = int(payload.get("series_run") or 0) == int(job["series_requested"])
            exact_model = str(payload.get("model_id") or "") == job["model_id"]
            observed_quantiles = [round(float(value), 10) for value in payload.get("quantile_levels") or []]
            exact_quantiles = observed_quantiles == EXPECTED_QUANTILES
            family_contract = True
            if job["family"] == "moirai":
                family_contract = (
                    int(payload.get("num_samples") or 0) == int(job["moirai_samples"])
                    and int(payload.get("seed") or -1) == 7
                    and payload.get("model_family") == "moirai1"
                )
            ready = raw_exists and history_exists and exact_windows and exact_series and exact_model and exact_quantiles and family_contract
            detail = (
                "ready" if ready else
                f"raw={raw_exists};history={history_exists};windows={windows_run};series={exact_series};"
                f"model={exact_model};q9={exact_quantiles};family={family_contract}"
            )
        inventory.append(
            {
                "job_id": job["job_id"],
                "domain": job["domain"],
                "dataset": job["dataset"],
                "family": job["family"],
                "ready": int(ready),
                "windows_run": windows_run,
                "raw_path": raw_path,
                "history_path": history_path,
                "detail": detail,
            }
        )
    return inventory


def write_preflight(inventory: list[dict[str, object]], protocol: dict[str, Any]) -> dict[str, Any]:
    ready = sum(int(row["ready"]) for row in inventory)
    payload = {
        "status": "ready_for_locked_evaluation" if ready == len(inventory) else "pending_external_artifacts",
        "protocol_id": protocol["protocol_id"],
        "external_candidate_id": protocol["method"]["external_candidate_id"],
        "jobs_ready": ready,
        "jobs_expected": len(inventory),
        "windows_ready": sum(int(row["windows_run"]) for row in inventory if int(row["ready"])),
        "windows_expected": sum(int(row["windows_run"]) if int(row["ready"]) else 16 for row in inventory),
        "outcomes_inspected": False,
        "detail": [row for row in inventory if not int(row["ready"])][:5],
    }
    PREFLIGHT_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_values(value: str) -> np.ndarray:
    parsed = json.loads(value)
    return np.asarray([float("nan") if item is None else float(item) for item in parsed], dtype=float)


def group_raw_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series_id"], row["window_index"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["horizon_index"]))
    return grouped


def q9_components(actual: np.ndarray, levels: list[float], grid: np.ndarray) -> tuple[float, float]:
    errors = actual[:, None] - grid
    taus = np.asarray(levels, dtype=float)[None, :]
    pinball = np.maximum(taus * errors, (taus - 1.0) * errors)
    return float(np.sum(pinball)), float(np.sum(np.abs(actual)))


def mase_scale(context: np.ndarray, season_length: int) -> float:
    finite = np.asarray(context, dtype=float)
    finite = finite[np.isfinite(finite)]
    period = max(1, int(season_length))
    if finite.size > period:
        scale = float(np.mean(np.abs(finite[period:] - finite[:-period])))
    elif finite.size:
        scale = float(np.mean(np.abs(finite - np.mean(finite))))
    else:
        scale = 0.0
    return scale


def metric_bundle(
    actual: np.ndarray,
    baseline: np.ndarray,
    levels: list[float],
    grid: np.ndarray,
    point: np.ndarray,
    native_grid: np.ndarray,
    native_point: np.ndarray,
    scale: float,
) -> dict[str, float]:
    pinball_sum, target_abs_sum = q9_components(actual, levels, grid)
    q10 = grid[:, 0]
    q90 = grid[:, -1]
    return {
        "pinball_sum": pinball_sum,
        "target_abs_sum": target_abs_sum,
        "abs_error_sum": float(np.sum(np.abs(actual - point))),
        "squared_error_sum": float(np.sum((actual - point) ** 2)),
        "scaled_abs_error_sum": float(np.sum(np.abs(actual - point)) / max(scale, 1e-12)),
        "baseline_abs_error_sum": float(np.sum(np.abs(actual - baseline))),
        "baseline_squared_error_sum": float(np.sum((actual - baseline) ** 2)),
        "coverage_count": float(np.sum((actual >= q10) & (actual <= q90))),
        "interval_width_sum": float(np.sum(q90 - q10)),
        "n_points": float(actual.size),
        "changed": float(
            not (
                np.allclose(grid, native_grid, atol=1e-12, rtol=0.0)
                and np.allclose(point, native_point, atol=1e-12, rtol=0.0)
            )
        ),
    }


def build_external_windows(
    jobs: list[dict[str, str]], inventory: list[dict[str, object]], protocol: dict[str, Any]
) -> list[ExternalWindow]:
    by_job = {str(row["job_id"]): row for row in inventory}
    method = protocol["method"]
    taxonomy = protocol["low_structure_taxonomy"]
    windows: list[ExternalWindow] = []
    for job in jobs:
        record = by_job[job["job_id"]]
        raw_rows = read_csv(artifact_path(str(record["raw_path"])))
        history_rows = read_csv(artifact_path(str(record["history_path"])))
        histories = {(row["series_id"], row["window_index"]): row for row in history_rows}
        grouped = group_raw_rows(raw_rows)
        if set(grouped) != set(histories) or len(grouped) != int(job["windows_requested"]):
            raise ValueError(f"Raw/history inventory mismatch for {job['job_id']}")
        for (series_id, window_index), rows in sorted(grouped.items()):
            history = histories[(series_id, window_index)]
            horizon = int(rows[0]["horizon"])
            if len(rows) != horizon or [int(row["horizon_index"]) for row in rows] != list(range(1, horizon + 1)):
                raise ValueError(f"Incomplete horizon for {job['job_id']} {series_id} {window_index}")
            levels = EXPECTED_QUANTILES
            quantile_columns = [f"forecast_q{int(level * 100):02d}" for level in levels]
            if any(column not in rows[0] for column in quantile_columns):
                raise ValueError(f"Missing common q9 grid in {job['job_id']}")
            grid = np.asarray([[float(row[column]) for column in quantile_columns] for row in rows], dtype=float)
            if np.any(np.diff(grid, axis=1) < -1e-10):
                raise ValueError(f"Crossing quantiles in {job['job_id']} {series_id} {window_index}")
            actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
            baseline = np.asarray([float(row["baseline_forecast"]) for row in rows], dtype=float)
            native_mean = np.asarray([float(row["forecast_mean"]) for row in rows], dtype=float)
            context = parse_values(history["context_values"])
            scale = mase_scale(context, horizon)
            if any(row.get("baseline_mode") != "rolling_pre_origin" for row in rows):
                raise ValueError(f"Non-frozen baseline mode in {job['job_id']}")
            if any(int(row.get("baseline_context_length") or 0) > int(job["baseline_context_cap"]) for row in rows):
                raise ValueError(f"Classical context cap violation in {job['job_id']}")
            if any(int(row.get("context_length") or 0) > int(job["context_cap"]) for row in rows):
                raise ValueError(f"TSFM context cap violation in {job['job_id']}")
            features = pre_origin_feature_vector(
                context,
                horizon,
                levels,
                grid,
                job["family"],
                method["smooth_interval_head"],
            )
            if set(method["features"]) - set(features):
                raise ValueError(f"External feature contract incomplete for {job['job_id']}")
            action_grids, diagnostics = frozen_action_grids(
                native_mean,
                baseline,
                levels,
                grid,
                features,
                taxonomy,
                method,
            )
            baseline_grid = np.repeat(baseline[:, None], len(levels), axis=1)
            blend_weight = float(protocol["co_primary_pass_criteria"]["fixed_blend_weight"])
            action_grids["fixed_blend_0.50"] = grid + blend_weight * (baseline_grid - grid)
            action_grids["classical_deterministic"] = baseline_grid
            native_point = native_mean
            metrics: dict[str, dict[str, float]] = {}
            valid = (
                np.isfinite(actual)
                & np.isfinite(baseline)
                & np.isfinite(native_mean)
                & np.all(np.isfinite(grid), axis=1)
            )
            if not np.any(valid):
                raise ValueError(f"No finite target points in {job['job_id']} {series_id} {window_index}")
            for action, action_grid in action_grids.items():
                if action == "native_tsfm":
                    point = native_point
                elif action == "classical_deterministic":
                    point = baseline
                else:
                    point = action_grid[:, 4]
                metrics[action] = metric_bundle(
                    actual[valid], baseline[valid], levels, action_grid[valid], point[valid], grid[valid],
                    native_point[valid], scale,
                )
            oracle_action = min(
                ["native_tsfm", *method["repair_actions"]],
                key=lambda action: metrics[action]["pinball_sum"],
            )
            metrics["oracle_best_action"] = dict(metrics[oracle_action])
            count = low_structure_count(features, taxonomy)
            low_structure = count >= int(taxonomy["minimum_active_factors"])
            key = (job["job_id"], series_id, window_index)
            windows.append(
                ExternalWindow(
                    key=key,
                    source=job["job_id"],
                    domain=job["domain"],
                    dataset=job["dataset"],
                    family=job["family"],
                    series_id=series_id,
                    window_index=window_index,
                    origin=int(rows[0]["origin"]),
                    horizon=horizon,
                    features=features,
                    low_structure=low_structure,
                    diagnostics={**diagnostics, "low_structure_factor_count": count},
                    metrics=metrics,
                    context_sha256=array_sha256(context),
                    target_sha256=array_sha256(actual),
                )
            )
    expected = len(jobs) * int(protocol["evaluation"]["windows_per_model_domain"])
    if len(windows) != expected:
        raise ValueError(f"Expected {expected} external windows, found {len(windows)}")
    validate_cross_family_pairing(windows, protocol)
    return windows


def validate_cross_family_pairing(windows: list[ExternalWindow], protocol: dict[str, Any]) -> None:
    excluded_bases = set(protocol["domain_sampling"]["exclude_development_base_datasets"])
    if any(window.dataset.split("/", 1)[0] in excluded_bases for window in windows):
        raise ValueError("Development base dataset appeared in external windows")
    paired: dict[tuple[str, str, str], list[ExternalWindow]] = defaultdict(list)
    for window in windows:
        paired[(window.dataset, window.series_id, window.window_index)].append(window)
    for key, group in paired.items():
        if {window.family for window in group} != {"chronos", "moirai", "timesfm"}:
            raise ValueError(f"Cross-family pairing incomplete for {key}")
        if len({window.origin for window in group}) != 1:
            raise ValueError(f"Cross-family origin mismatch for {key}")
        if len({window.context_sha256 for window in group}) != 1:
            raise ValueError(f"Cross-family context mismatch for {key}")
        if len({window.target_sha256 for window in group}) != 1:
            raise ValueError(f"Cross-family target mismatch for {key}")


def dev_uid(row: dict[str, str]) -> tuple[str, str, str]:
    return row["source"], row["series_id"], row["window_index"]


def load_frozen_calibration(protocol: dict[str, Any]):
    method = protocol["method"]
    actions = list(method["repair_actions"])
    methods = ["native_tsfm", *actions]
    replacements = [row for row in read_csv(DEV_REPLACEMENTS) if row["ready_for_final_main_refresh"] == "1"]
    sources = {row["rerun_slug"] for row in replacements}
    rows_by_method: dict[str, dict[tuple[str, str, str], dict[str, str]]] = defaultdict(dict)
    for row in read_csv(DEV_WINDOWS):
        if row["source"] in sources and row["candidate_id"] in methods and row["phase"] == "calibration":
            rows_by_method[row["candidate_id"]][dev_uid(row)] = row
    keys = sorted(rows_by_method["native_tsfm"])
    if len(keys) != 196 or any(set(rows_by_method[action]) != set(keys) for action in methods):
        raise ValueError("Frozen development calibration inventory is not exactly 196 windows")
    features: dict[tuple[str, str, str], dict[str, float]] = {}
    native_rows = rows_by_method["native_tsfm"]
    for replacement in replacements:
        source = replacement["rerun_slug"]
        for row in read_csv(ROOT / replacement["feature_path"]):
            key = (source, row["series_id"], row["window_index"])
            if key not in native_rows:
                continue
            parsed = {name: float_or_nan(row.get(name)) for name in method["features"]}
            parsed["native_width_ratio"] = float(native_rows[key]["native_width_ratio"])
            parsed["smooth_width_score"] = float(native_rows[key]["smooth_width_score"])
            parsed["family_is_timesfm"] = float(native_rows[key]["family"] == "timesfm")
            features[key] = parsed
    if set(features) != set(keys):
        raise ValueError("Frozen calibration feature inventory mismatch")
    return keys, rows_by_method, features


def benefit_target(native: dict[str, str], action: dict[str, str]) -> float:
    native_loss = float(native["model_pinball_sum"])
    action_loss = float(action["repair_pinball_sum"])
    return (native_loss - action_loss) / max(abs(native_loss), 1e-9)


def fit_frozen_benefit_models(protocol: dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    method = protocol["method"]
    keys, rows_by_method, features = load_frozen_calibration(protocol)
    names = list(method["features"])
    x = np.asarray([[features[key][name] for name in names] for key in keys], dtype=float)
    groups = np.asarray([key[0] for key in keys])
    folds = GroupKFold(n_splits=min(int(method["cross_fit_folds"]), len(set(groups))))
    actions = list(method["repair_actions"])
    targets = {
        action: np.asarray(
            [benefit_target(rows_by_method["native_tsfm"][key], rows_by_method[action][key]) for key in keys],
            dtype=float,
        )
        for action in actions
    }

    def model(seed: int):
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=int(method["n_estimators"]),
                min_samples_leaf=int(method["min_samples_leaf"]),
                random_state=int(method["regressor_random_state"]) + seed,
                n_jobs=1,
            ),
        )

    oof = {action: np.full(len(keys), np.nan) for action in actions}
    for fold_index, (train, valid) in enumerate(folds.split(x, groups=groups)):
        for action_index, action in enumerate(actions):
            fitted = model(100 * fold_index + action_index)
            fitted.fit(x[train], targets[action][train])
            oof[action][valid] = fitted.predict(x[valid])
    if any(np.isnan(values).any() for values in oof.values()):
        raise ValueError("Frozen cross-fit predictions contain missing values")
    offsets = {
        action: float(np.quantile(targets[action] - oof[action], float(method["residual_quantile"])))
        for action in actions
    }
    fitted_models = {}
    for action_index, action in enumerate(actions):
        fitted = model(10_000 + action_index)
        fitted.fit(x, targets[action])
        fitted_models[action] = fitted
    return fitted_models, offsets


def benefit_choices(
    windows: list[ExternalWindow], protocol: dict[str, Any]
) -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str, str], dict[str, float]]]:
    models, offsets = fit_frozen_benefit_models(protocol)
    method = protocol["method"]
    names = list(method["features"])
    x = np.asarray([[window.features[name] for name in names] for window in windows], dtype=float)
    predictions = {action: models[action].predict(x) for action in method["repair_actions"]}
    choices: dict[tuple[str, str, str], str] = {}
    scores: dict[tuple[str, str, str], dict[str, float]] = {}
    for index, window in enumerate(windows):
        action_scores = {action: float(predictions[action][index] + offsets[action]) for action in predictions}
        best_action = max(action_scores, key=lambda action: (action_scores[action], action))
        choices[window.key] = (
            best_action
            if action_scores[best_action] > float(method["intervene_only_if_lcb_above"])
            else "native_tsfm"
        )
        scores[window.key] = action_scores
    return choices, scores


def matched_choices(
    windows: list[ExternalWindow], selected: dict[tuple[str, str, str], str], mode: str
) -> dict[tuple[str, str, str], str]:
    by_source: dict[str, list[ExternalWindow]] = defaultdict(list)
    for window in windows:
        by_source[window.source].append(window)
    output: dict[tuple[str, str, str], str] = {}
    for source, source_windows in by_source.items():
        actions = sorted(
            [selected[window.key] for window in source_windows if selected[window.key] != "native_tsfm"],
            key=lambda action: stable_seed("matched-action", source, action),
        )
        if mode == "random":
            ordered = sorted(source_windows, key=lambda window: stable_seed("external-random-v1", *window.key))
        elif mode == "history_hcr":
            ordered = sorted(
                source_windows,
                key=lambda window: (-window.features["horizon_context_ratio"], stable_seed("external-hcr-v1", *window.key)),
            )
        else:
            raise ValueError(mode)
        for index, window in enumerate(ordered):
            output[window.key] = actions[index] if index < len(actions) else "native_tsfm"
    return output


def scope_match(window: ExternalWindow, scope: str) -> bool:
    return {
        "overall": True,
        "low_structure": window.low_structure,
        "structured_complement": not window.low_structure,
    }[scope]


def aggregate_metric(windows: list[ExternalWindow], choices: dict[tuple[str, str, str], str]) -> dict[str, float]:
    metrics = [window.metrics[choices[window.key]] for window in windows]
    pinball = sum(row["pinball_sum"] for row in metrics)
    target_abs = sum(row["target_abs_sum"] for row in metrics)
    points = sum(row["n_points"] for row in metrics)
    mae = sum(row["abs_error_sum"] for row in metrics) / points
    rmse = math.sqrt(sum(row["squared_error_sum"] for row in metrics) / points)
    baseline_mae = sum(row["baseline_abs_error_sum"] for row in metrics) / points
    baseline_rmse = math.sqrt(sum(row["baseline_squared_error_sum"] for row in metrics) / points)
    coverage = sum(row["coverage_count"] for row in metrics) / points
    return {
        "q9_wql": 2.0 * pinball / max(9.0 * target_abs, 1e-12),
        "relmae": mae / max(baseline_mae, 1e-12),
        "relrmse": rmse / max(baseline_rmse, 1e-12),
        "mase": sum(row["scaled_abs_error_sum"] for row in metrics) / points,
        "coverage": coverage,
        "coverage_gap": abs(coverage - NOMINAL_COVERAGE),
        "interval_width": sum(row["interval_width_sum"] for row in metrics) / points,
        "intervention_rate": sum(row["changed"] for row in metrics) / len(metrics),
        "n_windows": float(len(metrics)),
        "n_points": points,
    }


def domain_metrics(
    windows: list[ExternalWindow], choices: dict[tuple[str, str, str], str], scope: str
) -> dict[str, dict[str, float]]:
    by_domain: dict[str, list[ExternalWindow]] = defaultdict(list)
    for window in windows:
        if scope_match(window, scope):
            by_domain[window.dataset].append(window)
    return {domain: aggregate_metric(rows, choices) for domain, rows in by_domain.items() if rows}


def macro_metrics(per_domain: dict[str, dict[str, float]]) -> dict[str, float]:
    output = {"n_domains": float(len(per_domain))}
    for metric in ["q9_wql", "relmae", "relrmse", "mase", "coverage", "coverage_gap", "interval_width", "intervention_rate"]:
        output[metric] = float(np.mean([row[metric] for row in per_domain.values()]))
    output["n_windows"] = float(sum(row["n_windows"] for row in per_domain.values()))
    return output


def hierarchical_bootstrap_delta(
    windows: list[ExternalWindow],
    candidate: dict[tuple[str, str, str], str],
    comparator: dict[tuple[str, str, str], str],
    scope: str,
    metric: str,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    eligible = [window for window in windows if scope_match(window, scope)]
    by_domain_series: dict[str, dict[str, list[ExternalWindow]]] = defaultdict(lambda: defaultdict(list))
    for window in eligible:
        by_domain_series[window.dataset][window.series_id].append(window)
    domains = sorted(by_domain_series)
    if not domains:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for draw_index in range(samples):
        sampled_domain_deltas = []
        for domain in rng.choice(domains, size=len(domains), replace=True):
            series_map = by_domain_series[str(domain)]
            series = sorted(series_map)
            sampled_windows: list[ExternalWindow] = []
            for series_id in rng.choice(series, size=len(series), replace=True):
                sampled_windows.extend(series_map[str(series_id)])
            sampled_domain_deltas.append(
                aggregate_metric(sampled_windows, candidate)[metric]
                - aggregate_metric(sampled_windows, comparator)[metric]
            )
        draws[draw_index] = float(np.mean(sampled_domain_deltas))
    alpha = 1.0 - confidence
    point_candidate = macro_metrics(domain_metrics(eligible, candidate, "overall"))[metric]
    point_comparator = macro_metrics(domain_metrics(eligible, comparator, "overall"))[metric]
    return (
        point_candidate - point_comparator,
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def sign_flip_p(domain_deltas: list[float]) -> float:
    values = np.asarray(domain_deltas, dtype=float)
    if values.size == 0:
        return float("nan")
    observed = float(np.mean(values))
    if values.size <= 18:
        stats = [float(np.mean(values * np.asarray(signs))) for signs in itertools.product([-1.0, 1.0], repeat=values.size)]
        return float((sum(stat <= observed + 1e-15 for stat in stats) + 1) / (len(stats) + 1))
    rng = np.random.default_rng(stable_seed("sign-flip", str(values.size)))
    signs = rng.choice([-1.0, 1.0], size=(100_000, values.size))
    stats = np.mean(signs * values[None, :], axis=1)
    return float((np.sum(stats <= observed + 1e-15) + 1) / (stats.size + 1))


def lodo_stable(
    candidate_domains: dict[str, dict[str, float]], comparator_domains: dict[str, dict[str, float]], metric: str
) -> tuple[bool, float]:
    domains = sorted(set(candidate_domains) & set(comparator_domains))
    deltas = []
    for omitted in domains:
        kept = [domain for domain in domains if domain != omitted]
        deltas.append(float(np.mean([candidate_domains[d][metric] - comparator_domains[d][metric] for d in kept])))
    return all(delta <= 0.0 for delta in deltas), max(deltas) if deltas else float("nan")


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    verify_freeze_hashes(protocol)
    jobs = read_csv(JOBS)
    if len(jobs) != 42:
        raise ValueError(f"Frozen job inventory changed: {len(jobs)}")
    inventory = artifact_inventory(jobs)
    preflight = write_preflight(inventory, protocol)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if preflight["jobs_ready"] != preflight["jobs_expected"]:
        raise SystemExit("External evaluation remains locked until all 42 exact artifacts are ready")

    windows = build_external_windows(jobs, inventory, protocol)
    benefit, benefit_scores = benefit_choices(windows, protocol)
    native = {window.key: "native_tsfm" for window in windows}
    always_expert = {window.key: "drcr_expert_pull_1.25_cap_1.10" for window in windows}
    always_point = {window.key: "drcr_point" for window in windows}
    fixed_blend = {window.key: "fixed_blend_0.50" for window in windows}
    classical = {window.key: "classical_deterministic" for window in windows}
    oracle = {window.key: "oracle_best_action" for window in windows}
    matched_random = matched_choices(windows, benefit, "random")
    matched_hcr = matched_choices(windows, benefit, "history_hcr")
    methods = {
        "Native TSFM": native,
        "Classical deterministic": classical,
        "Always point repair": always_point,
        "Always ExpertPull": always_expert,
        "Fixed 50/50 blend": fixed_blend,
        "Matched random gate": matched_random,
        "Matched HCR gate": matched_hcr,
        "Benefit-LCB": benefit,
        "Oracle action": oracle,
    }

    summary_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    domain_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for method_name, choices in methods.items():
        for scope in ["overall", "low_structure", "structured_complement"]:
            per_domain = domain_metrics(windows, choices, scope)
            domain_cache[(method_name, scope)] = per_domain
            macro = macro_metrics(per_domain)
            summary_rows.append({"method": method_name, "scope": scope, "family": "all", **macro})
            for domain, values in per_domain.items():
                domain_rows.append(
                    {"method": method_name, "scope": scope, "family": "all", "data_domain": domain, **values}
                )
            for family in sorted({window.family for window in windows}):
                family_windows = [window for window in windows if window.family == family]
                family_domains = domain_metrics(family_windows, choices, scope)
                family_macro = macro_metrics(family_domains)
                summary_rows.append({"method": method_name, "scope": scope, "family": family, **family_macro})
                for domain, values in family_domains.items():
                    domain_rows.append(
                        {"method": method_name, "scope": scope, "family": family, "data_domain": domain, **values}
                    )

    comparisons = [
        ("overall_q9", "overall", "q9_wql", "Native TSFM"),
        ("low_structure_q9", "low_structure", "q9_wql", "Native TSFM"),
        ("structured_q9_harm", "structured_complement", "q9_wql", "Native TSFM"),
        ("structured_coverage_gap_harm", "structured_complement", "coverage_gap", "Native TSFM"),
        ("vs_always_repair", "overall", "q9_wql", "Always ExpertPull"),
        ("vs_fixed_blend", "overall", "q9_wql", "Fixed 50/50 blend"),
        ("vs_matched_random", "overall", "q9_wql", "Matched random gate"),
    ]
    inference = protocol["inference"]
    paired_rows: list[dict[str, object]] = []
    for comparison_id, scope, metric, comparator_name in comparisons:
        point, low, high = hierarchical_bootstrap_delta(
            windows,
            benefit,
            methods[comparator_name],
            scope,
            metric,
            int(inference["paired_hierarchical_bootstrap_samples"]),
            float(inference["confidence"]),
            stable_seed(protocol["protocol_id"], comparison_id),
        )
        candidate_domains = domain_cache[("Benefit-LCB", scope)]
        comparator_domains = domain_cache[(comparator_name, scope)]
        common = sorted(set(candidate_domains) & set(comparator_domains))
        deltas = [candidate_domains[domain][metric] - comparator_domains[domain][metric] for domain in common]
        stable, max_lodo = lodo_stable(candidate_domains, comparator_domains, metric)
        paired_rows.append(
            {
                "comparison_id": comparison_id,
                "candidate": "Benefit-LCB",
                "comparator": comparator_name,
                "family": "all",
                "scope": scope,
                "metric": metric,
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "n_domains": len(common),
                "one_sided_domain_sign_flip_p": sign_flip_p(deltas),
                "lodo_direction_stable": int(stable),
                "max_lodo_delta": max_lodo,
            }
        )

    for family in sorted({window.family for window in windows}):
        family_windows = [window for window in windows if window.family == family]
        point, low, high = hierarchical_bootstrap_delta(
            family_windows,
            benefit,
            native,
            "overall",
            "q9_wql",
            int(inference["paired_hierarchical_bootstrap_samples"]),
            float(inference["confidence"]),
            stable_seed(protocol["protocol_id"], "overall_q9", family),
        )
        candidate_domains = domain_metrics(family_windows, benefit, "overall")
        comparator_domains = domain_metrics(family_windows, native, "overall")
        common = sorted(set(candidate_domains) & set(comparator_domains))
        deltas = [candidate_domains[domain]["q9_wql"] - comparator_domains[domain]["q9_wql"] for domain in common]
        stable, max_lodo = lodo_stable(candidate_domains, comparator_domains, "q9_wql")
        paired_rows.append(
            {
                "comparison_id": f"overall_q9_{family}",
                "candidate": "Benefit-LCB",
                "comparator": "Native TSFM",
                "family": family,
                "scope": "overall",
                "metric": "q9_wql",
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "n_domains": len(common),
                "one_sided_domain_sign_flip_p": sign_flip_p(deltas),
                "lodo_direction_stable": int(stable),
                "max_lodo_delta": max_lodo,
            }
        )

    criteria = protocol["co_primary_pass_criteria"]
    paired = {row["comparison_id"]: row for row in paired_rows}
    benefit_structured = next(
        row for row in summary_rows
        if row["method"] == "Benefit-LCB" and row["scope"] == "structured_complement" and row["family"] == "all"
    )
    checks = {
        "overall_q9": float(paired["overall_q9"]["ci_high"]) <= float(criteria["overall_domain_macro_q9_wql_delta_ci_upper"]),
        "low_structure_q9": float(paired["low_structure_q9"]["ci_high"]) <= float(criteria["low_structure_domain_macro_q9_wql_delta_ci_upper"]),
        "structured_q9_harm": float(paired["structured_q9_harm"]["ci_high"]) <= float(criteria["structured_complement_q9_wql_delta_ci_upper"]),
        "structured_coverage_gap_harm": float(paired["structured_coverage_gap_harm"]["ci_high"]) <= float(criteria["structured_complement_coverage_gap_delta_ci_upper"]),
        "structured_intervention_rate": float(benefit_structured["intervention_rate"]) <= float(criteria["structured_complement_intervention_rate"]),
        "beats_always_repair": float(paired["vs_always_repair"]["ci_high"]) <= float(criteria["baseline_superiority_q9_wql_delta_ci_upper"]),
        "beats_fixed_blend": float(paired["vs_fixed_blend"]["ci_high"]) <= float(criteria["baseline_superiority_q9_wql_delta_ci_upper"]),
        "beats_matched_random": float(paired["vs_matched_random"]["ci_high"]) <= float(criteria["baseline_superiority_q9_wql_delta_ci_upper"]),
        "lodo_direction_stable": bool(paired["overall_q9"]["lodo_direction_stable"]),
    }

    window_rows = []
    for window in windows:
        selected = benefit[window.key]
        window_rows.append(
            {
                "source": window.source,
                "broad_domain": window.domain,
                "data_domain": window.dataset,
                "family": window.family,
                "series_id": window.series_id,
                "window_index": window.window_index,
                "origin": window.origin,
                "horizon": window.horizon,
                "low_structure": int(window.low_structure),
                "low_structure_factor_count": window.diagnostics["low_structure_factor_count"],
                "selected_action": selected,
                "intervention_active": int(window.metrics[selected]["changed"]),
                "lcb_drcr_point": benefit_scores[window.key]["drcr_point"],
                "lcb_expert_pull": benefit_scores[window.key]["drcr_expert_pull_1.25_cap_1.10"],
                **{f"feature_{name}": window.features[name] for name in protocol["method"]["features"]},
            }
        )
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    write_csv(PAIRED_OUT, paired_rows)
    write_csv(DOMAIN_OUT, domain_rows)
    forest_rows: list[dict[str, object]] = []
    for scope in ["overall", "low_structure", "structured_complement"]:
        for family in ["all", *sorted({window.family for window in windows})]:
            subset = windows if family == "all" else [window for window in windows if window.family == family]
            candidate_domains = domain_metrics(subset, benefit, scope)
            comparator_domains = domain_metrics(subset, native, scope)
            for domain in sorted(set(candidate_domains) & set(comparator_domains)):
                candidate_row = candidate_domains[domain]
                comparator_row = comparator_domains[domain]
                forest_rows.append(
                    {
                        "scope": scope,
                        "family": family,
                        "data_domain": domain,
                        "n_windows": candidate_row["n_windows"],
                        **{
                            f"delta_{metric}": candidate_row[metric] - comparator_row[metric]
                            for metric in ["q9_wql", "relmae", "relrmse", "mase", "coverage_gap", "interval_width"]
                        },
                    }
                )
    write_csv(FOREST_OUT, forest_rows)

    final_status = {
        **preflight,
        "status": "pass" if all(checks.values()) else "fail_locked_criteria",
        "outcomes_inspected": True,
        "n_windows": len(windows),
        "n_data_domains": len({window.dataset for window in windows}),
        "n_model_families": len({window.family for window in windows}),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "artifacts": {
            "windows": str(WINDOW_OUT.relative_to(ROOT)),
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "paired_stats": str(PAIRED_OUT.relative_to(ROOT)),
            "domain_metrics": str(DOMAIN_OUT.relative_to(ROOT)),
            "forest_ready": str(FOREST_OUT.relative_to(ROOT)),
        },
    }
    STATUS_OUT.write_text(json.dumps(final_status, indent=2), encoding="utf-8")

    compact = []
    for row in summary_rows:
        if row["scope"] not in {"overall", "low_structure", "structured_complement"} or row["family"] != "all":
            continue
        compact.append(
            {
                "Method": row["method"],
                "Scope": row["scope"],
                "N": int(row["n_windows"]),
                "Domains": int(row["n_domains"]),
                "q9-WQL": f"{float(row['q9_wql']):.4f}",
                "RelMAE": f"{float(row['relmae']):.3f}",
                "RelRMSE": f"{float(row['relrmse']):.3f}",
                "MASE": f"{float(row['mase']):.3f}",
                "Coverage": f"{100 * float(row['coverage']):.1f}%",
                "Intervene": f"{100 * float(row['intervention_rate']):.1f}%",
            }
        )
    REPORT_OUT.write_text(
        "# Benefit-Selective DRCR External Confirmation\n\n"
        f"Protocol: `{protocol['protocol_id']}`. Candidate: `{protocol['method']['external_candidate_id']}`.\n\n"
        f"Locked result: **{final_status['status']}** ({sum(checks.values())}/{len(checks)} criteria passed).\n\n"
        "## Main Results\n\n"
        + markdown_table(
            compact,
            [("Method", "Method"), ("Scope", "Scope"), ("N", "N"), ("Domains", "Domains"),
             ("q9-WQL", "q9-WQL"), ("RelMAE", "RelMAE"), ("RelRMSE", "RelRMSE"),
             ("MASE", "MASE"), ("Coverage", "Coverage"), ("Intervene", "Intervention")],
        )
        + "\n\n## Locked Criteria\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in checks.items())
        + "\n\nAll detailed domain-level and paired statistics are exported as CSV.\n",
        encoding="utf-8",
    )
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
