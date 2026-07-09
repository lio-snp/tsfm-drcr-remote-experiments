#!/usr/bin/env python
"""Run Conformal Policy Repair with LTT-certified policy selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_reviewer_required_baselines_goal as reviewer  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402
from low_snr_tsfm.risk_control import ltt_risk_tests, select_highest_utility_certified  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "conformal_policy_repair_grid.json"
OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "conformal_policy_repair_report.md"
STATUS_PATH = OUT_DIR / "conformal_policy_repair_status.json"


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


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


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def load_config() -> tuple[dict[str, object], str]:
    raw = CONFIG_PATH.read_text()
    parsed = json.loads(raw)
    normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return parsed, hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def candidate_policies(config: dict[str, object]) -> list[dict[str, object]]:
    grid = config["candidate_grid"]
    candidates: list[dict[str, object]] = [{"kind": "no_repair", "policy_id": "no_repair", "weight": 0.0}]
    for weight in grid["global_blend_weights"]:
        weight = float(weight)
        if weight <= 0.0:
            continue
        candidates.append(
            {
                "kind": "global_blend",
                "policy_id": f"global_blend_w{weight:.3f}",
                "weight": weight,
            }
        )
    conflict = grid["conflict_only"]
    for threshold in conflict["conflict_thresholds"]:
        for weight in conflict["weights"]:
            candidates.append(
                {
                    "kind": "conflict_only",
                    "policy_id": f"conflict_only_le{float(threshold):.2f}_w{float(weight):.3f}",
                    "conflict_threshold": float(threshold),
                    "weight": float(weight),
                }
            )
    cpr = grid["local_structure_cpr"]
    for min_active in cpr["min_active_factors"]:
        for base_weight in cpr["base_weights"]:
            for factor_step in cpr["factor_steps"]:
                for max_weight in cpr["max_weights"]:
                    if float(max_weight) < float(base_weight):
                        continue
                    for conflict_threshold in cpr["conflict_thresholds"]:
                        for shield_cap in cpr["shield_caps"]:
                            for hcr_threshold in cpr["hcr_thresholds"]:
                                for trend_threshold in cpr["trend_thresholds"]:
                                    candidates.append(
                                        {
                                            "kind": "local_structure_cpr",
                                            "policy_id": (
                                                f"cpr_n{int(min_active)}"
                                                f"_w{float(base_weight):.2f}"
                                                f"_s{float(factor_step):.3f}"
                                                f"_max{float(max_weight):.2f}"
                                                f"_ct{float(conflict_threshold):.2f}"
                                                f"_cap{float(shield_cap):.3f}"
                                                f"_hcr{float(hcr_threshold):.2f}"
                                                f"_trend{float(trend_threshold):.2f}"
                                            ),
                                            "min_active": int(min_active),
                                            "weight": float(base_weight),
                                            "factor_step": float(factor_step),
                                            "max_weight": float(max_weight),
                                            "conflict_threshold": float(conflict_threshold),
                                            "shield_cap": float(shield_cap),
                                            "hcr_threshold": float(hcr_threshold),
                                            "trend_threshold": float(trend_threshold),
                                        }
                                    )
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if str(candidate["policy_id"]) in seen:
            continue
        seen.add(str(candidate["policy_id"]))
        unique.append(candidate)
    return unique


def reference_conflict(window: dict[str, object]) -> tuple[float, float]:
    return reviewer.reference_conflict(window)


def feature_value(window: dict[str, object], name: str) -> float:
    return reviewer.feature_value(window, name)


def local_factor_count(window: dict[str, object]) -> int:
    return reviewer.low_structure_factor_count(window)


def degeneracy_compatible(window: dict[str, object], policy: dict[str, object]) -> bool:
    return (
        feature_value(window, "horizon_context_ratio") <= finite_float(policy["hcr_threshold"])
        and feature_value(window, "trend_strength") >= finite_float(policy["trend_threshold"])
    )


def policy_decision(window: dict[str, object], policy: dict[str, object]) -> dict[str, object]:
    kind = str(policy["kind"])
    if kind == "no_repair":
        return {"gate": 0, "weight": 0.0, "reason": "no_repair", "score": 0.0, "shield": 0, "override": 0}
    if kind == "global_blend":
        return {
            "gate": 1,
            "weight": finite_float(policy["weight"]),
            "reason": "global_blend",
            "score": finite_float(policy["weight"]),
            "shield": 0,
            "override": 0,
        }
    outside_rate, outside_ratio = reference_conflict(window)
    if kind == "conflict_only":
        gate = int(outside_rate <= finite_float(policy["conflict_threshold"]))
        return {
            "gate": gate,
            "weight": finite_float(policy["weight"]) if gate else 0.0,
            "reason": "conflict_compatible",
            "score": outside_rate,
            "shield": 0,
            "override": 0,
            "reference_outside_interval_mean_ratio": outside_ratio,
        }
    if kind != "local_structure_cpr":
        raise ValueError(f"Unsupported policy kind: {kind}")
    count = local_factor_count(window)
    gate = int(count >= int(policy["min_active"]))
    if not gate:
        return {"gate": 0, "weight": 0.0, "reason": "local_score_below_threshold", "score": count, "shield": 0, "override": 0}
    raw_weight = finite_float(policy["weight"]) + max(0, count - int(policy["min_active"])) * finite_float(
        policy["factor_step"]
    )
    weight = min(finite_float(policy["max_weight"]), raw_weight)
    shield = 0
    override = 0
    if outside_rate >= finite_float(policy["conflict_threshold"]) and weight > finite_float(policy["shield_cap"]):
        if degeneracy_compatible(window, policy):
            override = 1
        else:
            weight = finite_float(policy["shield_cap"])
            shield = 1
    return {
        "gate": gate,
        "weight": weight,
        "reason": "local_structure_cpr",
        "score": count,
        "shield": shield,
        "override": override,
        "reference_outside_interval_mean_ratio": outside_ratio,
    }


def apply_policy_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
    config_hash: str,
) -> dict[str, object]:
    decision = policy_decision(window, policy)
    actual = np.asarray(window["actual"], dtype=float)
    model = np.asarray(window["model_forecast"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    repaired = model + finite_float(decision["weight"]) * (baseline - model)
    repair_mae = mae(actual, repaired)
    repair_rer = relative_error_ratio(repair_mae, finite_float(window["baseline_mae"]))
    outside_rate, outside_ratio = reference_conflict(window)
    return {
        "strategy_id": strategy_id,
        "split_protocol": split_protocol,
        "split_id": split_id,
        "config_hash": config_hash,
        "selected_policy_id": policy["policy_id"],
        "policy_class": policy["kind"],
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "gate_active": int(decision["gate"]),
        "gate_reason": decision["reason"],
        "gate_score": finite_float(decision["score"]),
        "effective_weight": finite_float(decision["weight"]),
        "shield_active": int(decision["shield"]),
        "conflict_override": int(decision["override"]),
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_ratio,
        "low_structure_factor_count": local_factor_count(window),
        "model_mae": window["model_mae"],
        "baseline_mae": window["baseline_mae"],
        "repair_mae": repair_mae,
        "mae_delta_vs_model": repair_mae - finite_float(window["model_mae"]),
        "model_relative_error_ratio": window["model_rer"],
        "repair_relative_error_ratio": repair_rer,
        "relative_error_ratio_delta": repair_rer - finite_float(window["model_rer"]),
        "model_failure_delta_005": window["model_failure"],
        "repair_failure_delta_005": int(repair_rer > 1.05),
        "repair_improves_model": int(repair_mae < finite_float(window["model_mae"])),
        "horizon_context_ratio": feature_value(window, "horizon_context_ratio"),
        "trend_strength": feature_value(window, "trend_strength"),
        "zero_ratio": feature_value(window, "zero_ratio"),
        "coefficient_of_variation": feature_value(window, "coefficient_of_variation"),
        "spectral_entropy": feature_value(window, "spectral_entropy"),
    }


def apply_policy(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
    config_hash: str,
) -> list[dict[str, object]]:
    return [apply_policy_to_window(window, policy, strategy_id, split_protocol, split_id, config_hash) for window in windows]


def metric_bundle(rows: list[dict[str, object]]) -> dict[str, float]:
    return reviewer.metric_bundle(rows)


def calibration_metrics(rows: list[dict[str, object]], weak_budget: float) -> dict[str, float]:
    overall = metric_bundle(rows)
    failure = metric_bundle([row for row in rows if row["role"] == "failure_target"])
    stress = metric_bundle([row for row in rows if row["role"] == "stress_target"])
    pc = metric_bundle([row for row in rows if row["role"] == "positive_control"])
    weak_rows = [row for row in rows if row["role"] == "weak_positive_control"]
    weak = metric_bundle(weak_rows) if weak_rows else {"p90_rer_delta": float("nan")}
    weak_p90 = finite_float(weak.get("p90_rer_delta"), float("nan"))
    secondary_ok = int(not math.isfinite(weak_p90) or weak_p90 <= weak_budget)
    utility = (
        5.0 * overall["failure_rate_reduction"]
        + 0.75 * failure["failure_rate_reduction"]
        + 0.75 * stress["failure_rate_reduction"]
        - 0.10 * overall["gate_rate"]
    )
    if not secondary_ok:
        utility -= 100.0
    return {
        "overall_reduction": overall["failure_rate_reduction"],
        "failure_target_reduction": failure["failure_rate_reduction"],
        "stress_target_reduction": stress["failure_rate_reduction"],
        "gate_rate": overall["gate_rate"],
        "positive_control_p90_rer_delta": pc["p90_rer_delta"],
        "positive_control_tail_harm_rate": rate_for_tail(rows, "positive_control"),
        "weak_positive_control_p90_rer_delta": weak_p90,
        "secondary_weak_control_ok": secondary_ok,
        "selection_utility": utility,
    }


def rate_for_tail(rows: list[dict[str, object]], role: str, tau: float = 0.05) -> float:
    subset = [row for row in rows if row["role"] == role]
    if not subset:
        return float("nan")
    return mean([1.0 if finite_float(row["relative_error_ratio_delta"]) > tau else 0.0 for row in subset])


def primary_losses(rows: list[dict[str, object]], tau: float, role: str) -> list[float]:
    subset = [row for row in rows if row["role"] == role]
    if not subset:
        raise ValueError(f"No calibration rows for role {role}")
    return [1.0 if finite_float(row["relative_error_ratio_delta"]) > tau else 0.0 for row in subset]


def split_values(windows: list[dict[str, object]], split_key: str) -> list[str]:
    return sorted({str(window[split_key]) for window in windows})


def run_split_protocol(
    windows: list[dict[str, object]],
    split_key: str,
    policies: list[dict[str, object]],
    config: dict[str, object],
    config_hash: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    risk = config["risk"]
    tau = finite_float(risk["tau_pc_rer_delta"])
    alpha = finite_float(risk["alpha_pc_tail_harm"])
    delta = finite_float(risk["delta"])
    correction = str(risk["correction"])
    primary_role = str(risk["primary_role"])
    weak_budget = finite_float(risk["weak_positive_control_p90_budget"])
    split_protocol = f"leave_{split_key}"
    strategy_id = f"cpr_ltt_leave_{split_key}"
    window_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    policy_by_id = {str(policy["policy_id"]): policy for policy in policies}
    for value in split_values(windows, split_key):
        split_id = f"holdout_{split_key}:{value}"
        train = [window for window in windows if str(window[split_key]) != value]
        test = [window for window in windows if str(window[split_key]) == value]
        applied_by_policy: dict[str, list[dict[str, object]]] = {}
        losses_by_policy: dict[str, list[float]] = {}
        utilities: dict[str, float] = {}
        metrics_by_policy: dict[str, dict[str, float]] = {}
        for policy in policies:
            policy_id = str(policy["policy_id"])
            rows = apply_policy(train, policy, "calibration", split_protocol, split_id, config_hash)
            applied_by_policy[policy_id] = rows
            losses_by_policy[policy_id] = primary_losses(rows, tau, primary_role)
            metrics = calibration_metrics(rows, weak_budget)
            metrics_by_policy[policy_id] = metrics
            utilities[policy_id] = metrics["selection_utility"]
        tests = ltt_risk_tests(
            losses_by_policy,
            alpha=alpha,
            delta=delta,
            correction=correction,
            binary=True,
        )
        selected_policy_id = select_highest_utility_certified(tests, utilities, fallback_policy_id="no_repair")
        selected_policy = policy_by_id[selected_policy_id]
        test_lookup = {test.policy_id: test for test in tests}
        for test_result in tests:
            metrics = metrics_by_policy[test_result.policy_id]
            policy = policy_by_id[test_result.policy_id]
            calibration_rows.append(
                {
                    "config_hash": config_hash,
                    "strategy_id": strategy_id,
                    "split_protocol": split_protocol,
                    "split_id": split_id,
                    "candidate_policy_id": test_result.policy_id,
                    "policy_class": policy["kind"],
                    "ltt_accepted": int(test_result.accepted),
                    "risk_count": test_result.risk_count,
                    "risk_n": test_result.n,
                    "empirical_risk": test_result.empirical_risk,
                    "risk_alpha": test_result.alpha,
                    "p_value": test_result.p_value,
                    "corrected_threshold": test_result.corrected_threshold,
                    "correction": test_result.correction,
                    "ucb_hoeffding": test_result.ucb_hoeffding,
                    **metrics,
                    **policy_params_for_row(policy),
                }
            )
        selected_test = test_lookup[selected_policy_id]
        selected_metrics = metrics_by_policy[selected_policy_id]
        selected_rows.append(
            {
                "config_hash": config_hash,
                "strategy_id": strategy_id,
                "split_protocol": split_protocol,
                "split_id": split_id,
                "selected_policy_id": selected_policy_id,
                "policy_class": selected_policy["kind"],
                "selected_ltt_accepted": int(selected_test.accepted),
                "selected_risk_count": selected_test.risk_count,
                "selected_risk_n": selected_test.n,
                "selected_empirical_risk": selected_test.empirical_risk,
                "selected_p_value": selected_test.p_value,
                "selected_corrected_threshold": selected_test.corrected_threshold,
                "selected_utility": utilities[selected_policy_id],
                "n_certified_candidates": sum(1 for item in tests if item.accepted),
                "n_candidates": len(tests),
                "train_n_windows": len(train),
                "test_n_windows": len(test),
                **{f"selected_{key}": value for key, value in policy_params_for_row(selected_policy).items()},
                **{f"calibration_{key}": value for key, value in selected_metrics.items()},
            }
        )
        window_rows.extend(apply_policy(test, selected_policy, strategy_id, split_protocol, split_id, config_hash))
    return window_rows, calibration_rows, selected_rows


def policy_params_for_row(policy: dict[str, object]) -> dict[str, object]:
    keys = [
        "kind",
        "weight",
        "min_active",
        "factor_step",
        "max_weight",
        "conflict_threshold",
        "shield_cap",
        "hcr_threshold",
        "trend_threshold",
    ]
    return {key: policy[key] for key in keys if key in policy}


def groupby_map(rows: list[dict[str, object]], keys: list[str]) -> dict[object, list[dict[str, object]]]:
    groups: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        key_values = tuple(str(row.get(key, "")) for key in keys)
        key: object = key_values[0] if len(key_values) == 1 else key_values
        groups.setdefault(key, []).append(row)
    return groups


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return reviewer.build_summary(rows)


def selection_stability(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (strategy_id, split_protocol), group_rows in sorted(groupby_map(selected_rows, ["strategy_id", "split_protocol"]).items()):
        policies = [str(row["selected_policy_id"]) for row in group_rows]
        counter = Counter(policies)
        most_common, count = counter.most_common(1)[0]
        certified_counts = [finite_float(row["n_certified_candidates"]) for row in group_rows]
        rows.append(
            {
                "strategy_id": strategy_id,
                "split_protocol": split_protocol,
                "n_splits": len(group_rows),
                "n_candidates": int(finite_float(group_rows[0]["n_candidates"])),
                "mean_certified_candidates": mean(certified_counts),
                "unique_selected_policy_count": len(counter),
                "most_common_policy": most_common,
                "most_common_policy_share": count / len(group_rows),
                "no_repair_selection_rate": mean([1.0 if policy == "no_repair" else 0.0 for policy in policies]),
                "mean_selected_empirical_risk": mean([finite_float(row["selected_empirical_risk"]) for row in group_rows]),
                "mean_selected_p_value": mean([finite_float(row["selected_p_value"]) for row in group_rows]),
            }
        )
    return rows


def split_audit(window_rows: list[dict[str, object]], selected_rows: list[dict[str, object]]) -> dict[str, object]:
    source_rows = [row for row in window_rows if row["split_protocol"] == "leave_source"]
    family_rows = [row for row in window_rows if row["split_protocol"] == "leave_family"]
    natural_keys = ["family", "source", "role", "dataset", "model", "series_id", "origin", "window_index"]
    output_fields = [
        "gate_active",
        "shield_active",
        "effective_weight",
        "repair_mae",
        "repair_relative_error_ratio",
        "repair_failure_delta_005",
        "relative_error_ratio_delta",
    ]

    def key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(row.get(name, "")) for name in natural_keys)

    source_by_key = {key(row): row for row in source_rows}
    family_by_key = {key(row): row for row in family_rows}
    matched_keys = sorted(set(source_by_key).intersection(family_by_key))

    policy_different = 0
    output_different = 0
    for item_key in matched_keys:
        source = source_by_key[item_key]
        family = family_by_key[item_key]
        if source.get("selected_policy_id") != family.get("selected_policy_id"):
            policy_different += 1
        if any(str(source.get(field, "")) != str(family.get(field, "")) for field in output_fields):
            output_different += 1

    selected_by_protocol = groupby_map(selected_rows, ["split_protocol"])
    return {
        "source_split_count": len({row["split_id"] for row in source_rows}),
        "family_split_count": len({row["split_id"] for row in family_rows}),
        "source_selected_rows": len(selected_by_protocol.get("leave_source", [])),
        "family_selected_rows": len(selected_by_protocol.get("leave_family", [])),
        "source_unique_policies": len({row["selected_policy_id"] for row in selected_by_protocol.get("leave_source", [])}),
        "family_unique_policies": len({row["selected_policy_id"] for row in selected_by_protocol.get("leave_family", [])}),
        "matched_windows": len(matched_keys),
        "unmatched_source_windows": len(set(source_by_key) - set(family_by_key)),
        "unmatched_family_windows": len(set(family_by_key) - set(source_by_key)),
        "policy_different_windows": policy_different,
        "realized_output_different_windows": output_different,
        "output_fields": ", ".join(output_fields),
    }


def fmt_pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def fmt_num(value: object, digits: int = 3) -> str:
    return f"{finite_float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def row_lookup(rows: list[dict[str, object]], strategy: str, split: str, group: str) -> dict[str, object]:
    return next(row for row in rows if row["strategy_id"] == strategy and row["split_protocol"] == split and row["group"] == group)


def comparison_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    reviewer_summary_path = OUT_DIR / "reviewer_baseline_summary.csv"
    baseline_rows: list[dict[str, object]] = read_csv(reviewer_summary_path) if reviewer_summary_path.exists() else []
    merged: list[dict[str, object]] = []
    for split in ["leave_source", "leave_family"]:
        strategy_ids = [
            ("cpr_ltt_leave_source" if split == "leave_source" else "cpr_ltt_leave_family", summary),
            ("fixed_rr_cssr_trend_decay_adjudicator", baseline_rows),
            ("validation_tuned_global_blend", baseline_rows),
            ("conflict_only_selector", baseline_rows),
            ("learned_logistic_gate", baseline_rows),
            ("learned_stump_gate", baseline_rows),
        ]
        for strategy, rows in strategy_ids:
            if not rows:
                continue
            row = row_lookup(rows, strategy, split, "overall")
            pc = row_lookup(rows, strategy, split, "role:positive_control")
            weak = next(
                (
                    item
                    for item in rows
                    if item["strategy_id"] == strategy
                    and item["split_protocol"] == split
                    and item["group"] == "role:weak_positive_control"
                ),
                {},
            )
            merged.append(
                {
                    "Split": split,
                    "Strategy": strategy,
                    "Reduction": fmt_pct(row["failure_rate_reduction"]),
                    "95% CI": f"[{fmt_pct(row['failure_reduction_ci_low'])}, {fmt_pct(row['failure_reduction_ci_high'])}]",
                    "PC p90": fmt_num(pc.get("p90_rer_delta", float("nan"))),
                    "Weak-PC p90": fmt_num(weak.get("p90_rer_delta", float("nan"))),
                    "Gate": fmt_pct(row["gate_rate"]),
                    "McNemar p": fmt_num(row["mcnemar_exact_p"], 4),
                }
            )
    return merged


def write_report(
    config: dict[str, object],
    config_hash: str,
    summary: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    stability: list[dict[str, object]],
    audit: dict[str, object],
) -> None:
    compare = comparison_rows(summary)
    stability_table = [
        {
            "Split": row["split_protocol"],
            "Candidates": int(row["n_candidates"]),
            "Certified": fmt_num(row["mean_certified_candidates"], 1),
            "Unique": int(row["unique_selected_policy_count"]),
            "Top": row["most_common_policy"],
            "Top share": fmt_pct(row["most_common_policy_share"]),
            "No repair": fmt_pct(row["no_repair_selection_rate"]),
            "Risk": fmt_num(row["mean_selected_empirical_risk"], 3),
        }
        for row in stability
    ]
    selected_table = [
        {
            "Split": row["split_id"],
            "Policy": row["selected_policy_id"],
            "Accepted": row["selected_ltt_accepted"],
            "k/n": f"{row['selected_risk_count']}/{row['selected_risk_n']}",
            "p": fmt_num(row["selected_p_value"], 5),
            "thr": fmt_num(row["selected_corrected_threshold"], 5),
            "Certified": row["n_certified_candidates"],
        }
        for row in selected_rows
    ]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Conformal Policy Repair Report",
                "",
                "## Method Claim",
                "",
                "Conformal Policy Repair (CPR) reframes the previous router/shield/baseline mixture as a fixed policy class plus Learn-Then-Test risk calibration. The algorithm does not tune thresholds on held-out test windows: it first certifies candidate policies on independent calibration windows, then selects the highest-utility certified policy.",
                "",
                "## Fixed Protocol",
                "",
                f"- Config: `configs/conformal_policy_repair_grid.json`",
                f"- Config hash: `{config_hash}`",
                f"- Candidate count: `{len(candidate_policies(config))}`",
                f"- Primary risk: `P(dRER > {config['risk']['tau_pc_rer_delta']}) <= {config['risk']['alpha_pc_tail_harm']}` on `{config['risk']['primary_role']}` windows.",
                f"- Family-wise error: `{config['risk']['delta']}` with `{config['risk']['correction']}` correction.",
                "",
                "## Matched Comparison",
                "",
                markdown_table(
                    compare,
                    [
                        ("Split", "Split"),
                        ("Strategy", "Strategy"),
                        ("Reduction", "Failure Reduction"),
                        ("95% CI", "95% CI"),
                        ("PC p90", "PC p90 dRER"),
                        ("Weak-PC p90", "Weak-PC p90"),
                        ("Gate", "Gate"),
                        ("McNemar p", "McNemar p"),
                    ],
                ),
                "",
                "## LTT Selection Stability",
                "",
                markdown_table(
                    stability_table,
                    [
                        ("Split", "Split"),
                        ("Candidates", "Candidates"),
                        ("Certified", "Mean Certified"),
                        ("Unique", "Unique Selected"),
                        ("Top", "Most Common"),
                        ("Top share", "Top Share"),
                        ("No repair", "No-Repair Rate"),
                        ("Risk", "Mean Calib Risk"),
                    ],
                ),
                "",
                "## Split Audit",
                "",
                "The leave-source and leave-family protocols are run as separate LTT calibration problems. The identical CPR aggregate rows above should not be read as a cached-table shortcut or as proof of strong leave-family transfer.",
                "",
                markdown_table(
                    [
                        {
                            "Check": "leave_source splits",
                            "Value": audit["source_split_count"],
                        },
                        {
                            "Check": "leave_family splits",
                            "Value": audit["family_split_count"],
                        },
                        {
                            "Check": "matched source/family windows",
                            "Value": audit["matched_windows"],
                        },
                        {
                            "Check": "windows with different selected policy id",
                            "Value": audit["policy_different_windows"],
                        },
                        {
                            "Check": "windows with different realized repair output",
                            "Value": audit["realized_output_different_windows"],
                        },
                    ],
                    [("Check", "Check"), ("Value", "Value")],
                ),
                "",
                f"Audit note: the current slice has `{audit['policy_different_windows']}` matched windows where source/family calibration selected different policy IDs, but `{audit['realized_output_different_windows']}` windows where the realized gate, effective weight, repair MAE/RER, or failure flag changed. This means the table equality is a realized-output equivalence on this slice, not a license to claim leave-family calibration is solved.",
                "",
                "## Selected Policies",
                "",
                markdown_table(
                    selected_table,
                    [
                        ("Split", "Split"),
                        ("Policy", "Policy"),
                        ("Accepted", "Accepted"),
                        ("k/n", "Risk k/n"),
                        ("p", "p-value"),
                        ("thr", "Holm threshold"),
                        ("Certified", "Certified Count"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- If CPR keeps the RR-CSSR point-error gain while certifying positive-control tail risk on calibration splits, the method contribution becomes a risk-controlled repair protocol rather than a post-hoc fallback.",
                "- If CPR selects no-repair on some splits, that is not a failed run; it identifies where the current calibration slice is too small or too risky to certify aggressive repair.",
                "- WQL and coverage remain robustness-only unless full quantile/sample artifacts are regenerated; this method report only claims point-error failure repair under the declared MAE-RER risk.",
                "",
                "## Artifacts",
                "",
                "- `results/aaai_stress/cpr_ltt_windows.csv`",
                "- `results/aaai_stress/cpr_ltt_summary.csv`",
                "- `results/aaai_stress/cpr_ltt_calibration_tests.csv`",
                "- `results/aaai_stress/cpr_ltt_selected_policies.csv`",
                "- `results/aaai_stress/cpr_ltt_selection_stability.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, config_hash = load_config()
    policies = candidate_policies(config)
    windows, _ = cross.load_cross_family_windows()
    window_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for split_key in ["source", "family"]:
        rows, cal_rows, sel_rows = run_split_protocol(windows, split_key, policies, config, config_hash)
        window_rows.extend(rows)
        calibration_rows.extend(cal_rows)
        selected_rows.extend(sel_rows)
    summary = build_summary(window_rows)
    stability = selection_stability(selected_rows)
    audit = split_audit(window_rows, selected_rows)
    write_csv(OUT_DIR / "cpr_ltt_windows.csv", window_rows)
    write_csv(OUT_DIR / "cpr_ltt_summary.csv", summary)
    write_csv(OUT_DIR / "cpr_ltt_calibration_tests.csv", calibration_rows)
    write_csv(OUT_DIR / "cpr_ltt_selected_policies.csv", selected_rows)
    write_csv(OUT_DIR / "cpr_ltt_selection_stability.csv", stability)
    write_report(config, config_hash, summary, selected_rows, stability, audit)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_hash": config_hash,
        "n_windows": len(windows),
        "n_candidates": len(policies),
        "n_window_rows": len(window_rows),
        "n_calibration_test_rows": len(calibration_rows),
        "n_selected_rows": len(selected_rows),
        "split_audit": audit,
        "summary": "results/aaai_stress/cpr_ltt_summary.csv",
        "windows": "results/aaai_stress/cpr_ltt_windows.csv",
        "calibration_tests": "results/aaai_stress/cpr_ltt_calibration_tests.csv",
        "selected_policies": "results/aaai_stress/cpr_ltt_selected_policies.csv",
        "selection_stability": "results/aaai_stress/cpr_ltt_selection_stability.csv",
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
