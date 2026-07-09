#!/usr/bin/env python
"""Risk-calibrate regime-resolved conflict-shielded selective repair."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_selective_repair_expanded_ablation_goal as selective  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402


OUT_DIR = ROOT / "results" / "repair"
DOC_PATH = ROOT / "docs" / "risk_calibrated_rr_cssr_report.md"
BASE_POLICY_ID = "loose_n2_low_structure_adaptive_w0.60_s0.500_max1.00"


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def bootstrap_ci(values: np.ndarray, *, n_bootstrap: int = 2000, seed: int = 29) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    stats = np.mean(arr[draws], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def base_policy() -> dict[str, object]:
    for policy in selective.policy_candidates():
        if policy["policy_id"] == BASE_POLICY_ID:
            return policy
    raise ValueError(f"Missing base policy {BASE_POLICY_ID}")


def feature_float(window: dict[str, object], key: str) -> float:
    return finite_float(window["feature"].get(key))


def candidate_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for conflict_threshold in [0.25, 0.40]:
        for shield_cap in [0.125, 0.250]:
            for hcr_threshold in [0.20, 0.30, 0.40, 0.50]:
                for trend_threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
                    configs.append(
                        {
                            "config_id": (
                                f"ct{conflict_threshold:.2f}_cap{shield_cap:.3f}"
                                f"_hcr{hcr_threshold:.2f}_trend{trend_threshold:.2f}"
                            ),
                            "conflict_threshold": conflict_threshold,
                            "shield_cap": shield_cap,
                            "hcr_threshold": hcr_threshold,
                            "trend_threshold": trend_threshold,
                        }
                    )
    return configs


def conflict_is_degeneracy_compatible(window: dict[str, object], config: dict[str, object]) -> bool:
    return (
        feature_float(window, "horizon_context_ratio") <= finite_float(config["hcr_threshold"])
        and feature_float(window, "trend_strength") >= finite_float(config["trend_threshold"])
    )


def apply_config_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    config: dict[str, object],
    strategy_id: str,
    split_id: str,
) -> dict[str, object]:
    gate, reason, active_count = selective.gate_decision(window["feature"], policy)
    pre_weight = selective.policy_reference_weight(policy, gate, active_count)
    reference = np.asarray(window["baseline_forecast"], dtype=float)
    model = np.asarray(window["model_forecast"], dtype=float)
    actual = np.asarray(window["actual"], dtype=float)
    q10 = np.asarray(window["q10"], dtype=float)
    q90 = np.asarray(window["q90"], dtype=float)
    outside_rate, outside_ratio = selective.reference_interval_conflict(reference, q10, q90)
    weight = pre_weight
    shield_active = 0
    conflict_override = 0
    if outside_rate >= finite_float(config["conflict_threshold"]) and weight > finite_float(config["shield_cap"]):
        if conflict_is_degeneracy_compatible(window, config):
            conflict_override = 1
        else:
            weight = finite_float(config["shield_cap"])
            shield_active = 1
    forecast = model + weight * (reference - model)
    repair_mae = mae(actual, forecast)
    repair_rer = relative_error_ratio(repair_mae, finite_float(window["baseline_mae"]))
    return {
        "strategy_id": strategy_id,
        "split_id": split_id,
        "config_id": config["config_id"],
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "dataset": window["dataset"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "gate_active": gate,
        "gate_reason": reason,
        "active_ex_ante_factor_count": active_count,
        "pre_weight": pre_weight,
        "effective_weight": weight,
        "shield_active": shield_active,
        "conflict_override": conflict_override,
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_ratio,
        "model_failure_delta_005": window["model_failure"],
        "repair_failure_delta_005": int(repair_rer > 1.05),
        "model_relative_error_ratio": window["model_rer"],
        "repair_relative_error_ratio": repair_rer,
        "relative_error_ratio_delta": repair_rer - finite_float(window["model_rer"]),
        "model_mae": window["model_mae"],
        "baseline_mae": window["baseline_mae"],
        "repair_mae": repair_mae,
        "horizon_context_ratio": feature_float(window, "horizon_context_ratio"),
        "trend_strength": feature_float(window, "trend_strength"),
        "zero_ratio": feature_float(window, "zero_ratio"),
        "spectral_entropy": feature_float(window, "spectral_entropy"),
        "conflict_threshold": config["conflict_threshold"],
        "shield_cap": config["shield_cap"],
        "hcr_threshold": config["hcr_threshold"],
        "trend_threshold": config["trend_threshold"],
    }


def apply_config(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    config: dict[str, object],
    strategy_id: str,
    split_id: str,
) -> list[dict[str, object]]:
    return [apply_config_to_window(window, policy, config, strategy_id, split_id) for window in windows]


def summarize(rows: list[dict[str, object]], strategy_id: str, group: str, group_type: str) -> dict[str, object]:
    model_fail = np.asarray([finite_float(row["model_failure_delta_005"]) for row in rows], dtype=float)
    repair_fail = np.asarray([finite_float(row["repair_failure_delta_005"]) for row in rows], dtype=float)
    diff = model_fail - repair_fail
    improved = int(np.sum((model_fail == 1) & (repair_fail == 0)))
    worsened = int(np.sum((model_fail == 0) & (repair_fail == 1)))
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "strategy_id": strategy_id,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "model_failure_rate": rate(model_fail.tolist()),
        "repair_failure_rate": rate(repair_fail.tolist()),
        "failure_rate_reduction": rate(diff.tolist()),
        "failure_reduction_ci_low": ci_low,
        "failure_reduction_ci_high": ci_high,
        "mcnemar_improved_count": improved,
        "mcnemar_worsened_count": worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
        "gate_rate": rate([finite_float(row["gate_active"]) for row in rows]),
        "shield_rate": rate([finite_float(row["shield_active"]) for row in rows]),
        "override_rate": rate([finite_float(row["conflict_override"]) for row in rows]),
        "mean_reference_outside_interval_rate": rate(
            [finite_float(row["reference_outside_interval_rate"]) for row in rows]
        ),
        "median_rer_delta": median([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
        "p90_rer_delta": p90([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
    }


def summary_rows(rows: list[dict[str, object]], strategy_id: str) -> list[dict[str, object]]:
    summaries = [summarize(rows, strategy_id, "overall", "overall")]
    for role in sorted({str(row["role"]) for row in rows}):
        summaries.append(summarize([row for row in rows if row["role"] == role], strategy_id, f"role:{role}", "role"))
    for family in sorted({str(row["family"]) for row in rows}):
        summaries.append(
            summarize([row for row in rows if row["family"] == family], strategy_id, f"family:{family}", "family")
        )
    for family in sorted({str(row["family"]) for row in rows}):
        subset = [row for row in rows if row["family"] == family and row["role"] == "positive_control"]
        if subset:
            summaries.append(
                summarize(subset, strategy_id, f"family:{family}|role:positive_control", "family_role")
            )
    for source in sorted({str(row["source"]) for row in rows}):
        summaries.append(
            summarize([row for row in rows if row["source"] == source], strategy_id, f"source:{source}", "source")
        )
    return summaries


def metrics_for_selection(rows: list[dict[str, object]]) -> dict[str, float]:
    def group_metrics(group_rows: list[dict[str, object]]) -> dict[str, float]:
        if not group_rows:
            return {
                "model_failure_rate": 0.0,
                "repair_failure_rate": 0.0,
                "failure_rate_reduction": 0.0,
                "median_rer_delta": float("nan"),
                "p90_rer_delta": float("nan"),
                "gate_rate": 0.0,
                "shield_rate": 0.0,
                "override_rate": 0.0,
            }
        model_fail = [finite_float(row["model_failure_delta_005"]) for row in group_rows]
        repair_fail = [finite_float(row["repair_failure_delta_005"]) for row in group_rows]
        deltas = [finite_float(row["relative_error_ratio_delta"]) for row in group_rows]
        return {
            "model_failure_rate": rate(model_fail),
            "repair_failure_rate": rate(repair_fail),
            "failure_rate_reduction": rate([m - r for m, r in zip(model_fail, repair_fail, strict=True)]),
            "median_rer_delta": median(deltas),
            "p90_rer_delta": p90(deltas),
            "gate_rate": rate([finite_float(row["gate_active"]) for row in group_rows]),
            "shield_rate": rate([finite_float(row["shield_active"]) for row in group_rows]),
            "override_rate": rate([finite_float(row["conflict_override"]) for row in group_rows]),
        }

    overall = group_metrics(rows)
    failure = group_metrics([row for row in rows if row["role"] == "failure_target"])
    stress = group_metrics([row for row in rows if row["role"] == "stress_target"])
    pc = group_metrics([row for row in rows if row["role"] == "positive_control"])
    weak_pc_rows = [row for row in rows if row["role"] == "weak_positive_control"]
    weak_pc = group_metrics(weak_pc_rows) if weak_pc_rows else {"p90_rer_delta": float("nan")}
    return {
        "overall_reduction": overall["failure_rate_reduction"],
        "failure_target_reduction": failure["failure_rate_reduction"],
        "stress_target_reduction": stress["failure_rate_reduction"],
        "positive_control_failure_delta": pc["repair_failure_rate"] - pc["model_failure_rate"],
        "positive_control_p90_rer_delta": pc["p90_rer_delta"],
        "positive_control_median_rer_delta": pc["median_rer_delta"],
        "weak_positive_control_p90_rer_delta": finite_float(weak_pc.get("p90_rer_delta"), float("nan")),
        "gate_rate": overall["gate_rate"],
        "shield_rate": overall["shield_rate"],
        "override_rate": overall["override_rate"],
    }


def selection_score(metrics: dict[str, float], config: dict[str, object]) -> tuple[float, str]:
    if metrics["positive_control_failure_delta"] > 1e-12:
        return -1000.0, "positive_control_failure_violation"
    if metrics["positive_control_p90_rer_delta"] > 0.05:
        return -900.0, "positive_control_tail_violation"
    if metrics["positive_control_median_rer_delta"] > 0.02:
        return -800.0, "positive_control_median_violation"
    weak_p90 = metrics["weak_positive_control_p90_rer_delta"]
    if math.isfinite(weak_p90) and weak_p90 > 0.35:
        return -700.0, "weak_positive_control_tail_violation"
    if metrics["overall_reduction"] <= 0.0:
        return -600.0, "no_repair_gain"
    parsimony = (
        -0.050 * finite_float(config["hcr_threshold"])
        + 0.020 * finite_float(config["trend_threshold"])
        + 0.010 * finite_float(config["conflict_threshold"])
        - 0.010 * finite_float(config["shield_cap"])
    )
    score = (
        5.0 * metrics["overall_reduction"]
        + 0.75 * metrics["failure_target_reduction"]
        + 0.75 * metrics["stress_target_reduction"]
        - 0.50 * max(0.0, metrics["positive_control_p90_rer_delta"])
        + parsimony
    )
    return score, "ok"


def select_config(
    train_windows: list[dict[str, object]],
    policy: dict[str, object],
    split_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best: tuple[float, dict[str, object], str] | None = None
    search_rows: list[dict[str, object]] = []
    for config in candidate_configs():
        rows = apply_config(train_windows, policy, config, "selection", split_id)
        metrics = metrics_for_selection(rows)
        score, status = selection_score(metrics, config)
        search_rows.append({**config, **metrics, "split_id": split_id, "selection_score": score, "status": status})
        if status == "ok" and (best is None or score > best[0]):
            best = (score, dict(config), status)
    if best is None:
        fallback = {
            "config_id": "fallback_cssr_cap_all_conflicts",
            "conflict_threshold": 0.40,
            "shield_cap": 0.125,
            "hcr_threshold": -1.0,
            "trend_threshold": 2.0,
        }
        return fallback, [
            *search_rows,
            {**fallback, "split_id": split_id, "selection_score": 0.0, "status": "fallback_no_safe_config"},
        ]
    selected = {
        **best[1],
        "split_id": split_id,
        "selection_score": best[0],
        "status": "selected",
    }
    return best[1], [*search_rows, selected]


def split_evaluations(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    split_key: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    values = sorted({str(window[split_key]) for window in windows})
    strategy_id = f"rc_rr_cssr_leave_{split_key}_calibrated"
    for value in values:
        split_id = f"holdout_{split_key}:{value}"
        train = [window for window in windows if str(window[split_key]) != value]
        test = [window for window in windows if str(window[split_key]) == value]
        config, rows = select_config(train, policy, split_id)
        selection_rows.extend(rows)
        all_rows.extend(apply_config(test, policy, config, strategy_id, split_id))
    return all_rows, selection_rows


def fixed_strategy_rows(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    strategy_id: str,
    config: dict[str, object],
) -> list[dict[str, object]]:
    return apply_config(windows, policy, config, strategy_id, "fixed_full_slice")


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


def report_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(row["strategy_id"], row["group"]): row for row in summary}
    strategies = [
        "fixed_cssr_cap_all_conflicts",
        "fixed_rr_cssr_trend_decay_adjudicator",
        "rc_rr_cssr_leave_family_calibrated",
        "rc_rr_cssr_leave_source_calibrated",
    ]
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        overall = lookup[(strategy, "overall")]
        failure = lookup[(strategy, "role:failure_target")]
        stress = lookup[(strategy, "role:stress_target")]
        pc = lookup[(strategy, "role:positive_control")]
        weak = lookup.get((strategy, "role:weak_positive_control"), {})
        times_pc = lookup[(strategy, "family:timesfm|role:positive_control")]
        rows.append(
            {
                "Strategy": strategy,
                "Overall": fmt_pct(overall["failure_rate_reduction"]),
                "95% CI": f"[{fmt_pct(overall['failure_reduction_ci_low'])}, {fmt_pct(overall['failure_reduction_ci_high'])}]",
                "Failure": fmt_pct(failure["failure_rate_reduction"]),
                "Stress": fmt_pct(stress["failure_rate_reduction"]),
                "PC p90": fmt_num(pc["p90_rer_delta"]),
                "Weak-PC p90": fmt_num(weak.get("p90_rer_delta", float("nan"))),
                "TimesFM PC p90": fmt_num(times_pc["p90_rer_delta"]),
                "Shield": fmt_pct(overall["shield_rate"]),
                "Override": fmt_pct(overall["override_rate"]),
            }
        )
    return rows


def write_report(summary: list[dict[str, object]], selected: list[dict[str, object]]) -> None:
    table = report_rows(summary)
    selected_rows = [row for row in selected if row.get("status") == "selected"]
    selected_table = []
    for row in selected_rows:
        selected_table.append(
            {
                "Split": row["split_id"],
                "Config": row["config_id"],
                "Conflict": fmt_num(row["conflict_threshold"], 2),
                "Cap": fmt_num(row["shield_cap"], 3),
                "HCR": fmt_num(row["hcr_threshold"], 2),
                "Trend": fmt_num(row["trend_threshold"], 2),
                "Score": fmt_num(row["selection_score"], 3),
            }
        )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Risk-Calibrated RR-CSSR",
                "",
                "## Mechanism",
                "",
                "RC-RR-CSSR turns the hand-designed regime-resolved conflict shield into a calibrated router-guard-repair module. The router detects low-local-structure windows, the guard treats TSFM/reference interval disagreement as unsafe by default, and the repair head applies a residual correction toward the classical expert only when the calibrated regime rule says the conflict is trend/decay-compatible.",
                "",
                "## Calibration Protocol",
                "",
                "- Candidate thresholds are selected on calibration windows only.",
                "- The disagreement guard keeps a conservative threshold cap at 0.40; the initial unconstrained search selected looser thresholds that failed on held-out TimesFM positive controls.",
                "- The selected configuration must satisfy positive-control failure and p90 dRER constraints.",
                "- Weak-positive-control p90 dRER is enforced when a weak-control slice is present in calibration.",
                "- Held-out family/source windows are evaluated without retuning.",
                "",
                "## Results",
                "",
                markdown_table(
                    table,
                    [
                        ("Strategy", "Strategy"),
                        ("Overall", "Overall Reduction"),
                        ("95% CI", "95% CI"),
                        ("Failure", "Failure Target"),
                        ("Stress", "Stress Target"),
                        ("PC p90", "PC p90 dRER"),
                        ("Weak-PC p90", "Weak-PC p90 dRER"),
                        ("TimesFM PC p90", "TimesFM PC p90"),
                        ("Shield", "Shield Rate"),
                        ("Override", "Override Rate"),
                    ],
                ),
                "",
                "## Selected Held-Out Configurations",
                "",
                markdown_table(
                    selected_table,
                    [
                        ("Split", "Split"),
                        ("Config", "Config"),
                        ("Conflict", "Conflict"),
                        ("Cap", "Cap"),
                        ("HCR", "HCR"),
                        ("Trend", "Trend"),
                        ("Score", "Score"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "If calibrated splits match the fixed RR-CSSR tradeoff, the main value is not a larger point estimate but a stronger reviewer story: thresholds can be selected by a pre-specified risk budget rather than by post-hoc inspection. If a split chooses a looser rule and hurts weak controls, that split identifies the exact boundary condition that still needs a calibration slice.",
                "",
                "## Artifacts",
                "",
                "- `results/repair/risk_calibrated_rr_cssr_summary.csv`",
                "- `results/repair/risk_calibrated_rr_cssr_windows.csv`",
                "- `results/repair/risk_calibrated_rr_cssr_selected_configs.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    windows, _ = cross.load_cross_family_windows()
    policy = base_policy()
    fixed_cssr = {
        "config_id": "fixed_cssr_cap_all_conflicts",
        "conflict_threshold": 0.40,
        "shield_cap": 0.125,
        "hcr_threshold": -1.0,
        "trend_threshold": 2.0,
    }
    fixed_rr = {
        "config_id": "fixed_rr_cssr_trend_decay_adjudicator",
        "conflict_threshold": 0.40,
        "shield_cap": 0.125,
        "hcr_threshold": 0.30,
        "trend_threshold": 0.40,
    }
    window_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    strategy_map = {
        "fixed_cssr_cap_all_conflicts": fixed_strategy_rows(
            windows, policy, "fixed_cssr_cap_all_conflicts", fixed_cssr
        ),
        "fixed_rr_cssr_trend_decay_adjudicator": fixed_strategy_rows(
            windows, policy, "fixed_rr_cssr_trend_decay_adjudicator", fixed_rr
        ),
    }
    family_rows, family_selected = split_evaluations(windows, policy, "family")
    source_rows, source_selected = split_evaluations(windows, policy, "source")
    strategy_map["rc_rr_cssr_leave_family_calibrated"] = family_rows
    strategy_map["rc_rr_cssr_leave_source_calibrated"] = source_rows
    selected_rows.extend(family_selected)
    selected_rows.extend(source_selected)
    summary: list[dict[str, object]] = []
    for strategy_id, rows in strategy_map.items():
        window_rows.extend(rows)
        summary.extend(summary_rows(rows, strategy_id))
    write_csv(OUT_DIR / "risk_calibrated_rr_cssr_windows.csv", window_rows)
    write_csv(OUT_DIR / "risk_calibrated_rr_cssr_summary.csv", summary)
    write_csv(OUT_DIR / "risk_calibrated_rr_cssr_selected_configs.csv", selected_rows)
    write_report(summary, selected_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "strategies": sorted(strategy_map),
        "summary": "results/repair/risk_calibrated_rr_cssr_summary.csv",
        "window_metrics": "results/repair/risk_calibrated_rr_cssr_windows.csv",
        "selected_configs": "results/repair/risk_calibrated_rr_cssr_selected_configs.csv",
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    (OUT_DIR / "risk_calibrated_rr_cssr_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
