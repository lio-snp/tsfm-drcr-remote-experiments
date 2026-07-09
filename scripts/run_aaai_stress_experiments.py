#!/usr/bin/env python
"""Run AAAI-facing stress analyses for repair, gates, and claim hygiene."""

from __future__ import annotations

import csv
import contextlib
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_selective_repair_expanded_ablation_goal as selective  # noqa: E402


OUT_DIR = ROOT / "results" / "aaai_stress"
FIG_DIR = ROOT / "figures" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "aaai_stress_experiment_report.md"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

STRATEGY_METRICS = ROOT / "results" / "repair" / "cross_family_selective_repair_strategy_metrics.csv"
KEY_STRATEGIES = [
    "cross_family_margin_pareto",
    "cross_family_tail_safe_pareto",
    "risk_controlled_leave_family_out",
    "previous_high_recall",
    "global_blend_w0.75",
    "strict_safe_gate",
    "chronos_tuned_transfer",
    "leave_family_out_calibrated",
]


def refresh_cross_family_artifacts() -> None:
    """Rebuild dependent repair artifacts so stress stats cannot read stale CSVs."""
    with contextlib.redirect_stdout(io.StringIO()):
        cross.main()


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


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def percentile_interval(values: np.ndarray, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    *,
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    sampled = arr[draws]
    if statistic == "median":
        stats = np.median(sampled, axis=1)
    elif statistic == "mean":
        stats = np.mean(sampled, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    return percentile_interval(stats)


def exact_mcnemar_p(successes: int, failures: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired binary outcomes."""
    n = successes + failures
    if n == 0:
        return 1.0
    k = min(successes, failures)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def row_group(row: dict[str, str]) -> list[tuple[str, str]]:
    groups = [
        ("overall", "overall"),
        (f"role:{row['role']}", "role"),
        (f"family:{row['family']}", "family"),
        (f"source:{row['source']}", "source"),
    ]
    return groups


def paired_stats_for_rows(rows: list[dict[str, str]], strategy_id: str, group: str, group_type: str) -> dict[str, object]:
    model_fail = np.asarray([int(finite_float(row["model_failure_delta_005"])) for row in rows], dtype=int)
    repair_fail = np.asarray([int(finite_float(row["repair_failure_delta_005"])) for row in rows], dtype=int)
    model_minus_repair = model_fail - repair_fail
    rer_delta = np.asarray([finite_float(row["relative_error_ratio_delta"]) for row in rows], dtype=float)
    mae_delta = np.asarray([finite_float(row["mae_delta_vs_model"]) for row in rows], dtype=float)
    gate = np.asarray([int(finite_float(row["gate_active"])) for row in rows], dtype=int)
    wins = np.asarray([int(finite_float(row["repair_improves_model"])) for row in rows], dtype=int)
    improved = int(np.sum((model_fail == 1) & (repair_fail == 0)))
    worsened = int(np.sum((model_fail == 0) & (repair_fail == 1)))
    ci_low, ci_high = bootstrap_ci(model_minus_repair.astype(float), "mean")
    mae_low, mae_high = bootstrap_ci(mae_delta, "mean")
    rer_low, rer_high = bootstrap_ci(rer_delta, "median")
    return {
        "strategy_id": strategy_id,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "model_failure_rate": float(np.mean(model_fail)) if len(rows) else float("nan"),
        "repair_failure_rate": float(np.mean(repair_fail)) if len(rows) else float("nan"),
        "failure_rate_reduction": float(np.mean(model_minus_repair)) if len(rows) else float("nan"),
        "failure_reduction_ci_low": ci_low,
        "failure_reduction_ci_high": ci_high,
        "mcnemar_improved_count": improved,
        "mcnemar_worsened_count": worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
        "gate_rate": float(np.mean(gate)) if len(rows) else float("nan"),
        "repair_win_rate": float(np.mean(wins)) if len(rows) else float("nan"),
        "mean_mae_delta_vs_model": mean(mae_delta.tolist()),
        "mean_mae_delta_ci_low": mae_low,
        "mean_mae_delta_ci_high": mae_high,
        "median_rer_delta": median(rer_delta.tolist()),
        "median_rer_delta_ci_low": rer_low,
        "median_rer_delta_ci_high": rer_high,
    }


def paired_strategy_stats() -> list[dict[str, object]]:
    raw = [row for row in read_csv(STRATEGY_METRICS) if row["strategy_id"] in KEY_STRATEGIES]
    by_strategy: dict[str, list[dict[str, str]]] = {}
    for row in raw:
        by_strategy.setdefault(row["strategy_id"], []).append(row)
    stats_rows: list[dict[str, object]] = []
    for strategy_id, rows in by_strategy.items():
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            for key in row_group(row):
                grouped.setdefault(key, []).append(row)
        for (group, group_type), group_rows in grouped.items():
            stats_rows.append(paired_stats_for_rows(group_rows, strategy_id, group, group_type))
    return stats_rows


def classification_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    y = np.asarray([int(finite_float(row["model_failure_delta_005"])) for row in rows], dtype=int)
    pred = np.asarray([int(finite_float(row["gate_active"])) for row in rows], dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "balanced_accuracy": balanced_accuracy,
    }


def summarize_candidate_rows(rows: list[dict[str, object]], policy: dict[str, object]) -> dict[str, object]:
    overall = cross.summarize(rows, "overall", "overall")
    positive = cross.summarize([row for row in rows if row["role"] == "positive_control"], "positive_control", "role")
    failure = cross.summarize([row for row in rows if row["role"] == "failure_target"], "failure_target", "role")
    stress = cross.summarize([row for row in rows if row["role"] == "stress_target"], "stress_target", "role")
    family_positive = [
        cross.summarize(
            [row for row in rows if row["role"] == "positive_control" and row["family"] == family],
            f"family:{family}|role:positive_control",
            "family_role",
        )
        for family in sorted({str(row["family"]) for row in rows if row["role"] == "positive_control"})
    ]
    family_positive_p90 = [finite_float(row.get("p90_rer_delta")) for row in family_positive]
    family_positive_median = [finite_float(row.get("median_rer_delta")) for row in family_positive]
    cls = classification_metrics(rows)
    return {
        "policy_id": policy["policy_id"],
        "profile": policy.get("profile", ""),
        "min_active": policy.get("min_active", ""),
        "weight": policy.get("weight", ""),
        "weight_step": policy.get("weight_step", ""),
        "max_weight": policy.get("max_weight", ""),
        "anchor_mode": policy.get("anchor_mode", "none"),
        "shield_mode": policy.get("shield_mode", "none"),
        "shield_threshold": policy.get("shield_threshold", ""),
        "shield_weight_cap": policy.get("shield_weight_cap", ""),
        "policy_kind": policy.get("policy_kind", "fixed_weight"),
        "n_windows": overall["n_windows"],
        "gate_rate": overall["gate_rate"],
        "failure_rate_reduction": overall["failure_rate_reduction"],
        "repair_failure_rate": overall["repair_failure_rate_delta_005"],
        "model_failure_rate": overall["model_failure_rate_delta_005"],
        "positive_control_failure_delta": positive["repair_failure_rate_delta_005"] - positive["model_failure_rate_delta_005"],
        "positive_control_median_rer_delta": positive["median_rer_delta"],
        "positive_control_mean_rer_delta": mean(
            [finite_float(row["relative_error_ratio_delta"]) for row in rows if row["role"] == "positive_control"]
        ),
        "positive_control_p90_rer_delta": positive.get("p90_rer_delta", float("nan")),
        "positive_control_family_max_median_rer_delta": max(family_positive_median) if family_positive_median else float("nan"),
        "positive_control_family_max_p90_rer_delta": max(family_positive_p90) if family_positive_p90 else float("nan"),
        "failure_target_reduction": failure["failure_rate_reduction"],
        "stress_target_reduction": stress["failure_rate_reduction"],
        **cls,
    }


def policy_grid() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows, inventory = cross.load_cross_family_windows()
    del inventory
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for policy in cross.policy_candidates():
        policy_id = str(policy["policy_id"])
        repaired = cross.apply_single_policy(windows, policy, policy_id)
        rows.append(summarize_candidate_rows(repaired, policy))
        metric_rows.extend(repaired)
    for profile in ["loose", "balanced", "strict"]:
        for min_active in [2, 3]:
            policy = selective.constant_policy(profile, min_active, 1.0, f"{profile}_n{min_active}_baseline_fallback")
            repaired = cross.apply_single_policy(windows, policy, str(policy["policy_id"]))
            rows.append(summarize_candidate_rows(repaired, policy))
            metric_rows.extend(repaired)
    for weight in [0.25, 0.50, 0.75, 1.00]:
        policy = selective.global_blend_policy(weight)
        policy["policy_id"] = f"global_blend_w{weight:.2f}"
        repaired = cross.apply_single_policy(windows, policy, str(policy["policy_id"]))
        rows.append(summarize_candidate_rows(repaired, policy))
        metric_rows.extend(repaired)
    return rows, metric_rows


def mark_pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        median_safe = (
            finite_float(row["positive_control_failure_delta"]) <= 1e-12
            and finite_float(row["positive_control_median_rer_delta"]) <= 0.05
        )
        tail_safe = median_safe and finite_float(row["positive_control_p90_rer_delta"]) <= 0.25
        family_tail_safe = tail_safe and finite_float(row["positive_control_family_max_p90_rer_delta"]) <= 0.25
        item = dict(row)
        item["positive_control_safe"] = int(median_safe)
        item["positive_control_tail_safe"] = int(tail_safe)
        item["positive_control_family_tail_safe"] = int(family_tail_safe)
        item["pareto_frontier"] = 0
        item["tail_safe_pareto_frontier"] = 0
        item["family_tail_safe_pareto_frontier"] = 0
        enriched.append(item)
    for safety_key, frontier_key in [
        ("positive_control_safe", "pareto_frontier"),
        ("positive_control_tail_safe", "tail_safe_pareto_frontier"),
        ("positive_control_family_tail_safe", "family_tail_safe_pareto_frontier"),
    ]:
        safe = [row for row in enriched if int(row[safety_key]) == 1]
        for row in safe:
            dominated = False
            for other in safe:
                if other is row:
                    continue
                better_or_equal = (
                    finite_float(other["failure_rate_reduction"]) >= finite_float(row["failure_rate_reduction"]) - 1e-12
                    and finite_float(other["gate_rate"]) <= finite_float(row["gate_rate"]) + 1e-12
                    and finite_float(other["positive_control_p90_rer_delta"])
                    <= finite_float(row["positive_control_p90_rer_delta"]) + 1e-12
                )
                strictly_better = (
                    finite_float(other["failure_rate_reduction"]) > finite_float(row["failure_rate_reduction"]) + 1e-12
                    or finite_float(other["gate_rate"]) < finite_float(row["gate_rate"]) - 1e-12
                    or finite_float(other["positive_control_p90_rer_delta"])
                    < finite_float(row["positive_control_p90_rer_delta"]) - 1e-12
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            row[frontier_key] = int(not dominated)
    return enriched


def plot_figures(policy_rows: list[dict[str, object]], stats_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    safe_x = [finite_float(row["gate_rate"]) for row in policy_rows if int(row["positive_control_safe"]) == 1]
    safe_y = [finite_float(row["failure_rate_reduction"]) for row in policy_rows if int(row["positive_control_safe"]) == 1]
    unsafe_x = [finite_float(row["gate_rate"]) for row in policy_rows if int(row["positive_control_safe"]) == 0]
    unsafe_y = [finite_float(row["failure_rate_reduction"]) for row in policy_rows if int(row["positive_control_safe"]) == 0]
    frontier = sorted(
        [row for row in policy_rows if int(row["pareto_frontier"]) == 1],
        key=lambda item: finite_float(item["gate_rate"]),
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.scatter(unsafe_x, unsafe_y, c="#b8b8b8", label="PC-cost candidates", alpha=0.65)
    ax.scatter(safe_x, safe_y, c="#2474a6", label="PC-safe candidates", alpha=0.85)
    if frontier:
        ax.plot(
            [finite_float(row["gate_rate"]) for row in frontier],
            [finite_float(row["failure_rate_reduction"]) for row in frontier],
            c="#c0392b",
            lw=2.0,
            marker="o",
            label="PC-safe Pareto frontier",
        )
    for row in policy_rows:
        if str(row["policy_id"]) in {"loose_n2_w0.75", "balanced_n2_w0.75", "global_blend_w0.75"}:
            ax.annotate(
                str(row["policy_id"]).replace("_", " "),
                (finite_float(row["gate_rate"]), finite_float(row["failure_rate_reduction"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Gate rate")
    ax.set_ylabel("Failure-rate reduction")
    ax.set_title("Cross-family repair tradeoff")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latest_repair_pareto_frontier.png", dpi=180)
    fig.savefig(FIG_DIR / "latest_repair_pareto_frontier.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = [finite_float(row["false_positive_rate"]) for row in policy_rows]
    y = [finite_float(row["recall"]) for row in policy_rows]
    colors = [finite_float(row["failure_rate_reduction"]) for row in policy_rows]
    scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=48, alpha=0.9)
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall / true positive rate")
    ax.set_title("Gate operating points")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Failure-rate reduction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latest_gate_operating_points.png", dpi=180)
    fig.savefig(FIG_DIR / "latest_gate_operating_points.pdf")
    plt.close(fig)

    overall = [
        row
        for row in stats_rows
        if row["group"] == "overall"
        and row["strategy_id"]
        in {
            "cross_family_margin_pareto",
            "cross_family_tail_safe_pareto",
            "risk_controlled_leave_family_out",
            "previous_high_recall",
            "global_blend_w0.75",
        }
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = [str(row["strategy_id"]).replace("_", "\n") for row in overall]
    values = [finite_float(row["failure_rate_reduction"]) for row in overall]
    err_low = [finite_float(row["failure_rate_reduction"]) - finite_float(row["failure_reduction_ci_low"]) for row in overall]
    err_high = [finite_float(row["failure_reduction_ci_high"]) - finite_float(row["failure_rate_reduction"]) for row in overall]
    ax.bar(
        range(len(overall)),
        values,
        yerr=[err_low, err_high],
        color=["#2474a6", "#45a173", "#d99a2b", "#5499c7", "#8e44ad"][: len(overall)],
        capsize=4,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(len(overall)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Failure-rate reduction with 95% bootstrap CI")
    ax.set_title("Paired repair effect")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latest_paired_repair_effects.png", dpi=180)
    fig.savefig(FIG_DIR / "latest_paired_repair_effects.pdf")
    plt.close(fig)


def format_pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def report(stats_rows: list[dict[str, object]], policy_rows: list[dict[str, object]]) -> None:
    lookup = {(row["strategy_id"], row["group"]): row for row in stats_rows}
    cross_summary = read_csv(ROOT / "results" / "repair" / "cross_family_selective_repair_strategy_summary.csv")
    cross_lookup = {(row["strategy_id"], row["group"]): row for row in cross_summary}
    selected_rows = read_csv(ROOT / "results" / "repair" / "cross_family_selective_repair_selected_policies.csv")
    selected_lookup = {row["selection_scope"]: row for row in selected_rows}
    policy_lookup = {str(row["policy_id"]): row for row in policy_rows}

    def selected_policy_row(scope: str, prefix: str) -> dict[str, object]:
        selected_id = str(selected_lookup.get(scope, {}).get("selected_policy_id", ""))
        policy_id = selected_id.removeprefix(prefix)
        return policy_lookup.get(policy_id, {})

    margin_policy = selected_policy_row("cross_family_margin_pareto", "cross_family_margin_pareto_")
    tail_policy = selected_policy_row("cross_family_tail_safe_pareto", "cross_family_tail_safe_")
    risk_control = cross_lookup[("risk_controlled_leave_family_out", "positive_control")]
    risk_timesfm_control = cross_lookup[("risk_controlled_leave_family_out", "family:timesfm|role:positive_control")]
    tail_timesfm_control = cross_lookup[("cross_family_tail_safe_pareto", "family:timesfm|role:positive_control")]
    pareto = [row for row in policy_rows if int(row["pareto_frontier"]) == 1]
    pareto = sorted(pareto, key=lambda row: (finite_float(row["gate_rate"]), -finite_float(row["failure_rate_reduction"])))
    key_lines = [
        "| Strategy | Group | Model fail | Repair fail | Reduction | 95% CI | McNemar p | Gate | Median RER delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in [
        "cross_family_margin_pareto",
        "cross_family_tail_safe_pareto",
        "risk_controlled_leave_family_out",
        "previous_high_recall",
        "global_blend_w0.75",
        "strict_safe_gate",
    ]:
        row = lookup[(strategy, "overall")]
        key_lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    "overall",
                    format_pct(row["model_failure_rate"]),
                    format_pct(row["repair_failure_rate"]),
                    format_pct(row["failure_rate_reduction"]),
                    f"[{format_pct(row['failure_reduction_ci_low'])}, {format_pct(row['failure_reduction_ci_high'])}]",
                    f"{finite_float(row['mcnemar_exact_p']):.4f}",
                    format_pct(row["gate_rate"]),
                    f"{finite_float(row['median_rer_delta']):.3f}",
                ]
            )
            + " |"
        )
    family_lines = [
        "| Strategy | Family | N | Reduction | 95% CI | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for family in ["family:chronos", "family:moirai", "family:timesfm"]:
        row = lookup[("cross_family_margin_pareto", family)]
        family_lines.append(
            "| "
            + " | ".join(
                [
                    "cross_family_margin_pareto",
                    family.replace("family:", ""),
                    str(int(row["n_windows"])),
                    format_pct(row["failure_rate_reduction"]),
                    f"[{format_pct(row['failure_reduction_ci_low'])}, {format_pct(row['failure_reduction_ci_high'])}]",
                    f"{finite_float(row['mcnemar_exact_p']):.4f}",
                ]
            )
            + " |"
        )
    frontier_lines = [
        "| Policy | Gate | Reduction | PC median dRER | PC p90 dRER | Precision | Recall | BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pareto[:8]:
        frontier_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["policy_id"]),
                    format_pct(row["gate_rate"]),
                    format_pct(row["failure_rate_reduction"]),
                    f"{finite_float(row['positive_control_median_rer_delta']):.3f}",
                    f"{finite_float(row['positive_control_p90_rer_delta']):.3f}",
                    format_pct(row["precision"]),
                    format_pct(row["recall"]),
                    format_pct(row["balanced_accuracy"]),
                ]
            )
            + " |"
        )
    tail_pareto = [row for row in policy_rows if int(row["tail_safe_pareto_frontier"]) == 1]
    tail_pareto = sorted(tail_pareto, key=lambda row: (finite_float(row["gate_rate"]), -finite_float(row["failure_rate_reduction"])))
    tail_lines = [
        "| Policy | Gate | Reduction | PC median dRER | PC p90 dRER | Precision | Recall | BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in tail_pareto[:8]:
        tail_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["policy_id"]),
                    format_pct(row["gate_rate"]),
                    format_pct(row["failure_rate_reduction"]),
                    f"{finite_float(row['positive_control_median_rer_delta']):.3f}",
                    f"{finite_float(row['positive_control_p90_rer_delta']):.3f}",
                    format_pct(row["precision"]),
                    format_pct(row["recall"]),
                    format_pct(row["balanced_accuracy"]),
                ]
            )
            + " |"
        )
    family_tail_pareto = [row for row in policy_rows if int(row["family_tail_safe_pareto_frontier"]) == 1]
    family_tail_pareto = sorted(
        family_tail_pareto,
        key=lambda row: (-finite_float(row["failure_rate_reduction"]), finite_float(row["gate_rate"])),
    )
    family_tail_lines = [
        "| Policy | Gate | Reduction | PC p90 dRER | Max-family PC p90 dRER | Precision | Recall | BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in family_tail_pareto[:8]:
        family_tail_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["policy_id"]),
                    format_pct(row["gate_rate"]),
                    format_pct(row["failure_rate_reduction"]),
                    f"{finite_float(row['positive_control_p90_rer_delta']):.3f}",
                    f"{finite_float(row['positive_control_family_max_p90_rer_delta']):.3f}",
                    format_pct(row["precision"]),
                    format_pct(row["recall"]),
                    format_pct(row["balanced_accuracy"]),
                ]
            )
            + " |"
        )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# AAAI Stress Experiment Report",
                "",
                "## What This Tests",
                "",
                "This pass targets the main reviewer weaknesses: whether the repair effect is statistically visible under paired tests, whether selective gates occupy a real Pareto frontier rather than merely duplicating global blending, and whether the gate should be framed as a high-recall trigger rather than a robust predictor.",
                "",
                "## Paired Repair Statistics",
                "",
                "\n".join(key_lines),
                "",
                "## Cross-Family Boundary Under The Pareto Gate",
                "",
                "\n".join(family_lines),
                "",
                "## Median-Safe Policy Frontier",
                "",
                "\n".join(frontier_lines),
                "",
                "## Tail-Safe Policy Frontier",
                "",
                "\n".join(tail_lines),
                "",
                "## Family-Wise Tail-Safe Policy Frontier",
                "",
                "\n".join(family_tail_lines),
                "",
                "## Interpretation",
                "",
                "- If a bootstrap CI excludes zero and McNemar p is small, the repair effect is not just table noise.",
                (
                    "- The adaptive policy search now yields two publishable operating points: "
                    f"median-safe repair reduces failure by {format_pct(lookup[('cross_family_margin_pareto', 'overall')]['failure_rate_reduction'])} "
                    f"but has positive-control p90 dRER {finite_float(margin_policy.get('positive_control_p90_rer_delta')):.3f}; "
                    f"conflict-shielded tail-safe repair reduces failure by {format_pct(lookup[('cross_family_tail_safe_pareto', 'overall')]['failure_rate_reduction'])} "
                    f"with aggregate positive-control p90 dRER {finite_float(tail_policy.get('positive_control_p90_rer_delta')):.3f} "
                    f"and TimesFM positive-control family p90 {finite_float(tail_timesfm_control['p90_rer_delta']):.3f}."
                ),
                (
                    "- The risk-controlled leave-family-out strategy is the clean validation analogue: it selects only low-structure "
                    "anchored policies on calibration families under a positive-control p90 RER-delta budget; high repair weights are allowed "
                    "only when protected by an interval conflict shield, then evaluated on the "
                    f"held-out family. It reduces failure by {format_pct(lookup[('risk_controlled_leave_family_out', 'overall')]['failure_rate_reduction'])}, "
                    f"with aggregate positive-control p90 {finite_float(risk_control['p90_rer_delta']):.3f} and TimesFM positive-control family p90 "
                    f"{finite_float(risk_timesfm_control['p90_rer_delta']):.3f}."
                ),
                "- Global blending remains a useful control because it is competitive and exposes positive-control RER cost; it no longer dominates the anchored median-safe selective policy.",
                "- Direct baseline fallback is a useful model-selection control, but because failure is defined relative to that same baseline it must be audited with positive-control tail costs, not just median RER.",
                "- If BA stays mediocre while recall is high, the gate is a failure-aware trigger, not a calibrated cross-domain classifier.",
                "",
                "## Artifacts",
                "",
                "- `results/aaai_stress/repair_paired_stats.csv`",
                "- `results/aaai_stress/policy_pareto_grid.csv`",
                "- `results/aaai_stress/policy_grid_window_metrics.csv`",
                "- `figures/aaai_stress/latest_repair_pareto_frontier.png`",
                "- `figures/aaai_stress/latest_gate_operating_points.png`",
                "- `figures/aaai_stress/latest_paired_repair_effects.png`",
            ]
        )
        + "\n"
    )


def main() -> None:
    refresh_cross_family_artifacts()
    stats_rows = paired_strategy_stats()
    policy_rows, policy_metric_rows = policy_grid()
    policy_rows = mark_pareto(policy_rows)
    write_csv(OUT_DIR / "repair_paired_stats.csv", stats_rows)
    write_csv(OUT_DIR / "policy_pareto_grid.csv", policy_rows)
    write_csv(OUT_DIR / "policy_grid_window_metrics.csv", policy_metric_rows)
    plot_figures(policy_rows, stats_rows)
    report(stats_rows, policy_rows)
    print((OUT_DIR / "repair_paired_stats.csv").relative_to(ROOT))
    print((OUT_DIR / "policy_pareto_grid.csv").relative_to(ROOT))
    print(DOC_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
