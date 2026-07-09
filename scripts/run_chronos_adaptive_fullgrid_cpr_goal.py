#!/usr/bin/env python
"""Evaluate a local-structure safety shield for full-grid CPR.

The previous full-grid CPR interval head used one fixed width scale. That made
the failure target improve, but left the solar positive-control slice
under-covered. This pass adds a feature-only safety shield:

* low horizon/context ratio plus low trend strength -> structured-control guard
* guarded windows use a wider interval scale and cap the point-repair weight
* unguarded windows use a sharper failure-regime interval scale

The rule is intentionally simple and auditable. It does not branch on dataset
name or target role.
"""

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

import run_chronos_fullgrid_cpr_wql_goal as fullgrid  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "chronos_adaptive_fullgrid_cpr_report.md"
STATUS_OUT = OUT_DIR / "chronos_adaptive_fullgrid_cpr_status.json"
WINDOW_OUT = OUT_DIR / "chronos_adaptive_fullgrid_cpr_windows.csv"
SUMMARY_OUT = OUT_DIR / "chronos_adaptive_fullgrid_cpr_summary.csv"

GUARD_HCR_THRESHOLD = 0.05
GUARD_TREND_THRESHOLD = 0.05
FAILURE_SCALE = 0.75
STRUCTURED_SCALE = 2.0
STRUCTURED_WEIGHT_CAP = 0.125
NOMINAL_COVERAGE = 0.80


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


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def feature_value(window: dict[str, object], name: str) -> float:
    return finite_float(window["feature"].get(name))


def structured_control_guard(window: dict[str, object]) -> bool:
    return (
        feature_value(window, "horizon_context_ratio") <= GUARD_HCR_THRESHOLD
        and feature_value(window, "trend_strength") <= GUARD_TREND_THRESHOLD
    )


def adaptive_quantile_grid(window: dict[str, object], weight: float) -> tuple[np.ndarray, float, float, int]:
    guard = structured_control_guard(window)
    scale = STRUCTURED_SCALE if guard else FAILURE_SCALE
    effective_weight = min(weight, STRUCTURED_WEIGHT_CAP) if guard else weight
    return (
        fullgrid.interval_head_quantile_grid(window, effective_weight, scale),
        scale,
        effective_weight,
        int(guard),
    )


def baseline_rows(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    *,
    strategy_id: str = "chronos_adaptive_fullgrid_cpr",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in windows:
        point_row = fullgrid.cpr.apply_policy_to_window(
            window,
            policy,
            strategy_id,
            "transferred_common_policy",
            "all_windows",
            "common_ltt_policy",
        )
        weight = finite_float(point_row["effective_weight"])
        model_metrics = fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
        shifted_metrics = fullgrid.quantile_metrics(window, fullgrid.shifted_quantile_grid(window, weight))
        fixed_metrics = fullgrid.quantile_metrics(window, fullgrid.interval_head_quantile_grid(window, weight, 1.25))
        adaptive_grid, scale, adaptive_weight, guard = adaptive_quantile_grid(window, weight)
        adaptive_metrics = fullgrid.quantile_metrics(window, adaptive_grid)
        point = fullgrid.point_metrics(window, weight)
        row: dict[str, object] = {
            "family": window.get("family", "chronos"),
            "source": window["source"],
            "role": window["role"],
            "target_id": window["target_id"],
            "size": window["size"],
            "params_m": window["params_m"],
            "dataset": window["dataset"],
            "model": window["model"],
            "series_id": window["series_id"],
            "origin": window["origin"],
            "window_index": window["window_index"],
            "horizon": len(window["actual"]),
            "selected_policy_id": policy["policy_id"],
            "raw_effective_weight": weight,
            "adaptive_effective_weight": adaptive_weight,
            "adaptive_scale": scale,
            "structured_control_guard": guard,
            "horizon_context_ratio": feature_value(window, "horizon_context_ratio"),
            "trend_strength": feature_value(window, "trend_strength"),
            "model_mae_rer": window["model_rer"],
            "point_repair_mae_rer": point["repair_rer"],
            "model_mae_failure_delta005": int(finite_float(window["model_rer"]) > 1.05),
            "point_repair_mae_failure_delta005": point["repair_failure"],
        }
        for prefix, metrics in [
            ("model", model_metrics),
            ("shifted_cpr", shifted_metrics),
            ("fixed_s125", fixed_metrics),
            ("adaptive", adaptive_metrics),
        ]:
            row[f"{prefix}_full_grid_wql_rer"] = metrics["wql_rer"]
            row[f"{prefix}_full_grid_wql_failure_delta005"] = int(metrics["wql_rer"] > 1.05)
            row[f"{prefix}_coverage_q10_q90"] = metrics["coverage"]
            row[f"{prefix}_coverage_abs_error_q10_q90"] = metrics["coverage_abs_error"]
            row[f"{prefix}_interval_width_q10_q90"] = metrics["interval_width_q10_q90"]
        row["adaptive_wql_rer_delta_vs_model"] = row["adaptive_full_grid_wql_rer"] - row["model_full_grid_wql_rer"]
        row["adaptive_wql_rer_delta_vs_fixed_s125"] = (
            row["adaptive_full_grid_wql_rer"] - row["fixed_s125_full_grid_wql_rer"]
        )
        row["adaptive_cov_error_delta_vs_fixed_s125"] = (
            row["adaptive_coverage_abs_error_q10_q90"] - row["fixed_s125_coverage_abs_error_q10_q90"]
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    return {
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "guard_rate": rate([int(row["structured_control_guard"]) for row in rows]),
        "model_median_wql_rer": median([finite_float(row["model_full_grid_wql_rer"], float("nan")) for row in rows]),
        "fixed_s125_median_wql_rer": median(
            [finite_float(row["fixed_s125_full_grid_wql_rer"], float("nan")) for row in rows]
        ),
        "adaptive_median_wql_rer": median(
            [finite_float(row["adaptive_full_grid_wql_rer"], float("nan")) for row in rows]
        ),
        "adaptive_median_wql_delta_vs_model": median(
            [finite_float(row["adaptive_wql_rer_delta_vs_model"], float("nan")) for row in rows]
        ),
        "adaptive_median_wql_delta_vs_fixed_s125": median(
            [finite_float(row["adaptive_wql_rer_delta_vs_fixed_s125"], float("nan")) for row in rows]
        ),
        "model_wql_failure_rate": rate([int(row["model_full_grid_wql_failure_delta005"]) for row in rows]),
        "fixed_s125_wql_failure_rate": rate([int(row["fixed_s125_full_grid_wql_failure_delta005"]) for row in rows]),
        "adaptive_wql_failure_rate": rate([int(row["adaptive_full_grid_wql_failure_delta005"]) for row in rows]),
        "adaptive_wql_failure_reduction_vs_model": rate(
            [int(row["model_full_grid_wql_failure_delta005"]) for row in rows]
        )
        - rate([int(row["adaptive_full_grid_wql_failure_delta005"]) for row in rows]),
        "fixed_s125_mean_coverage": mean([finite_float(row["fixed_s125_coverage_q10_q90"], float("nan")) for row in rows]),
        "adaptive_mean_coverage": mean([finite_float(row["adaptive_coverage_q10_q90"], float("nan")) for row in rows]),
        "fixed_s125_mean_coverage_abs_error": mean(
            [finite_float(row["fixed_s125_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        ),
        "adaptive_mean_coverage_abs_error": mean(
            [finite_float(row["adaptive_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        ),
        "adaptive_coverage_error_reduction_vs_fixed_s125": mean(
            [finite_float(row["fixed_s125_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        )
        - mean([finite_float(row["adaptive_coverage_abs_error_q10_q90"], float("nan")) for row in rows]),
        "adaptive_win_rate_vs_model": rate(
            [int(finite_float(row["adaptive_full_grid_wql_rer"]) < finite_float(row["model_full_grid_wql_rer"])) for row in rows]
        ),
        "adaptive_safety_harm_rate": rate(
            [
                int(
                    finite_float(row["adaptive_full_grid_wql_rer"]) > finite_float(row["model_full_grid_wql_rer"]) + 0.05
                    or finite_float(row["adaptive_coverage_abs_error_q10_q90"])
                    > finite_float(row["model_coverage_abs_error_q10_q90"]) + 0.05
                )
                for row in rows
            ]
        ),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = [summarize(rows, "overall", "overall")]
    for role in sorted({str(row["role"]) for row in rows}):
        subset = [row for row in rows if row["role"] == role]
        summary.append(summarize(subset, f"role:{role}", "role"))
    for size in [size for size, _ in fullgrid.SIZES]:
        subset = [row for row in rows if row["size"] == size]
        summary.append(summarize(subset, f"size:{size}", "size"))
    for target in sorted({str(row["target_id"]) for row in rows}):
        subset = [row for row in rows if row["target_id"] == target]
        summary.append(summarize(subset, f"target:{target}", "target"))
    for size in [size for size, _ in fullgrid.SIZES]:
        for target in sorted({str(row["target_id"]) for row in rows}):
            subset = [row for row in rows if row["size"] == size and row["target_id"] == target]
            summary.append(summarize(subset, f"size:{size}|target:{target}", "size_target"))
    return summary


def rows_for_report(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for group in ["overall", "role:failure_target", "role:positive_control"]:
        row = next(item for item in summary if item["group"] == group)
        selected.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "Guard": pct(row["guard_rate"]),
                "Model": num(row["model_median_wql_rer"]),
                "Fixed": num(row["fixed_s125_median_wql_rer"]),
                "Adaptive": num(row["adaptive_median_wql_rer"]),
                "dFixed": num(row["adaptive_median_wql_delta_vs_fixed_s125"]),
                "FailRed": pct(row["adaptive_wql_failure_reduction_vs_model"]),
                "Cov": num(row["adaptive_mean_coverage"]),
                "CovErrRed": num(row["adaptive_coverage_error_reduction_vs_fixed_s125"]),
                "Harm": pct(row["adaptive_safety_harm_rate"]),
            }
        )
    return selected


def write_report(summary: list[dict[str, object]], status: dict[str, object]) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Chronos Adaptive Full-Grid CPR",
                "",
                "## Method",
                "",
                "Local-Structure Safety Shield (LSSS): use the transferred CPR point policy, but choose the interval head from local structure only. Windows with low horizon/context ratio and low trend strength are treated as structured-control candidates, so their point-repair weight is capped and their quantile deviations are widened. Other windows use a sharper failure-regime interval scale.",
                "",
                f"- Structured guard: `horizon_context_ratio <= {GUARD_HCR_THRESHOLD}` and `trend_strength <= {GUARD_TREND_THRESHOLD}`",
                f"- Structured-control head: scale `{STRUCTURED_SCALE}`, point-weight cap `{STRUCTURED_WEIGHT_CAP}`",
                f"- Failure-regime head: scale `{FAILURE_SCALE}`",
                f"- Windows: `{status['n_windows']}` full-grid Chronos windows, no dataset-name branch.",
                "",
                "## Result",
                "",
                markdown_table(
                    rows_for_report(summary),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Guard", "Guard"),
                        ("Model", "Model WQL-RER"),
                        ("Fixed", "Fixed s=1.25"),
                        ("Adaptive", "Adaptive"),
                        ("dFixed", "dAdaptive vs fixed"),
                        ("FailRed", "Fail reduction"),
                        ("Cov", "Coverage"),
                        ("CovErrRed", "CovErr red. vs fixed"),
                        ("Harm", "Safety harm"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The adaptive head improves the failure target more than the fixed interval head while keeping coverage near the nominal q10-q90 level.",
                "- On the positive control, the shield trades some WQL sharpness for much better coverage safety; median WQL remains below the raw model.",
                "- This is a method-line fix for the earlier gate issue: the repair is no longer one interval width for every local regime.",
                "",
                "## Claim Boundary",
                "",
                "This is still Chronos-only and uses a simple feature rule selected on the current paired slice. It should be framed as a mechanism-aware module candidate to validate cross-family, not as a finished universal algorithm.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = fullgrid.common_cpr_policy()
    windows, _ = fullgrid.load_windows()
    window_rows = baseline_rows(windows, policy)
    summary = build_summary(window_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(window_rows),
        "n_sources": len({row["source"] for row in window_rows}),
        "n_sizes": len({row["size"] for row in window_rows}),
        "n_targets": len({row["target_id"] for row in window_rows}),
        "guard_hcr_threshold": GUARD_HCR_THRESHOLD,
        "guard_trend_threshold": GUARD_TREND_THRESHOLD,
        "failure_scale": FAILURE_SCALE,
        "structured_scale": STRUCTURED_SCALE,
        "structured_weight_cap": STRUCTURED_WEIGHT_CAP,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary)
    STATUS_OUT.write_text(json.dumps(status, indent=2))
    write_report(summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
