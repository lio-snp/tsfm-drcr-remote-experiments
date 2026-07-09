#!/usr/bin/env python
"""Component ablation for the frozen DRCR-Smooth expansion.

The frozen expansion showed useful failure-side transfer but visible
positive-control/stress harm.  This script decomposes the deployed repair into
its two moving parts:

- point shift: moving the forecast center toward the classical baseline,
- interval scale: widening/sharpening the quantile grid around that center.

No candidate parameter is tuned here.  The selected DRCR-Smooth head remains
the frozen one from configs/drcr_smooth_frozen_protocol.json.
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

import run_drcr_smooth_frozen_expansion_goal as expansion  # noqa: E402
import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_smooth_component_ablation_report.md"
STATUS_OUT = OUT_DIR / "drcr_smooth_component_ablation_status.json"
WINDOW_OUT = OUT_DIR / "drcr_smooth_component_ablation_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_smooth_component_ablation_summary.csv"

VARIANTS = [
    {
        "variant_id": "native_model",
        "point_shift": 0,
        "interval_scale": 0,
        "description": "native TSFM quantile grid, no intervention",
    },
    {
        "variant_id": "interval_only",
        "point_shift": 0,
        "interval_scale": 1,
        "description": "apply frozen DRCR-Smooth interval scale, no point shift",
    },
    {
        "variant_id": "point_shift_only",
        "point_shift": 1,
        "interval_scale": 0,
        "description": "apply frozen CPR point shift, keep native interval shape",
    },
    {
        "variant_id": "full_drcr_smooth",
        "point_shift": 1,
        "interval_scale": 1,
        "description": "current frozen DRCR-Smooth deployment",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def selected_candidate() -> dict[str, object]:
    candidates = {str(candidate["candidate_id"]): candidate for candidate in smooth.candidate_configs()}
    return candidates[smooth.SMOOTH_CANDIDATE_ID]


def repair_grid(
    window: dict[str, object],
    *,
    effective_weight: float,
    scale: float,
    point_shift: int,
    interval_scale: int,
) -> np.ndarray:
    if not point_shift and not interval_scale:
        return np.asarray(window["quantile_grid"], dtype=float)
    weight = effective_weight if point_shift else 0.0
    applied_scale = scale if interval_scale else 1.0
    return smooth.crc.fullgrid.interval_head_quantile_grid(window, weight, applied_scale)


def build_rows() -> list[dict[str, object]]:
    smooth.validate_protocol_config()
    windows, _ = expansion.load_expansion_windows()
    policy = smooth.crc.fullgrid.common_cpr_policy()
    candidate = selected_candidate()
    rows: list[dict[str, object]] = []
    for window in windows:
        point_row = smooth.crc.fullgrid.cpr.apply_policy_to_window(
            window,
            policy,
            "drcr_smooth_component_ablation",
            smooth.SPLIT_PROTOCOL,
            smooth.SPLIT_ID,
            "common_ltt_policy",
        )
        raw_weight = finite_float(point_row["effective_weight"])
        scale, structured_guard, width_guard, smooth_score = smooth.scale_for_window(window, candidate)
        effective_weight = min(raw_weight, smooth.crc.STRUCTURED_WEIGHT_CAP) if structured_guard else raw_weight
        model_metrics = smooth.crc.fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
        for variant in VARIANTS:
            grid = repair_grid(
                window,
                effective_weight=effective_weight,
                scale=scale,
                point_shift=int(variant["point_shift"]),
                interval_scale=int(variant["interval_scale"]),
            )
            metrics = smooth.crc.fullgrid.quantile_metrics(window, grid)
            rows.append(
                {
                    "protocol_id": smooth.PROTOCOL_ID,
                    "protocol_sha256": smooth.protocol_sha256(),
                    "candidate_id": candidate["candidate_id"],
                    "variant_id": variant["variant_id"],
                    "variant_description": variant["description"],
                    "variant_point_shift": variant["point_shift"],
                    "variant_interval_scale": variant["interval_scale"],
                    "split_protocol": smooth.SPLIT_PROTOCOL,
                    "split_id": smooth.SPLIT_ID,
                    "family": window["family"],
                    "source": window["source"],
                    "role": window["role"],
                    "target_id": window["target_id"],
                    "evidence_tier": window["evidence_tier"],
                    "expansion_set": window["expansion_set"],
                    "dataset": window["dataset"],
                    "model": window["model"],
                    "series_id": window["series_id"],
                    "origin": window["origin"],
                    "window_index": window["window_index"],
                    "quantile_grid_n_levels": window["quantile_grid_n_levels"],
                    "selected_policy_id": policy["policy_id"],
                    "gate_active": point_row["gate_active"],
                    "shield_active": point_row["shield_active"],
                    "conflict_override": point_row["conflict_override"],
                    "reference_outside_interval_rate": point_row["reference_outside_interval_rate"],
                    "low_structure_factor_count": point_row["low_structure_factor_count"],
                    "raw_effective_weight": raw_weight,
                    "effective_weight": effective_weight,
                    "interval_scale": scale,
                    "structured_control_guard": structured_guard,
                    "width_collapse_guard": width_guard,
                    "smooth_width_score": smooth_score,
                    "native_width_ratio": smooth.crc.native_width_ratio(window),
                    "model_wql_rer": model_metrics["wql_rer"],
                    "variant_wql_rer": metrics["wql_rer"],
                    "variant_wql_delta_vs_model": metrics["wql_rer"] - model_metrics["wql_rer"],
                    "model_wql_failure_delta005": int(model_metrics["wql_rer"] > 1.05),
                    "variant_wql_failure_delta005": int(metrics["wql_rer"] > 1.05),
                    "model_coverage_q10_q90": model_metrics["coverage"],
                    "variant_coverage_q10_q90": metrics["coverage"],
                    "variant_undercoverage_risk": max(0.0, smooth.crc.NOMINAL_COVERAGE - metrics["coverage"]),
                    "model_coverage_abs_error": model_metrics["coverage_abs_error"],
                    "variant_coverage_abs_error": metrics["coverage_abs_error"],
                    "variant_interval_width_q10_q90": metrics["interval_width_q10_q90"],
                    "variant_win_vs_model": int(metrics["wql_rer"] < model_metrics["wql_rer"]),
                    "variant_wql_noninferiority_harm": int(
                        metrics["wql_rer"] > model_metrics["wql_rer"] + smooth.WQL_HARM_MARGIN
                    ),
                    "variant_safety_harm": int(
                        metrics["wql_rer"] > model_metrics["wql_rer"] + smooth.WQL_HARM_MARGIN
                        or metrics["coverage_abs_error"] > model_metrics["coverage_abs_error"] + 0.05
                    ),
                }
            )
    return rows


def summarize_variant(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    return {
        "group": group,
        "group_type": group_type,
        "variant_id": rows[0]["variant_id"],
        "variant_description": rows[0]["variant_description"],
        "n_windows": len(rows),
        "quantile_grid_n_levels": ";".join(
            str(value) for value in sorted({int(row["quantile_grid_n_levels"]) for row in rows})
        ),
        "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in rows]),
        "variant_median_wql_rer": median([finite_float(row["variant_wql_rer"], float("nan")) for row in rows]),
        "variant_mean_wql_delta_vs_model": mean(
            [finite_float(row["variant_wql_delta_vs_model"], float("nan")) for row in rows]
        ),
        "variant_median_wql_delta_vs_model": median(
            [finite_float(row["variant_wql_delta_vs_model"], float("nan")) for row in rows]
        ),
        "wql_failure_reduction_vs_model": rate([int(row["model_wql_failure_delta005"]) for row in rows])
        - rate([int(row["variant_wql_failure_delta005"]) for row in rows]),
        "variant_win_rate_vs_model": rate([int(row["variant_win_vs_model"]) for row in rows]),
        "variant_mean_coverage": mean([finite_float(row["variant_coverage_q10_q90"], float("nan")) for row in rows]),
        "variant_undercoverage_risk": mean(
            [finite_float(row["variant_undercoverage_risk"], float("nan")) for row in rows]
        ),
        "variant_wql_noninferiority_harm_rate": rate(
            [int(row["variant_wql_noninferiority_harm"]) for row in rows]
        ),
        "variant_safety_harm_rate": rate([int(row["variant_safety_harm"]) for row in rows]),
        "gate_rate": rate([int(row["gate_active"]) for row in rows]),
        "structured_guard_rate": rate([int(row["structured_control_guard"]) for row in rows]),
        "mean_effective_weight": mean([finite_float(row["effective_weight"], float("nan")) for row in rows]),
        "mean_interval_scale": mean([finite_float(row["interval_scale"], float("nan")) for row in rows]),
        "smooth_width_score_mean": mean([finite_float(row["smooth_width_score"], float("nan")) for row in rows]),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for variant_id in [str(variant["variant_id"]) for variant in VARIANTS]:
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        groups: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", variant_rows)]
        for key in ["evidence_tier", "family", "role", "target_id", "source"]:
            for value in sorted({str(row[key]) for row in variant_rows}):
                groups.append((f"{key}:{value}", key, [row for row in variant_rows if str(row[key]) == value]))
        for tier in sorted({str(row["evidence_tier"]) for row in variant_rows}):
            for role in sorted({str(row["role"]) for row in variant_rows if row["evidence_tier"] == tier}):
                groups.append(
                    (
                        f"evidence_tier:{tier}|role:{role}",
                        "tier_role",
                        [row for row in variant_rows if row["evidence_tier"] == tier and row["role"] == role],
                    )
                )
        for family in sorted({str(row["family"]) for row in variant_rows}):
            for role in sorted({str(row["role"]) for row in variant_rows if row["family"] == family}):
                groups.append(
                    (
                        f"family:{family}|role:{role}",
                        "family_role",
                        [row for row in variant_rows if row["family"] == family and row["role"] == role],
                    )
                )
        for group, group_type, subset in groups:
            if subset:
                summary.append(summarize_variant(subset, group, group_type))
    return summary


def summary_lookup(summary: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["variant_id"]), str(row["group"])): row for row in summary}


def report_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = summary_lookup(summary)
    wanted_groups = [
        "overall",
        "evidence_tier:q9_fullgrid|role:failure_target",
        "evidence_tier:q9_fullgrid|role:positive_control",
        "family:timesfm|role:failure_target",
        "family:timesfm|role:stress_target",
        "family:timesfm|role:positive_control",
        "target_id:finance_fred_stress",
    ]
    rows: list[dict[str, object]] = []
    for group in wanted_groups:
        for variant in VARIANTS:
            row = lookup.get((str(variant["variant_id"]), group))
            if row is None:
                continue
            rows.append(
                {
                    "Group": group,
                    "Variant": row["variant_id"],
                    "N": row["n_windows"],
                    "Levels": row["quantile_grid_n_levels"],
                    "ModelWQL": num(row["model_median_wql_rer"]),
                    "VariantWQL": num(row["variant_median_wql_rer"]),
                    "dWQL": num(row["variant_median_wql_delta_vs_model"]),
                    "FailRed": pct(row["wql_failure_reduction_vs_model"]),
                    "Coverage": num(row["variant_mean_coverage"]),
                    "WQLHarm": pct(row["variant_wql_noninferiority_harm_rate"]),
                    "Safety": pct(row["variant_safety_harm_rate"]),
                    "Struct": pct(row["structured_guard_rate"]),
                    "Scale": num(row["mean_interval_scale"]),
                }
            )
    return rows


def diagnostic_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = summary_lookup(summary)
    groups = [
        "evidence_tier:q9_fullgrid|role:positive_control",
        "family:timesfm|role:stress_target",
        "target_id:finance_fred_stress",
        "evidence_tier:q9_fullgrid|role:failure_target",
        "family:timesfm|role:failure_target",
    ]
    output: list[dict[str, object]] = []
    for group in groups:
        native = lookup[("native_model", group)]
        interval = lookup[("interval_only", group)]
        point = lookup[("point_shift_only", group)]
        full = lookup[("full_drcr_smooth", group)]
        output.append(
            {
                "Group": group,
                "N": native["n_windows"],
                "Native": num(native["variant_median_wql_rer"]),
                "IntervalOnly": num(interval["variant_median_wql_rer"]),
                "PointOnly": num(point["variant_median_wql_rer"]),
                "Full": num(full["variant_median_wql_rer"]),
                "IntervalHarm": pct(interval["variant_wql_noninferiority_harm_rate"]),
                "PointHarm": pct(point["variant_wql_noninferiority_harm_rate"]),
                "FullHarm": pct(full["variant_wql_noninferiority_harm_rate"]),
            }
        )
    return output


def write_report(summary: list[dict[str, object]], status: dict[str, object]) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR-Smooth Component Ablation",
                "",
                "## Purpose",
                "",
                "This decomposes frozen DRCR-Smooth into point-shift and interval-scale components. It is diagnostic only: the DRCR-Smooth candidate class remains frozen and no new smooth-head parameters are introduced.",
                "",
                "## Protocol",
                "",
                f"- Frozen protocol: `{status['protocol_id']}`.",
                f"- Protocol SHA-256: `{status['protocol_sha256']}`.",
                f"- Windows: `{status['n_windows']}` expansion windows from `{status['n_sources']}` sources.",
                "- Variants: native model, interval-only, point-shift-only, full DRCR-Smooth.",
                "",
                "## Headline Decomposition",
                "",
                markdown_table(
                    diagnostic_rows(summary),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Native", "Native"),
                        ("IntervalOnly", "Interval only"),
                        ("PointOnly", "Point only"),
                        ("Full", "Full"),
                        ("IntervalHarm", "Interval harm"),
                        ("PointHarm", "Point harm"),
                        ("FullHarm", "Full harm"),
                    ],
                ),
                "",
                "## Full Summary",
                "",
                markdown_table(
                    report_rows(summary),
                    [
                        ("Group", "Group"),
                        ("Variant", "Variant"),
                        ("N", "N"),
                        ("Levels", "Q"),
                        ("ModelWQL", "Model"),
                        ("VariantWQL", "Variant"),
                        ("dWQL", "Median dWQL"),
                        ("FailRed", "Fail red."),
                        ("Coverage", "Coverage"),
                        ("WQLHarm", "WQL harm"),
                        ("Safety", "Safety"),
                        ("Struct", "Struct guard"),
                        ("Scale", "Mean scale"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- Use this report to decide whether the next safety layer should veto point shifts, interval scaling, or the full intervention.",
                "- If interval-only already harms positive controls, the safety fix should target structured-control interval scaling rather than only CPR point routing.",
                "- If point-only harms finance/stress, the safety fix should additionally constrain CPR point shifts on those ex-ante regimes.",
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
    rows = build_rows()
    summary = build_summary(rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "protocol_id": smooth.PROTOCOL_ID,
        "protocol_config": str(smooth.CONFIG_PATH.relative_to(ROOT)),
        "protocol_sha256": smooth.protocol_sha256(),
        "selected_candidate_id": smooth.SMOOTH_CANDIDATE_ID,
        "n_windows": len(rows) // len(VARIANTS),
        "n_variant_rows": len(rows),
        "n_sources": len({row["source"] for row in rows}),
        "variants": [variant["variant_id"] for variant in VARIANTS],
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, rows)
    write_csv(SUMMARY_OUT, summary)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
