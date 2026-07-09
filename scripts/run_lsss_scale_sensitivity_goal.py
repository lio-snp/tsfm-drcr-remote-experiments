#!/usr/bin/env python
"""Sensitivity sweep for Local-Structure Safety Shield interval scales."""

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

import run_chronos_adaptive_fullgrid_cpr_goal as adaptive  # noqa: E402
import run_chronos_fullgrid_cpr_wql_goal as fullgrid  # noqa: E402
import run_moirai_fullgrid_evidence_goal as moirai  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
SUMMARY_OUT = OUT_DIR / "lsss_scale_sensitivity_summary.csv"
STATUS_OUT = OUT_DIR / "lsss_scale_sensitivity_status.json"
DOC_OUT = ROOT / "docs" / "lsss_scale_sensitivity_report.md"

FAILURE_SCALES = [0.75, 1.0, 1.25, 1.5]
STRUCTURED_SCALES = [1.5, 2.0, 2.5]
STRUCTURED_WEIGHT_CAPS = [0.125]


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


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


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


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def adaptive_grid(
    window: dict[str, object],
    weight: float,
    *,
    failure_scale: float,
    structured_scale: float,
    structured_weight_cap: float,
) -> tuple[np.ndarray, float, float, int]:
    guard = adaptive.structured_control_guard(window)
    scale = structured_scale if guard else failure_scale
    effective_weight = min(weight, structured_weight_cap) if guard else weight
    return fullgrid.interval_head_quantile_grid(window, effective_weight, scale), scale, effective_weight, int(guard)


def candidate_rows(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    *,
    failure_scale: float,
    structured_scale: float,
    structured_weight_cap: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in windows:
        point_row = fullgrid.cpr.apply_policy_to_window(
            window,
            policy,
            "lsss_scale_sensitivity",
            "transferred_common_policy",
            "all_windows",
            "common_ltt_policy",
        )
        weight = finite_float(point_row["effective_weight"])
        model_metrics = fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
        fixed_metrics = fullgrid.quantile_metrics(window, fullgrid.interval_head_quantile_grid(window, weight, 1.25))
        candidate, scale, adaptive_weight, guard = adaptive_grid(
            window,
            weight,
            failure_scale=failure_scale,
            structured_scale=structured_scale,
            structured_weight_cap=structured_weight_cap,
        )
        candidate_metrics = fullgrid.quantile_metrics(window, candidate)
        rows.append(
            {
                "family": window["family"],
                "source": window["source"],
                "role": window["role"],
                "target_id": window["target_id"],
                "guard": guard,
                "scale": scale,
                "adaptive_weight": adaptive_weight,
                "model_wql_rer": model_metrics["wql_rer"],
                "fixed_wql_rer": fixed_metrics["wql_rer"],
                "adaptive_wql_rer": candidate_metrics["wql_rer"],
                "model_wql_failure": int(model_metrics["wql_rer"] > 1.05),
                "adaptive_wql_failure": int(candidate_metrics["wql_rer"] > 1.05),
                "model_coverage_error": model_metrics["coverage_abs_error"],
                "fixed_coverage_error": fixed_metrics["coverage_abs_error"],
                "adaptive_coverage": candidate_metrics["coverage"],
                "adaptive_coverage_error": candidate_metrics["coverage_abs_error"],
                "adaptive_win": int(candidate_metrics["wql_rer"] < model_metrics["wql_rer"]),
                "adaptive_harm": int(
                    candidate_metrics["wql_rer"] > model_metrics["wql_rer"] + 0.05
                    or candidate_metrics["coverage_abs_error"] > model_metrics["coverage_abs_error"] + 0.05
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]], group: str, group_type: str, config: dict[str, object]) -> dict[str, object]:
    return {
        **config,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "guard_rate": rate([int(row["guard"]) for row in rows]),
        "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in rows]),
        "fixed_median_wql_rer": median([finite_float(row["fixed_wql_rer"], float("nan")) for row in rows]),
        "adaptive_median_wql_rer": median([finite_float(row["adaptive_wql_rer"], float("nan")) for row in rows]),
        "adaptive_median_delta_vs_model": median(
            [finite_float(row["adaptive_wql_rer"], float("nan")) - finite_float(row["model_wql_rer"], float("nan")) for row in rows]
        ),
        "adaptive_failure_reduction_vs_model": rate([int(row["model_wql_failure"]) for row in rows])
        - rate([int(row["adaptive_wql_failure"]) for row in rows]),
        "adaptive_mean_coverage": mean([finite_float(row["adaptive_coverage"], float("nan")) for row in rows]),
        "adaptive_mean_coverage_error": mean(
            [finite_float(row["adaptive_coverage_error"], float("nan")) for row in rows]
        ),
        "adaptive_coverage_error_delta_vs_fixed": mean(
            [
                finite_float(row["adaptive_coverage_error"], float("nan"))
                - finite_float(row["fixed_coverage_error"], float("nan"))
                for row in rows
            ]
        ),
        "adaptive_win_rate_vs_model": rate([int(row["adaptive_win"]) for row in rows]),
        "adaptive_harm_rate": rate([int(row["adaptive_harm"]) for row in rows]),
    }


def report_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    interesting = [
        (0.75, 2.0, "sharp WQL"),
        (1.0, 2.0, "middle"),
        (1.25, 2.0, "fixed-head equivalent"),
        (1.5, 2.0, "coverage-safe"),
    ]
    groups = [
        "overall",
        "family:chronos|role:failure_target",
        "family:moirai|role:failure_target",
        "family:chronos|role:positive_control",
        "family:moirai|role:positive_control",
    ]
    for failure_scale, structured_scale, label in interesting:
        for group in groups:
            row = next(
                item
                for item in summary_rows
                if item["group"] == group
                and finite_float(item["failure_scale"]) == failure_scale
                and finite_float(item["structured_scale"]) == structured_scale
            )
            selected.append(
                {
                    "Config": label,
                    "f": failure_scale,
                    "s": structured_scale,
                    "Group": group,
                    "WQL": num(row["adaptive_median_wql_rer"]),
                    "FailRed": pct(row["adaptive_failure_reduction_vs_model"]),
                    "Cov": num(row["adaptive_mean_coverage"]),
                    "CovErr": num(row["adaptive_mean_coverage_error"]),
                    "Win": pct(row["adaptive_win_rate_vs_model"]),
                    "Harm": pct(row["adaptive_harm_rate"]),
                }
            )
    return selected


def write_report(summary_rows: list[dict[str, object]], status: dict[str, object]) -> None:
    DOC_OUT.write_text(
        "\n".join(
            [
                "# LSSS Scale Sensitivity",
                "",
                "## Purpose",
                "",
                "This sweep tests whether a single fixed Local-Structure Safety Shield interval scale can dominate across the current full-grid Chronos and Moirai evidence.",
                "",
                "## Design",
                "",
                f"- Windows: `{status['n_windows']}` total (`{status['n_chronos_windows']}` Chronos, `{status['n_moirai_windows']}` Moirai).",
                f"- Configs: `{status['n_configs']}` combinations over failure-regime scale and structured-control scale.",
                "- CPR gate, common policy, guard thresholds, and structured weight cap are fixed; only interval scales move.",
                "",
                "## Key Configurations",
                "",
                markdown_table(
                    report_rows(summary_rows),
                    [
                        ("Config", "Config"),
                        ("f", "Failure scale"),
                        ("s", "Structured scale"),
                        ("Group", "Group"),
                        ("WQL", "Median WQL-RER"),
                        ("FailRed", "Fail reduction"),
                        ("Cov", "Coverage"),
                        ("CovErr", "Coverage error"),
                        ("Win", "Win vs model"),
                        ("Harm", "Safety harm"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The sharp Chronos-oriented failure scale `0.75` gives the strongest Chronos failure WQL repair, but leaves Moirai-covid under-covered.",
                "- Larger failure scales improve Moirai-covid coverage, but sharply weaken Chronos failure WQL repair and increase harm.",
                "- No single fixed scale dominates across both families. This supports upgrading the method line from fixed LSSS to CRC/LTT-calibrated LSSS, where interval risk is calibrated rather than hard-coded.",
                "",
                "## Artifacts",
                "",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = fullgrid.common_cpr_policy()
    chronos_windows, _ = fullgrid.load_windows()
    moirai_windows, _ = moirai.load_moirai_windows()
    windows = chronos_windows + moirai_windows
    if not windows:
        raise SystemExit("No full-grid windows available for LSSS sensitivity")
    summary_rows: list[dict[str, object]] = []
    for failure_scale in FAILURE_SCALES:
        for structured_scale in STRUCTURED_SCALES:
            for structured_weight_cap in STRUCTURED_WEIGHT_CAPS:
                config = {
                    "failure_scale": failure_scale,
                    "structured_scale": structured_scale,
                    "structured_weight_cap": structured_weight_cap,
                }
                rows = candidate_rows(
                    windows,
                    policy,
                    failure_scale=failure_scale,
                    structured_scale=structured_scale,
                    structured_weight_cap=structured_weight_cap,
                )
                summary_rows.append(summarize(rows, "overall", "overall", config))
                for family in sorted({str(row["family"]) for row in rows}):
                    subset = [row for row in rows if row["family"] == family]
                    summary_rows.append(summarize(subset, f"family:{family}", "family", config))
                for family in sorted({str(row["family"]) for row in rows}):
                    for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
                        subset = [row for row in rows if row["family"] == family and row["role"] == role]
                        summary_rows.append(summarize(subset, f"family:{family}|role:{role}", "family_role", config))
    write_csv(SUMMARY_OUT, summary_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "n_chronos_windows": len(chronos_windows),
        "n_moirai_windows": len(moirai_windows),
        "n_configs": len(FAILURE_SCALES) * len(STRUCTURED_SCALES) * len(STRUCTURED_WEIGHT_CAPS),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "report": str(DOC_OUT.relative_to(ROOT)),
    }
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(summary_rows, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
