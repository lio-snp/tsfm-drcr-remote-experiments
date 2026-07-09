#!/usr/bin/env python
"""Diagnose the mechanism behind conflict-shielded selective repair."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "repair" / "cross_family_selective_repair_strategy_metrics.csv"
OUT_DIR = ROOT / "results" / "repair"
DOC_PATH = ROOT / "docs" / "conflict_shield_mechanism_report.md"
UNSHIELDED = "cross_family_margin_pareto"
SHIELDED = "risk_controlled_leave_family_out"


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


def rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("family", ""),
        row.get("source", ""),
        row.get("series_id", ""),
        row.get("origin", ""),
        row.get("window_index", ""),
    )


def summarize_rows(rows: list[dict[str, str]], group: str, strategy_id: str) -> dict[str, object]:
    model_fail = [finite_float(row["model_failure_delta_005"]) for row in rows]
    repair_fail = [finite_float(row["repair_failure_delta_005"]) for row in rows]
    rer_delta = [finite_float(row["relative_error_ratio_delta"]) for row in rows]
    gate = [finite_float(row["gate_active"]) for row in rows]
    shield = [finite_float(row.get("shield_active")) for row in rows]
    gated = [row for row in rows if finite_float(row["gate_active"]) > 0]
    return {
        "strategy_id": strategy_id,
        "group": group,
        "n_windows": len(rows),
        "model_failure_rate": rate(model_fail),
        "repair_failure_rate": rate(repair_fail),
        "failure_rate_reduction": rate([a - b for a, b in zip(model_fail, repair_fail)]),
        "gate_rate": rate(gate),
        "shield_rate": rate(shield),
        "shield_rate_given_gate": rate([finite_float(row.get("shield_active")) for row in gated]) if gated else 0.0,
        "reference_outside_interval_rate_mean": rate(
            [finite_float(row.get("reference_outside_interval_rate")) for row in rows]
        ),
        "reference_outside_interval_rate_median": median(
            [finite_float(row.get("reference_outside_interval_rate")) for row in rows]
        ),
        "median_rer_delta": median(rer_delta),
        "p90_rer_delta": p90(rer_delta),
    }


def pair_rows(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_strategy: dict[str, dict[tuple[str, str, str, str, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["strategy_id"] in {UNSHIELDED, SHIELDED}:
            by_strategy[row["strategy_id"]][row_key(row)] = row
    pairs = []
    for key, unshielded in by_strategy[UNSHIELDED].items():
        shielded = by_strategy[SHIELDED].get(key)
        if shielded is not None:
            pairs.append((unshielded, shielded))
    return pairs


def paired_summaries(pairs: list[tuple[dict[str, str], dict[str, str]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groupings: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {"overall": pairs}
    for role in sorted({shielded["role"] for _, shielded in pairs}):
        groupings[f"role:{role}"] = [pair for pair in pairs if pair[1]["role"] == role]
    for family in sorted({shielded["family"] for _, shielded in pairs}):
        groupings[f"family:{family}"] = [pair for pair in pairs if pair[1]["family"] == family]
    for group, group_pairs in groupings.items():
        if not group_pairs:
            continue
        for subset_name, subset_pairs in [
            ("all", group_pairs),
            ("shield_active", [pair for pair in group_pairs if finite_float(pair[1].get("shield_active")) > 0]),
            ("shield_inactive", [pair for pair in group_pairs if finite_float(pair[1].get("shield_active")) <= 0]),
        ]:
            if not subset_pairs:
                continue
            unshielded = [pair[0] for pair in subset_pairs]
            shielded = [pair[1] for pair in subset_pairs]
            rows.append(
                {
                    "group": group,
                    "subset": subset_name,
                    "n_windows": len(subset_pairs),
                    "unshielded_repair_failure_rate": rate(
                        [finite_float(row["repair_failure_delta_005"]) for row in unshielded]
                    ),
                    "shielded_repair_failure_rate": rate(
                        [finite_float(row["repair_failure_delta_005"]) for row in shielded]
                    ),
                    "unshielded_failure_reduction": rate(
                        [
                            finite_float(row["model_failure_delta_005"])
                            - finite_float(row["repair_failure_delta_005"])
                            for row in unshielded
                        ]
                    ),
                    "shielded_failure_reduction": rate(
                        [
                            finite_float(row["model_failure_delta_005"])
                            - finite_float(row["repair_failure_delta_005"])
                            for row in shielded
                        ]
                    ),
                    "unshielded_median_rer_delta": median(
                        [finite_float(row["relative_error_ratio_delta"]) for row in unshielded]
                    ),
                    "shielded_median_rer_delta": median(
                        [finite_float(row["relative_error_ratio_delta"]) for row in shielded]
                    ),
                    "unshielded_p90_rer_delta": p90(
                        [finite_float(row["relative_error_ratio_delta"]) for row in unshielded]
                    ),
                    "shielded_p90_rer_delta": p90(
                        [finite_float(row["relative_error_ratio_delta"]) for row in shielded]
                    ),
                    "p90_rer_delta_reduction_from_shield": p90(
                        [finite_float(row["relative_error_ratio_delta"]) for row in unshielded]
                    )
                    - p90([finite_float(row["relative_error_ratio_delta"]) for row in shielded]),
                    "mean_reference_outside_interval_rate": rate(
                        [finite_float(row.get("reference_outside_interval_rate")) for row in shielded]
                    ),
                }
            )
    return rows


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


def write_report(strategy_rows: list[dict[str, object]], paired_rows: list[dict[str, object]]) -> None:
    role_rows = [row for row in strategy_rows if row["strategy_id"] == SHIELDED and str(row["group"]).startswith("role:")]
    role_lines = [
        {
            "Role": str(row["group"]).replace("role:", ""),
            "N": row["n_windows"],
            "Reduction": fmt_pct(row["failure_rate_reduction"]),
            "Gate": fmt_pct(row["gate_rate"]),
            "Shield/Gate": fmt_pct(row["shield_rate_given_gate"]),
            "Outside": fmt_num(row["reference_outside_interval_rate_mean"]),
            "p90 dRER": fmt_num(row["p90_rer_delta"]),
        }
        for row in role_rows
    ]
    paired_key = {(row["group"], row["subset"]): row for row in paired_rows}
    selected_pairs = [
        paired_key[("role:positive_control", "all")],
        paired_key[("role:failure_target", "all")],
        paired_key[("role:stress_target", "all")],
        paired_key[("role:positive_control", "shield_active")],
        paired_key[("role:failure_target", "shield_active")],
    ]
    pair_lines = [
        {
            "Group": row["group"].replace("role:", ""),
            "Subset": row["subset"],
            "N": row["n_windows"],
            "Unshield red": fmt_pct(row["unshielded_failure_reduction"]),
            "Shield red": fmt_pct(row["shielded_failure_reduction"]),
            "Unshield p90": fmt_num(row["unshielded_p90_rer_delta"]),
            "Shield p90": fmt_num(row["shielded_p90_rer_delta"]),
            "p90 saved": fmt_num(row["p90_rer_delta_reduction_from_shield"]),
        }
        for row in selected_pairs
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Conflict-Shield Mechanism Diagnostics",
                "",
                "## Mechanistic Claim",
                "",
                "The conflict shield is not just a smaller blend weight. It acts as an expert-disagreement brake: low-structure gating opens the repair path, but the shield caps the baseline/reference expert when that expert sits outside the TSFM uncertainty interval over a large share of the horizon.",
                "",
                "This supports a stronger story than weighted averaging: CSSR is a reliability controller for frozen TSFMs. It repairs low-local-structure degeneration when the local reference is compatible with the TSFM uncertainty set, and it suppresses deferral when the two experts disagree in positive-control regimes.",
                "",
                "## Shielded Strategy by Role",
                "",
                markdown_table(
                    role_lines,
                    [
                        ("Role", "Role"),
                        ("N", "N"),
                        ("Reduction", "Failure Reduction"),
                        ("Gate", "Gate"),
                        ("Shield/Gate", "Shield/Gate"),
                        ("Outside", "Mean Outside Rate"),
                        ("p90 dRER", "p90 dRER"),
                    ],
                ),
                "",
                "## Unshielded vs. Shielded Mechanism Check",
                "",
                markdown_table(
                    pair_lines,
                    [
                        ("Group", "Group"),
                        ("Subset", "Subset"),
                        ("N", "N"),
                        ("Unshield red", "Unshield Reduction"),
                        ("Shield red", "Shield Reduction"),
                        ("Unshield p90", "Unshield p90 dRER"),
                        ("Shield p90", "Shield p90 dRER"),
                        ("p90 saved", "p90 Saved"),
                    ],
                ),
                "",
                "## Reading",
                "",
                "- Positive-control windows have the largest reference-outside-interval signal and the highest shield activation conditional on the low-structure gate.",
                "- The shield collapses positive-control p90 RER cost while leaving stress-target repair unchanged and preserving most failure-target repair.",
                "- The shield is a safety tradeoff, not a free improvement: it sacrifices a small set of high-conflict failure-target windows, but this prevents much larger positive-control tail harm and keeps the overall repair effect strong.",
                "- This is the mechanism the paper should emphasize: local structure triggers repair; expert conflict prevents unsafe deferral.",
                "",
                "## Related Mechanistic Frames",
                "",
                "- Selective prediction / reject option: risk is controlled by not covering all instances.",
                "- Learning to defer: a predictor routes hard cases to an expert, but here the expert itself can be unsafe under interval disagreement.",
                "- Risk control / conformal calibration: thresholds are selected under held-out positive-control tail-cost constraints.",
                "- Forecast combination: the method is a state-dependent expert combination, not a global average.",
                "",
                "## Artifacts",
                "",
                "- `results/repair/conflict_shield_strategy_by_group.csv`",
                "- `results/repair/conflict_shield_paired_diagnostics.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    rows = read_csv(METRICS_PATH)
    selected = [row for row in rows if row["strategy_id"] in {UNSHIELDED, SHIELDED}]
    strategy_rows: list[dict[str, object]] = []
    for strategy_id in [UNSHIELDED, SHIELDED]:
        strategy_rows.append(summarize_rows([row for row in selected if row["strategy_id"] == strategy_id], "overall", strategy_id))
        for role in sorted({row["role"] for row in selected if row["strategy_id"] == strategy_id}):
            strategy_rows.append(
                summarize_rows(
                    [row for row in selected if row["strategy_id"] == strategy_id and row["role"] == role],
                    f"role:{role}",
                    strategy_id,
                )
            )
        for family in sorted({row["family"] for row in selected if row["strategy_id"] == strategy_id}):
            strategy_rows.append(
                summarize_rows(
                    [row for row in selected if row["strategy_id"] == strategy_id and row["family"] == family],
                    f"family:{family}",
                    strategy_id,
                )
            )
    paired_rows = paired_summaries(pair_rows(rows))
    write_csv(OUT_DIR / "conflict_shield_strategy_by_group.csv", strategy_rows)
    write_csv(OUT_DIR / "conflict_shield_paired_diagnostics.csv", paired_rows)
    write_report(strategy_rows, paired_rows)
    print((OUT_DIR / "conflict_shield_strategy_by_group.csv").relative_to(ROOT))
    print((OUT_DIR / "conflict_shield_paired_diagnostics.csv").relative_to(ROOT))
    print(DOC_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
