#!/usr/bin/env python3
"""Build a compact objective-baseline comparison from final-main artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "aaai_stress" / "final_main_figure_summary.csv"
CANDIDATES = ROOT / "results" / "aaai_stress" / "final_main_figure_candidates.csv"
OUT_CSV = ROOT / "results" / "aaai_stress" / "main_baseline_comparison.csv"
OUT_MD = ROOT / "docs" / "main_baseline_comparison.md"

GROUPS = [
    ("overall", "Overall"),
    ("evidence_tier:q9_fullgrid|role:failure_target", "q9 full-grid failure"),
    ("role:positive_control", "Positive controls"),
    ("role:stress_target", "Stress targets"),
    ("target_id:finance_fred_stress", "Finance FRED stress"),
]

METHODS = [
    ("native_tsfm", "anchor", "Native TSFM"),
    ("classical_deterministic", "objective_baseline", "Classical deterministic"),
    ("classical_residual_calibrated", "objective_baseline", "Classical residual calibrated"),
    ("global_blend_w0.50", "objective_baseline", "Global 50/50 blend"),
    ("smooth_score_gate_t0.50_w1.00", "objective_baseline", "Simple score gate"),
    ("width_gate_t0.10_w1.00", "objective_baseline", "Simple width gate"),
    ("oracle_native_classical_drcr", "upper_bound", "Oracle upper bound"),
    ("drcr_full", "ablation", "Full DRCR"),
    ("drcr_point", "ablation", "Point-only DRCR"),
    ("drcr_cap_1.10", "ablation", "Cap 1.10 DRCR"),
    ("drcr_expert_pull_1.25_cap_1.10", "proposed", "DRCR-ExpertPull selected"),
    ("drcr_width_veto_expert_pull_1.50_cap_1.10", "frontier_probe", "DRCR-WidthVeto probe"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: str, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    summary = {
        (row["candidate_id"], row["phase"], row["group"]): row
        for row in read_csv(SUMMARY)
    }
    candidates = {row["candidate_id"]: row for row in read_csv(CANDIDATES)}
    rows: list[dict[str, object]] = []
    for group_key, group_label in GROUPS:
        for candidate_id, category, label in METHODS:
            row = summary[(candidate_id, "test", group_key)]
            cand = candidates[candidate_id]
            rows.append(
                {
                    "group": group_key,
                    "group_label": group_label,
                    "candidate_id": candidate_id,
                    "label": label,
                    "category": category,
                    "selected": cand.get("selected", "0"),
                    "tri_risk_accepted": cand.get("tri_risk_accepted", "0"),
                    "eligible_for_selection": cand.get("eligible_for_selection", ""),
                    "n_windows": row["n_windows"],
                    "wql_rer": row["repair_median_wql_rer"],
                    "wql_rer_ci_low": row["repair_median_wql_rer_ci_low"],
                    "wql_rer_ci_high": row["repair_median_wql_rer_ci_high"],
                    "wql_harm_rate": row["wql_harm_rate"],
                    "coverage_q10_q90": row["repair_mean_coverage"],
                    "mae_rer": row["repair_median_mae_rer"],
                    "rmse_rer": row["repair_median_rmse_rer"],
                    "wape_rer": row["repair_median_wape_rer"],
                }
            )
    write_csv(OUT_CSV, rows)

    by_group = {(row["candidate_id"], row["group"]): row for row in rows}
    selected = by_group[("drcr_expert_pull_1.25_cap_1.10", "overall")]
    score_gate_q9 = by_group[("smooth_score_gate_t0.50_w1.00", "evidence_tier:q9_fullgrid|role:failure_target")]
    score_gate_overall = by_group[("smooth_score_gate_t0.50_w1.00", "overall")]
    oracle_q9 = by_group[("oracle_native_classical_drcr", "evidence_tier:q9_fullgrid|role:failure_target")]

    lines = [
        "# Objective Baseline Comparison",
        "",
        "Generated from the frozen final-main artifacts. These baselines are comparison policies, not selectable DRCR methods unless the frozen protocol explicitly marks them eligible.",
        "",
        "## Key Read",
        "",
        f"- Selected DRCR remains `drcr_expert_pull_1.25_cap_1.10`; overall WQL-RER `{num(selected['wql_rer'])}`, harm `{pct(selected['wql_harm_rate'])}`, coverage `{pct(selected['coverage_q10_q90'])}`.",
        f"- The simple score gate is a useful stress test: q9 failure WQL-RER `{num(score_gate_q9['wql_rer'])}`, but overall WQL-RER `{num(score_gate_overall['wql_rer'])}`. This means a simple gate can attack failure windows but is not a clean overall replacement for calibrated DRCR.",
        f"- Oracle upper bound q9 failure WQL-RER is `{num(oracle_q9['wql_rer'])}`, showing substantial remaining headroom and motivating better calibrated selectors.",
        "- Global blend and width gate expose why objective baselines are needed: some improve one slice while failing calibration or protected-window safety.",
        "",
    ]

    for group_key, group_label in GROUPS:
        group_rows = []
        for candidate_id, category, label in METHODS:
            row = by_group[(candidate_id, group_key)]
            group_rows.append(
                {
                    "Method": label,
                    "Cat": category,
                    "Tri": "yes" if row["tri_risk_accepted"] == "1" else "no",
                    "WQL": num(row["wql_rer"]),
                    "Harm": pct(row["wql_harm_rate"]),
                    "Cov": pct(row["coverage_q10_q90"]),
                    "MAE": num(row["mae_rer"]),
                }
            )
        lines.extend(
            [
                f"## {group_label}",
                "",
                markdown_table(
                    group_rows,
                    [
                        ("Method", "Method"),
                        ("Cat", "Type"),
                        ("Tri", "Tri-risk"),
                        ("WQL", "WQL-RER"),
                        ("Harm", "Harm"),
                        ("Cov", "Coverage"),
                        ("MAE", "MAE-RER"),
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{OUT_CSV.relative_to(ROOT)}`",
            f"- `{OUT_MD.relative_to(ROOT)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

