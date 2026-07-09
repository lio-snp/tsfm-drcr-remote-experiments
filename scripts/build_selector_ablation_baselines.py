#!/usr/bin/env python3
"""Build selector-ablation baselines from already evaluated final-main candidates."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"
CANDIDATES = OUT / "final_main_figure_candidates.csv"
CALIBRATION = OUT / "final_main_figure_calibration_tests.csv"
SUMMARY = OUT / "final_main_figure_summary.csv"
OUT_CSV = OUT / "selector_ablation_baselines.csv"
OUT_MD = DOCS / "selector_ablation_baselines.md"

SELECTED = "drcr_expert_pull_1.25_cap_1.10"
Q9 = "evidence_tier:q9_fullgrid|role:failure_target"

RULES = [
    ("no_risk_highest_utility", "Ignore all risk tests; choose highest calibration utility among eligible DRCR candidates."),
    ("wql_only", "Require only WQL non-inferiority harm LTT acceptance."),
    ("harm_only", "Require WQL harm and protected harm acceptance; ignore undercoverage."),
    ("undercoverage_only", "Require only undercoverage non-inferiority harm acceptance."),
    ("absolute_coverage_only", "Require strict absolute undercoverage audit; if none pass, choose lowest absolute undercoverage risk."),
    ("dual_tri_risk", "Require WQL harm, protected harm, and undercoverage harm acceptance."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: object) -> float:
    return float(value)


def num(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    candidates = {row["candidate_id"]: row for row in read_csv(CANDIDATES)}
    eligible = {
        cid: row
        for cid, row in candidates.items()
        if row.get("eligible_for_selection") == "1"
    }
    risk = {}
    for row in read_csv(CALIBRATION):
        risk.setdefault(row["candidate_id"], {})[row["risk_name"]] = row
    summary = {
        (row["candidate_id"], row["phase"], row["group"]): row
        for row in read_csv(SUMMARY)
    }

    def accepted(cid: str, risk_name: str) -> bool:
        return risk[cid][risk_name]["accepted"] == "1"

    def utility(cid: str) -> float:
        return f(candidates[cid]["calibration_utility"])

    def choose(rule: str) -> tuple[str, str]:
        pool = list(eligible)
        if rule == "no_risk_highest_utility":
            return max(pool, key=utility), "selected highest calibration utility, ignoring risk"
        if rule == "wql_only":
            accepted_pool = [cid for cid in pool if accepted(cid, "wql_noninferiority_harm")]
            return max(accepted_pool, key=utility), "selected among WQL-risk accepted candidates"
        if rule == "harm_only":
            accepted_pool = [
                cid
                for cid in pool
                if accepted(cid, "wql_noninferiority_harm") and accepted(cid, "protected_wql_harm")
            ]
            return max(accepted_pool, key=utility), "selected among WQL+protected harm accepted candidates"
        if rule == "undercoverage_only":
            accepted_pool = [cid for cid in pool if accepted(cid, "undercoverage_noninferiority_harm")]
            return max(accepted_pool, key=utility), "selected among undercoverage-harm accepted candidates"
        if rule == "absolute_coverage_only":
            accepted_pool = [cid for cid in pool if accepted(cid, "absolute_undercoverage_audit")]
            if accepted_pool:
                return max(accepted_pool, key=utility), "selected among absolute-undercoverage accepted candidates"
            return min(pool, key=lambda cid: f(risk[cid]["absolute_undercoverage_audit"]["empirical_risk"])), "no candidate passed strict absolute coverage; chose lowest absolute undercoverage risk"
        if rule == "dual_tri_risk":
            accepted_pool = [
                cid
                for cid in pool
                if accepted(cid, "wql_noninferiority_harm")
                and accepted(cid, "protected_wql_harm")
                and accepted(cid, "undercoverage_noninferiority_harm")
            ]
            return max(accepted_pool, key=utility), "selected among tri-risk accepted candidates"
        raise ValueError(rule)

    rows = []
    for rule, description in RULES:
        cid, note = choose(rule)
        overall = summary[(cid, "test", "overall")]
        q9 = summary[(cid, "test", Q9)]
        rows.append(
            {
                "selector_rule": rule,
                "description": description,
                "selected_candidate_id": cid,
                "matches_current_selected": int(cid == SELECTED),
                "tri_risk_accepted": candidates[cid]["tri_risk_accepted"],
                "selection_note": note,
                "calibration_utility": candidates[cid]["calibration_utility"],
                "calibration_wql_harm": risk[cid]["wql_noninferiority_harm"]["empirical_risk"],
                "calibration_protected_harm": risk[cid]["protected_wql_harm"]["empirical_risk"],
                "calibration_uc_harm": risk[cid]["undercoverage_noninferiority_harm"]["empirical_risk"],
                "calibration_abs_undercoverage": risk[cid]["absolute_undercoverage_audit"]["empirical_risk"],
                "overall_wql_rer": overall["repair_median_wql_rer"],
                "overall_harm": overall["wql_harm_rate"],
                "overall_coverage": overall["repair_mean_coverage"],
                "q9_wql_rer": q9["repair_median_wql_rer"],
                "q9_harm": q9["wql_harm_rate"],
                "q9_coverage": q9["repair_mean_coverage"],
            }
        )
    write_csv(OUT_CSV, rows)

    table_rows = [
        {
            "Rule": row["selector_rule"],
            "Selected": row["selected_candidate_id"],
            "Tri": "yes" if row["tri_risk_accepted"] == "1" else "no",
            "Overall": num(row["overall_wql_rer"]),
            "OHarm": pct(row["overall_harm"]),
            "Q9": num(row["q9_wql_rer"]),
            "Q9Cov": pct(row["q9_coverage"]),
            "Note": row["selection_note"],
        }
        for row in rows
    ]
    lines = [
        "# Selector-Ablation Baselines",
        "",
        "These baselines are derived from the already evaluated final-main candidate set. They test whether the proposed tri-risk selector is doing more than selecting a single obvious fallback rule.",
        "",
        markdown_table(
            table_rows,
            [
                ("Rule", "Selector rule"),
                ("Selected", "Selected candidate"),
                ("Tri", "Tri-risk"),
                ("Overall", "Overall WQL"),
                ("OHarm", "Overall harm"),
                ("Q9", "q9 WQL"),
                ("Q9Cov", "q9 coverage"),
                ("Note", "Selection note"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Coverage-only selection falls back to the lowest absolute-undercoverage candidate because no eligible candidate passes strict absolute coverage; this points toward `drcr_full`, which fails the WQL/protected harm screen.",
        "- WQL-only, harm-only, undercoverage-only, and tri-risk selection collapse to the same selected candidate under the current utility, which is good news for this suite but should be reported as a result rather than assumed generally.",
        "- The ablation supports the story that the most dangerous alternative is not a WQL-only selector here, but coverage/full-repair thinking and unconditional fallback.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_MD.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

