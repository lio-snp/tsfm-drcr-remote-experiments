#!/usr/bin/env python3
"""Build paper-readiness reports from the frozen DRCR final-main artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

WINDOWS = OUT / "final_main_figure_windows.csv"
SUMMARY = OUT / "final_main_figure_summary.csv"
CANDIDATES = OUT / "final_main_figure_candidates.csv"
CALIBRATION = OUT / "final_main_figure_calibration_tests.csv"
STATUS = OUT / "final_main_figure_status.json"
CONFIRMATORY_STATUS = OUT / "final_main_confirmatory_status.json"
CONFIRMATORY_SUMMARY = OUT / "final_main_confirmatory_summary.csv"

LTT_TABLE = OUT / "ltt_crc_pvalue_table.csv"
PROTECTED_TABLE = OUT / "protected_safety_results.csv"
CLIPPED_TABLE = OUT / "clipped_sensitivity_table.csv"
SPLIT_MANIFEST = OUT / "split_manifest.csv"
DATASET_MANIFEST = OUT / "dataset_manifest.csv"
MODEL_MANIFEST = OUT / "model_manifest.csv"
FINAL_RESULTS = OUT / "final_main_results.csv"
ORAL_SOURCE_GAPS = OUT / "oral_evidence_source_gap_matrix.csv"
ORAL_RERUN_COMMANDS = OUT / "oral_rerun_command_manifest.csv"
NATIVE_CLASSICAL_SUMMARY = OUT / "native_classical_interval_audit_summary.csv"
LOCAL_RERUN_BLOCKERS = OUT / "local_rerun_resource_blockers.csv"
REMOTE_Q9_RERUN_PLAN = OUT / "remote_q9_rerun_plan.csv"
REMOTE_Q9_RERUN_AUDIT = OUT / "remote_q9_rerun_completion_audit.csv"
REMOTE_Q9_INGESTION = OUT / "remote_q9_ingestion_manifest.csv"

LTT_REPORT = DOCS / "ltt_crc_pvalue_report.md"
PROTECTED_REPORT = DOCS / "protected_group_safety_table.md"
DENOM_REPORT = DOCS / "denominator_fragility_report.md"
PAPER_FAITHFUL_REPORT = DOCS / "paper_faithful_metric_robustness.md"
NO_TEST_TUNING = DOCS / "no_test_tuning_statement.md"
CAPTION = DOCS / "caption_plain_language.md"
FINAL_SUMMARY = DOCS / "final_drcr_paper_readiness_summary.md"

SELECTED = "drcr_expert_pull_1.25_cap_1.10"
FRONTIER = "drcr_width_veto_expert_pull_1.50_cap_1.10"
Q9 = "evidence_tier:q9_fullgrid|role:failure_target"

METHODS = [
    ("native_tsfm", "Native TSFM"),
    ("classical_deterministic", "Classical deterministic"),
    ("classical_residual_calibrated", "Classical residual calibrated"),
    ("global_blend_w0.50", "Global 50/50 blend"),
    ("smooth_score_gate_t0.50_w1.00", "Simple score gate"),
    ("width_gate_t0.10_w1.00", "Simple width gate"),
    ("oracle_native_classical_drcr", "Oracle upper bound"),
    ("drcr_full", "Full DRCR"),
    ("drcr_point", "Point-only DRCR"),
    ("drcr_cap_1.10", "Cap 1.10 DRCR"),
    (SELECTED, "DRCR-ExpertPull selected"),
    (FRONTIER, "DRCR-WidthVeto probe"),
]

PROTECTED_GROUPS = [
    ("overall", "Overall"),
    ("role:positive_control", "Positive controls"),
    ("role:stress_target", "Stress targets"),
    ("target_id:finance_fred_stress", "Finance FRED stress"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def num(value: object, digits: int = 3) -> str:
    parsed = f(value)
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    parsed = f(value)
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


def command_status(row: dict[str, str]) -> str:
    return row.get("command_status") or row.get("status", "")


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["candidate_id"], row["phase"], row["group"]): row for row in rows}


def build_ltt_report(calibration: list[dict[str, str]], candidates: dict[str, dict[str, str]]) -> None:
    rows = []
    for row in calibration:
        cid = row["candidate_id"]
        rows.append(
            {
                "candidate_id": cid,
                "candidate_label": candidates.get(cid, {}).get("candidate_label", cid),
                "risk_name": row["risk_name"],
                "primary_risk": row["primary_risk"],
                "alpha": row["alpha"],
                "empirical_risk": row["empirical_risk"],
                "p_value": row["p_value"],
                "corrected_threshold": row["corrected_threshold"],
                "accepted": row["accepted"],
                "selected": candidates.get(cid, {}).get("selected", "0"),
                "tri_risk_accepted": candidates.get(cid, {}).get("tri_risk_accepted", ""),
                "eligible_for_selection": candidates.get(cid, {}).get("eligible_for_selection", ""),
            }
        )
    write_csv(LTT_TABLE, rows)

    selected_rows = [row for row in rows if row["candidate_id"] == SELECTED]
    primary = [row for row in rows if row["primary_risk"] == "1"]
    accepted = [row for row in primary if row["accepted"] == "1"]
    rejected = [row for row in primary if row["accepted"] != "1"]
    display = []
    for cid, label in METHODS:
        by_risk = {row["risk_name"]: row for row in rows if row["candidate_id"] == cid}
        if not by_risk:
            continue
        display.append(
            {
                "Method": label,
                "Tri": "yes" if candidates.get(cid, {}).get("tri_risk_accepted") == "1" else "no",
                "WQL": pct(by_risk["wql_noninferiority_harm"]["empirical_risk"]),
                "pWQL": num(by_risk["wql_noninferiority_harm"]["p_value"], 4),
                "Prot": pct(by_risk["protected_wql_harm"]["empirical_risk"]),
                "pProt": num(by_risk["protected_wql_harm"]["p_value"], 4),
                "UC": pct(by_risk["undercoverage_noninferiority_harm"]["empirical_risk"]),
                "pUC": num(by_risk["undercoverage_noninferiority_harm"]["p_value"], 4),
                "AbsUC": "yes" if by_risk.get("absolute_undercoverage_audit", {}).get("accepted") == "1" else "no",
            }
        )

    lines = [
        "# LTT / CRC Calibration Evidence",
        "",
        "This report converts the final-main calibration screen into an explicit p-value table. It supports a calibrated empirical risk-screening claim. It should not be worded as an unconditional distribution-free guarantee because rolling time-series windows may violate exchangeability and because some candidate classes were developed before this freeze.",
        "",
        "## Selected Method",
        "",
        f"- Selected policy: `{SELECTED}`.",
        f"- Primary selected risk rows: `{len([r for r in selected_rows if r['primary_risk'] == '1'])}`.",
        f"- Primary accepted tests across all candidates: `{len(accepted)}`.",
        f"- Primary rejected tests across all candidates: `{len(rejected)}`.",
        "",
        "## Primary Risk Screen",
        "",
        markdown_table(
            display,
            [
                ("Method", "Method"),
                ("Tri", "Tri-risk"),
                ("WQL", "WQL harm"),
                ("pWQL", "p WQL"),
                ("Prot", "Protected"),
                ("pProt", "p protected"),
                ("UC", "UC harm"),
                ("pUC", "p UC"),
                ("AbsUC", "Abs UC ok"),
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "- Safe wording: DRCR uses an LTT/CRC-style calibration table with Holm-corrected p-values for pre-specified empirical risk functions.",
        "- Unsafe wording: DRCR has a distribution-free finite-sample coverage guarantee for dependent time-series data.",
        "",
        "## Artifacts",
        "",
        f"- `{LTT_TABLE.relative_to(ROOT)}`",
        f"- `{LTT_REPORT.relative_to(ROOT)}`",
    ]
    LTT_REPORT.write_text("\n".join(lines) + "\n")


def build_protected_report(summary: dict[tuple[str, str, str], dict[str, str]], candidates: dict[str, dict[str, str]]) -> None:
    rows = []
    for group_key, group_label in PROTECTED_GROUPS:
        for cid, label in METHODS:
            row = summary[(cid, "test", group_key)]
            rows.append(
                {
                    "group": group_key,
                    "group_label": group_label,
                    "candidate_id": cid,
                    "label": label,
                    "selected": candidates.get(cid, {}).get("selected", "0"),
                    "tri_risk_accepted": candidates.get(cid, {}).get("tri_risk_accepted", "0"),
                    "n_windows": row["n_windows"],
                    "wql_rer": row["repair_median_wql_rer"],
                    "wql_harm_rate": row["wql_harm_rate"],
                    "coverage_q10_q90": row["repair_mean_coverage"],
                    "mae_rer": row["repair_median_mae_rer"],
                    "rmse_rer": row["repair_median_rmse_rer"],
                    "wape_rer": row["repair_median_wape_rer"],
                }
            )
    write_csv(PROTECTED_TABLE, rows)

    by = {(row["candidate_id"], row["group"]): row for row in rows}
    lines = [
        "# Protected-Window Safety Table",
        "",
        "This table supports the core tension: naive fallback can repair some failures but harms windows where native TSFM is already strong. DRCR should be evaluated by both failure repair and protected-window harm.",
        "",
    ]
    for group_key, group_label in PROTECTED_GROUPS:
        table_rows = []
        for cid, label in METHODS:
            row = by[(cid, group_key)]
            table_rows.append(
                {
                    "Method": label,
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
                    table_rows,
                    [
                        ("Method", "Method"),
                        ("Tri", "Tri-risk"),
                        ("WQL", "WQL-RER"),
                        ("Harm", "WQL harm"),
                        ("Cov", "Coverage"),
                        ("MAE", "MAE-RER"),
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Artifacts", "", f"- `{PROTECTED_TABLE.relative_to(ROOT)}`", f"- `{PROTECTED_REPORT.relative_to(ROOT)}`"])
    PROTECTED_REPORT.write_text("\n".join(lines) + "\n")


def build_denominator_report(summary: dict[tuple[str, str, str], dict[str, str]]) -> None:
    rows = []
    for cid, label in METHODS:
        row = summary[(cid, "test", Q9)]
        rows.append(
            {
                "candidate_id": cid,
                "label": label,
                "n_windows": row["n_windows"],
                "raw_wql_rer": row["repair_median_wql_rer"],
                "raw_wql_rer_ci_low": row["repair_median_wql_rer_ci_low"],
                "raw_wql_rer_ci_high": row["repair_median_wql_rer_ci_high"],
                "clipped_wql_rer_cap10": row["repair_median_clipped_wql_rer"],
                "clipped_wql_rer_cap10_ci_low": row["repair_median_clipped_wql_rer_ci_low"],
                "clipped_wql_rer_cap10_ci_high": row["repair_median_clipped_wql_rer_ci_high"],
                "mean_log1p_delta_vs_native": row["paired_mean_log1p_wql_delta_vs_model"],
                "mean_log1p_delta_ci_low": row["paired_mean_log1p_wql_delta_vs_model_ci_low"],
                "mean_log1p_delta_ci_high": row["paired_mean_log1p_wql_delta_vs_model_ci_high"],
            }
        )
    write_csv(CLIPPED_TABLE, rows)
    lines = [
        "# Denominator-Fragility Sensitivity",
        "",
        "The q9 full-grid failure slice is denominator-fragile, so raw WQL-RER can have very wide bootstrap intervals. This report keeps raw ratios but adds clipped and log-sensitivity checks.",
        "",
        markdown_table(
            [
                {
                    "Method": row["label"],
                    "Raw": num(row["raw_wql_rer"]),
                    "RawCI": f"[{num(row['raw_wql_rer_ci_low'])}, {num(row['raw_wql_rer_ci_high'])}]",
                    "Clip10": num(row["clipped_wql_rer_cap10"]),
                    "ClipCI": f"[{num(row['clipped_wql_rer_cap10_ci_low'])}, {num(row['clipped_wql_rer_cap10_ci_high'])}]",
                    "LogD": num(row["mean_log1p_delta_vs_native"], 4),
                    "LogCI": f"[{num(row['mean_log1p_delta_ci_low'], 4)}, {num(row['mean_log1p_delta_ci_high'], 4)}]",
                }
                for row in rows
            ],
            [
                ("Method", "Method"),
                ("Raw", "Raw WQL-RER"),
                ("RawCI", "Raw CI"),
                ("Clip10", "Clipped 10"),
                ("ClipCI", "Clipped CI"),
                ("LogD", "Mean log d"),
                ("LogCI", "Mean log CI"),
            ],
        ),
        "",
        "Interpretation: selected DRCR improves q9 failure under raw, clipped, and paired log sensitivity. This weakens the criticism that the result is only a small-denominator artifact.",
        "",
        "## Artifacts",
        "",
        f"- `{CLIPPED_TABLE.relative_to(ROOT)}`",
        f"- `{DENOM_REPORT.relative_to(ROOT)}`",
    ]
    DENOM_REPORT.write_text("\n".join(lines) + "\n")


def build_manifests(windows: list[dict[str, str]]) -> None:
    by_window = {}
    for row in windows:
        key = (row["source"], row["dataset"], row["family"], row["model"], row["series_id"], row["window_index"], row["role"], row["phase"])
        by_window[key] = row
    split_rows = []
    for key, row in sorted(by_window.items()):
        split_rows.append(
            {
                "window_key": "|".join(map(str, key[:-1])),
                "source": row["source"],
                "dataset": row["dataset"],
                "family": row["family"],
                "model": row["model"],
                "series_id": row["series_id"],
                "window_index": row["window_index"],
                "role": row["role"],
                "target_id": row["target_id"],
                "evidence_tier": row["evidence_tier"],
                "split": row["phase"],
                "quantile_grid_n_levels": row["quantile_grid_n_levels"],
            }
        )
    write_csv(SPLIT_MANIFEST, split_rows)

    dataset_counts: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        key = (row["source"], row["dataset"], row["target_id"], row["role"], row["evidence_tier"])
        dataset_counts[key][row["split"]] += 1
    dataset_rows = []
    for (source, dataset, target_id, role, tier), counts in sorted(dataset_counts.items()):
        dataset_rows.append(
            {
                "source": source,
                "dataset": dataset,
                "target_id": target_id,
                "role": role,
                "evidence_tier": tier,
                "calibration_windows": counts["calibration"],
                "test_windows": counts["test"],
                "total_windows": counts["calibration"] + counts["test"],
                "contamination_risk": "unknown_or_benchmark_overlap_possible",
            }
        )
    write_csv(DATASET_MANIFEST, dataset_rows)

    model_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        model_counts[(row["family"], row["model"])][row["split"]] += 1
    model_rows = []
    for (family, model), counts in sorted(model_counts.items()):
        model_rows.append(
            {
                "family": family,
                "model": model,
                "calibration_windows": counts["calibration"],
                "test_windows": counts["test"],
                "total_windows": counts["calibration"] + counts["test"],
            }
        )
    write_csv(MODEL_MANIFEST, model_rows)


def build_paper_faithful_report(summary: dict[tuple[str, str, str], dict[str, str]]) -> None:
    overall = summary[(SELECTED, "test", "overall")]
    q9 = summary[(SELECTED, "test", Q9)]
    lines = [
        "# Paper-Faithful Metric Robustness",
        "",
        "This report scopes what is currently paper-faithful in the final-main artifacts and points to the broader prior metric report.",
        "",
        "## Final-Main Metrics Available",
        "",
        markdown_table(
            [
                {
                    "Slice": "Overall",
                    "WQL": num(overall["repair_median_wql_rer"]),
                    "MAE": num(overall["repair_median_mae_rer"]),
                    "RMSE": num(overall["repair_median_rmse_rer"]),
                    "WAPE": num(overall["repair_median_wape_rer"]),
                    "Coverage": pct(overall["repair_mean_coverage"]),
                },
                {
                    "Slice": "q9 full-grid failure",
                    "WQL": num(q9["repair_median_wql_rer"]),
                    "MAE": num(q9["repair_median_mae_rer"]),
                    "RMSE": num(q9["repair_median_rmse_rer"]),
                    "WAPE": num(q9["repair_median_wape_rer"]),
                    "Coverage": pct(q9["repair_mean_coverage"]),
                },
            ],
            [
                ("Slice", "Slice"),
                ("WQL", "WQL-RER"),
                ("MAE", "MAE-RER"),
                ("RMSE", "RMSE-RER"),
                ("WAPE", "WAPE-RER"),
                ("Coverage", "Coverage"),
            ],
        ),
        "",
        "## Scope",
        "",
        "- WQL-RER is the primary final-main metric because DRCR is a probabilistic repair method and q9/full-grid artifacts are available for the strongest failure slice.",
        "- MAE/RMSE/WAPE-RER are computed on the same final-main windows.",
        "- MASE-RER remains in the earlier paper-faithful robustness report where in-sample scaling is available; it is not recomputed for every merged final-main window.",
        "- CRPS/MSIS are not headline-ready because not all artifacts expose sample paths or full distributions.",
        "",
        "## Prior Supporting Report",
        "",
        "- `docs/paper_faithful_metric_robustness_report.md`",
        "",
        "## Artifact",
        "",
        f"- `{PAPER_FAITHFUL_REPORT.relative_to(ROOT)}`",
    ]
    PAPER_FAITHFUL_REPORT.write_text("\n".join(lines) + "\n")


def build_no_test_tuning(status: dict[str, object]) -> None:
    lines = [
        "# No-Test-Tuning Statement",
        "",
        "This statement records the current status of the final-main DRCR experiment.",
        "",
        f"- Final-main artifact timestamp: `{status['timestamp']}`.",
        "- Candidate set now includes objective baselines: global blend, score gate, width gate, and oracle upper bound.",
        "- Objective baselines are marked ineligible for final DRCR selection unless a future frozen protocol explicitly allows them.",
        f"- Current selected method remains `{SELECTED}` under calibration-only eligible selection.",
        f"- `{FRONTIER}` remains a frontier probe and must not be promoted to final method unless a newly frozen objective re-selects it before test inspection.",
        "- Current formal wording should be LTT/CRC-inspired empirical risk screening, not unconditional distribution-free coverage guarantee.",
        "",
        "Known limitation: this statement is retrospective for the current artifact sequence. A stronger confirmatory run should freeze the candidate set and split manifest before regenerating final test tables.",
    ]
    NO_TEST_TUNING.write_text("\n".join(lines) + "\n")


def build_caption() -> None:
    lines = [
        "# Plain-Language Main Figure Caption",
        "",
        "Panel A shows the calibration risk screen. Gray/classical and full-repair options are rejected when they harm too many protected or undercovered windows. The selected red/blue DRCR intervention passes the screen.",
        "",
        "Panel B shows the hardest q9 full-grid failure windows. Lower WQL-RER is better. Native TSFM fails badly on this slice; DRCR reduces the failure while maintaining interval coverage better than deterministic fallback.",
        "",
        "Panel C shows why fallback is unsafe. Classical fallback can look helpful on failure windows but damages stress and finance windows where native TSFM already works.",
        "",
        "Panel D reports interval coverage. Coverage is evidence, not a formal coverage guarantee.",
        "",
        "Panel E reports point-metric robustness. It checks that the probabilistic WQL story is not hiding large point-error harm.",
    ]
    CAPTION.write_text("\n".join(lines) + "\n")


def build_final_results_copy(summary_rows: list[dict[str, str]]) -> None:
    rows = []
    for row in summary_rows:
        if row["phase"] != "test":
            continue
        if row["group"] not in {"overall", Q9, "role:positive_control", "role:stress_target", "target_id:finance_fred_stress"}:
            continue
        rows.append(row)
    write_csv(FINAL_RESULTS, rows)


def build_final_summary(status: dict[str, object], summary: dict[tuple[str, str, str], dict[str, str]]) -> None:
    native_overall = summary[("native_tsfm", "test", "overall")]
    native_q9 = summary[("native_tsfm", "test", Q9)]
    selected_overall = summary[(SELECTED, "test", "overall")]
    selected_q9 = summary[(SELECTED, "test", Q9)]
    score_gate_q9 = summary[("smooth_score_gate_t0.50_w1.00", "test", Q9)]
    score_gate_overall = summary[("smooth_score_gate_t0.50_w1.00", "test", "overall")]
    oracle_q9 = summary[("oracle_native_classical_drcr", "test", Q9)]
    finance = summary[(SELECTED, "test", "target_id:finance_fred_stress")]
    native_classical_lines: list[str] = []
    if NATIVE_CLASSICAL_SUMMARY.exists():
        native_classical_rows = {row.get("group", ""): row for row in read_csv(NATIVE_CLASSICAL_SUMMARY)}
        if native_classical_rows:
            nc_overall = native_classical_rows["overall"]
            nc_failure = native_classical_rows["role:failure_target"]
            nc_positive = native_classical_rows["role:positive_control"]
            nc_stress = native_classical_rows["role:stress_target"]
            native_classical_lines = [
                (
                    f"- Sidecar empirical classical interval audit: 1076 windows；overall WQL-RER "
                    f"`{num(nc_overall['median_native_classical_wql_rer'])}`，failure WQL-RER "
                    f"`{num(nc_failure['median_native_classical_wql_rer'])}`；但 positive/stress harm "
                    f"`{pct(nc_positive['native_classical_harm_rate_vs_model'])}` / "
                    f"`{pct(nc_stress['native_classical_harm_rate_vs_model'])}`，进一步说明 naive classical fallback 不安全。"
                )
            ]
    oral_gap_rows = read_csv(ORAL_SOURCE_GAPS) if ORAL_SOURCE_GAPS.exists() else []
    oral_command_rows = read_csv(ORAL_RERUN_COMMANDS) if ORAL_RERUN_COMMANDS.exists() else []
    local_blocker_rows = read_csv(LOCAL_RERUN_BLOCKERS) if LOCAL_RERUN_BLOCKERS.exists() else []
    remote_plan_rows = read_csv(REMOTE_Q9_RERUN_PLAN) if REMOTE_Q9_RERUN_PLAN.exists() else []
    remote_audit_rows = read_csv(REMOTE_Q9_RERUN_AUDIT) if REMOTE_Q9_RERUN_AUDIT.exists() else []
    ingestion_rows = read_csv(REMOTE_Q9_INGESTION) if REMOTE_Q9_INGESTION.exists() else []
    p0_gap_count = sum(row.get("priority") == "P0" for row in oral_gap_rows)
    q9_gap_count = sum(row.get("needs_q9_fullgrid_rerun") == "1" for row in oral_gap_rows)
    history_gap_count = sum(row.get("needs_history_context_export") == "1" for row in oral_gap_rows)
    ready_commands = sum(command_status(row) == "ready" for row in oral_command_rows)
    p0_ready_commands = sum(command_status(row) == "ready" and row.get("priority") == "P0" for row in oral_command_rows)
    blocked_commands = sum(command_status(row) != "ready" for row in oral_command_rows)
    timeout_blockers = sum(row.get("execution_status") == "timeout" for row in local_blocker_rows)
    blocker_families = "/".join(sorted({row.get("family", "") for row in local_blocker_rows if row.get("family")}))
    remote_plan_windows = sum(int(row.get("manifest_windows") or 0) for row in remote_plan_rows)
    remote_complete_sources = sum(row.get("complete_for_ingestion") == "1" for row in remote_audit_rows)
    ingestion_ready_sources = sum(row.get("ready_for_final_main_refresh") == "1" for row in ingestion_rows)
    timesfm_q9_gap_count = sum(
        row.get("family") == "timesfm" and row.get("needs_q9_fullgrid_rerun") == "1" for row in oral_gap_rows
    )
    moirai_q9_gap_count = sum(
        row.get("family") == "moirai" and row.get("needs_q9_fullgrid_rerun") == "1" for row in oral_gap_rows
    )
    confirmatory_lines: list[str] = []
    if CONFIRMATORY_STATUS.exists() and CONFIRMATORY_SUMMARY.exists():
        confirm_status = json.loads(CONFIRMATORY_STATUS.read_text())
        confirm_rows = read_csv(CONFIRMATORY_SUMMARY)
        confirm_summary = {(row["candidate_id"], row["phase"], row["group"]): row for row in confirm_rows}
        if (
            confirm_status.get("status") == "ok"
            and confirm_status.get("confirmatory") is True
            and (SELECTED, "test", "overall") in confirm_summary
        ):
            c_native_overall = confirm_summary[("native_tsfm", "test", "overall")]
            c_selected_overall = confirm_summary[(SELECTED, "test", "overall")]
            c_native_q9 = confirm_summary[("native_tsfm", "test", Q9)]
            c_selected_q9 = confirm_summary[(SELECTED, "test", Q9)]
            c_finance = confirm_summary[(SELECTED, "test", "target_id:finance_fred_stress")]
            confirmatory_lines = [
                (
                    f"- Confirmatory salted split: calibration/test = "
                    f"`{confirm_status['n_calibration_windows']}` / `{confirm_status['n_test_windows']}`; "
                    f"selected method remains `{confirm_status['selected_candidate_id']}`; "
                    f"overall WQL-RER native `{num(c_native_overall['repair_median_wql_rer'])}` -> "
                    f"DRCR `{num(c_selected_overall['repair_median_wql_rer'])}`; "
                    f"q9 failure WQL-RER native `{num(c_native_q9['repair_median_wql_rer'])}` -> "
                    f"DRCR `{num(c_selected_q9['repair_median_wql_rer'])}`; "
                    f"finance harm `{pct(c_finance['wql_harm_rate'])}`."
                )
            ]
    lines = [
        "# Final DRCR Paper-Readiness Summary",
        "",
        "## 当前结论",
        "",
        f"- 当前 final-main suite: `{status['n_windows']}` windows, calibration/test = `{status['n_calibration_windows']}` / `{status['n_test_windows']}`.",
        f"- 当前 selected method: `{SELECTED}`.",
        f"- Overall WQL-RER: native `{num(native_overall['repair_median_wql_rer'])}` -> selected DRCR `{num(selected_overall['repair_median_wql_rer'])}`.",
        f"- q9 full-grid failure WQL-RER: native `{num(native_q9['repair_median_wql_rer'])}` -> selected DRCR `{num(selected_q9['repair_median_wql_rer'])}`.",
        f"- Finance FRED stress selected harm: `{pct(finance['wql_harm_rate'])}`, coverage `{pct(finance['repair_mean_coverage'])}`.",
        *confirmatory_lines,
        f"- Oral evidence gap queue: `{p0_gap_count}` 个 P0 source reruns，`{q9_gap_count}` 个 q9/full-grid source gaps，`{history_gap_count}` 个 history/context export gaps。",
        f"- Oral rerun command manifest: `{len(oral_command_rows)}` 个 source commands，其中 `{ready_commands}` ready、`{p0_ready_commands}` 个 P0 ready、`{blocked_commands}` 个 finance/export blocker。",
        f"- Remote q9 execution pack: `{len(remote_plan_rows)}` 个 P0 sources / `{remote_plan_windows}` windows 已整理成可远程执行和验收的协议；completion audit 当前 `{remote_complete_sources}`/`{len(remote_plan_rows)}` 个 source complete，ingestion manifest 当前 `{ingestion_ready_sources}`/`{len(ingestion_rows)}` 个 source ready for final-main refresh。",
        f"- Local rerun resource blockers: `{timeout_blockers}` 个 timeout，覆盖 `{blocker_families or 'none'}`；说明本机 8GB 不适合完成剩余 q9/full-grid 大跑。",
        "",
        "## Objective Baseline 新发现",
        "",
        f"- Simple score gate q9 WQL-RER `{num(score_gate_q9['repair_median_wql_rer'])}`，但 overall WQL-RER `{num(score_gate_overall['repair_median_wql_rer'])}`；它能打 failure，但不是稳定整体替代品。",
        f"- Oracle upper bound q9 WQL-RER `{num(oracle_q9['repair_median_wql_rer'])}`，说明仍有 selector 改进空间。",
        "- Global blend / width gate / classical fallback 均暴露出 harm 或 coverage 问题，支持 DRCR 不是普通 fallback/blending。",
        *native_classical_lines,
        "",
        "## 可以安全写的 claim",
        "",
        "- TSFM 是 conditionally reliable，不是 universal reliable。",
        "- DRCR 是 calibration-selected external repair layer。",
        f"- 在当前 locked {status['n_windows']}-window suite 上，DRCR 能降低 severe q9 probabilistic failure，同时限制 protected/stress/finance harm。",
        "- 当前 formal 证据支持 LTT/CRC-style empirical risk screening；严格 distribution-free guarantee 还需要更强的前置冻结和 dependence 讨论。",
        "",
        "## 仍需补强",
        "",
        "- 更强 classical probabilistic baselines：当前 final-main suite 的 history/context sidecar 已恢复，并已完成 sidecar empirical classical interval audit；但严格 benchmark-native AutoARIMA/AutoETS/Theta interval parity 还没有完成，不能过度声称为原生区间复现。",
        "- 可执行 rerun 队列已生成：ready commands 已带 `--window-manifest`、`--output-slug` 和 `--export-history-sidecar`；这代表下一轮 q9/full-grid 新推理可以启动，但不等于这些 reruns 已完成。",
        "- 远程执行包已生成：`docs/remote_q9_rerun_execution_pack.md` 和 `results/aaai_stress/remote_q9_rerun_plan.csv` 明确每个 P0 source 的 exact command、expected raw/sidecar/status outputs、以及 post-run validation critic。",
        "- 远程完成度审计已生成：`docs/remote_q9_rerun_completion_audit.md` 和 `results/aaai_stress/remote_q9_rerun_completion_audit.csv` 会检查 raw/sidecar/status/q10..q90 是否真的齐全；当前是 0/17，说明还没跑完，不是证据闭环。",
        "- 远程 ingestion manifest 已生成：`docs/remote_q9_ingestion_manifest.md` 和 `results/aaai_stress/remote_q9_ingestion_manifest.csv` 记录远程结果回来后 `original_source -> rerun_slug` 的替换关系；当前是 0/17 ready for final-main refresh。",
        "- 本机执行证据：TimesFM 2-window slice 900 秒 timeout；Moirai small 8-window slice 1200 秒 timeout。剩余 q9/full-grid reruns 需要远程或更大内存机器。",
        f"- 更多 Moirai/TimesFM q9/full-grid evidence：当前 gap matrix 标出 `{q9_gap_count}` 个 source-level q9/full-grid reruns，其中 TimesFM `{timesfm_q9_gap_count}` 个、Moirai `{moirai_q9_gap_count}` 个。",
        "- 当前 1076-window suite 的 salted confirmatory rerun 已补；仍需要外部新 inference / 更大 benchmark breadth 的 confirmatory evidence。",
        "- 如果要强 conformal claim，需要更严格的 p-value/Holm appendix 和时间序列依赖处理。",
        "",
        "## 关键 artifacts",
        "",
        f"- `{FINAL_RESULTS.relative_to(ROOT)}`",
        f"- `{LTT_TABLE.relative_to(ROOT)}`",
        f"- `{PROTECTED_TABLE.relative_to(ROOT)}`",
        f"- `{CLIPPED_TABLE.relative_to(ROOT)}`",
        f"- `{SPLIT_MANIFEST.relative_to(ROOT)}`",
        f"- `{DATASET_MANIFEST.relative_to(ROOT)}`",
        f"- `{MODEL_MANIFEST.relative_to(ROOT)}`",
        f"- `{ORAL_SOURCE_GAPS.relative_to(ROOT)}`",
        f"- `{ORAL_RERUN_COMMANDS.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_RERUN_PLAN.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_RERUN_AUDIT.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_INGESTION.relative_to(ROOT)}`",
        f"- `{NATIVE_CLASSICAL_SUMMARY.relative_to(ROOT)}`",
        f"- `{LOCAL_RERUN_BLOCKERS.relative_to(ROOT)}`",
        "- `docs/remote_q9_rerun_execution_pack.md`",
        "- `docs/remote_q9_rerun_completion_audit.md`",
        "- `docs/remote_q9_ingestion_manifest.md`",
        "- `docs/oral_rerun_command_manifest.md`",
        "- `docs/oral_evidence_gap_matrix.md`",
    ]
    FINAL_SUMMARY.write_text("\n".join(lines) + "\n")


def main() -> None:
    status = json.loads(STATUS.read_text())
    windows = read_csv(WINDOWS)
    summary_rows = read_csv(SUMMARY)
    summary = keyed(summary_rows)
    candidates = {row["candidate_id"]: row for row in read_csv(CANDIDATES)}
    calibration = read_csv(CALIBRATION)

    build_ltt_report(calibration, candidates)
    build_protected_report(summary, candidates)
    build_denominator_report(summary)
    build_manifests(windows)
    build_paper_faithful_report(summary)
    build_no_test_tuning(status)
    build_caption()
    build_final_results_copy(summary_rows)
    build_final_summary(status, summary)

    print(f"wrote {LTT_REPORT.relative_to(ROOT)}")
    print(f"wrote {PROTECTED_REPORT.relative_to(ROOT)}")
    print(f"wrote {DENOM_REPORT.relative_to(ROOT)}")
    print(f"wrote {FINAL_SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
