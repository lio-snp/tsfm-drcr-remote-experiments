#!/usr/bin/env python3
"""Build an AAAI/oral-readiness status dashboard from current DRCR artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

WINDOWS = OUT / "final_main_figure_windows.csv"
SUMMARY = OUT / "final_main_figure_summary.csv"
CANDIDATES = OUT / "final_main_figure_candidates.csv"
LTT = OUT / "ltt_crc_pvalue_table.csv"
SPLIT = OUT / "split_manifest.csv"
DATASET_MANIFEST = OUT / "dataset_manifest.csv"
MODEL_MANIFEST = OUT / "model_manifest.csv"
BASELINES = OUT / "main_baseline_comparison.csv"
CONFIRMATORY_STATUS = OUT / "final_main_confirmatory_status.json"
CONFIRMATORY_SUMMARY = OUT / "final_main_confirmatory_summary.csv"

REQ_OUT = OUT / "aaai_oral_requirement_matrix.csv"
COVERAGE_OUT = OUT / "benchmark_coverage_table.csv"
MODEL_DATASET_OUT = OUT / "model_dataset_matrix.csv"
Q9_OUT = OUT / "q9_full_grid_results.csv"
CONTAM_OUT = OUT / "contamination_risk_table.csv"
CLASSICAL_FEASIBILITY = OUT / "classical_probabilistic_baseline_feasibility.csv"
NATIVE_CLASSICAL_SUMMARY = OUT / "native_classical_interval_audit_summary.csv"
LOCAL_RERUN_BLOCKERS = OUT / "local_rerun_resource_blockers.csv"
REMOTE_Q9_RERUN_PLAN = OUT / "remote_q9_rerun_plan.csv"
REMOTE_Q9_RERUN_AUDIT = OUT / "remote_q9_rerun_completion_audit.csv"
REMOTE_Q9_INGESTION = OUT / "remote_q9_ingestion_manifest.csv"
ORAL_SOURCE_GAPS = OUT / "oral_evidence_source_gap_matrix.csv"
ORAL_FAMILY_GAPS = OUT / "oral_evidence_family_gap_matrix.csv"
ORAL_RERUN_COMMANDS = OUT / "oral_rerun_command_manifest.csv"
DOC_OUT = DOCS / "aaai_oral_goal_status.md"

SELECTED = "drcr_expert_pull_1.25_cap_1.10"
FRONTIER = "drcr_width_veto_expert_pull_1.50_cap_1.10"
Q9_GROUP = "evidence_tier:q9_fullgrid|role:failure_target"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{parsed:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{100.0 * parsed:.{digits}f}%"


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


def status_json() -> dict[str, object]:
    path = OUT / "final_main_figure_status.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def confirmatory_status_json() -> dict[str, object]:
    if not CONFIRMATORY_STATUS.exists():
        return {}
    return json.loads(CONFIRMATORY_STATUS.read_text())


def build_coverage(split_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        key = (
            row.get("family", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("role", ""),
            row.get("evidence_tier", ""),
        )
        grouped[key][row.get("split", "")] += 1
    rows = []
    for (family, model, dataset, role, tier), counts in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "model": model,
                "dataset": dataset,
                "role": role,
                "evidence_tier": tier,
                "calibration_windows": counts.get("calibration", 0),
                "test_windows": counts.get("test", 0),
                "total_windows": sum(counts.values()),
            }
        )
    write_csv(COVERAGE_OUT, rows)
    return rows


def build_model_dataset_matrix(split_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        grouped[(row.get("model", ""), row.get("dataset", ""))][row.get("split", "")] += 1
    rows = []
    for (model, dataset), counts in sorted(grouped.items()):
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "calibration_windows": counts.get("calibration", 0),
                "test_windows": counts.get("test", 0),
                "total_windows": sum(counts.values()),
            }
        )
    write_csv(MODEL_DATASET_OUT, rows)
    return rows


def build_q9_results(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in summary_rows:
        if row.get("phase") != "test" or row.get("group") != Q9_GROUP:
            continue
        rows.append(
            {
                "candidate_id": row["candidate_id"],
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
    write_csv(Q9_OUT, rows)
    return rows


def build_contamination(dataset_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in dataset_rows:
        rows.append(
            {
                "source": row.get("source", ""),
                "dataset": row.get("dataset", ""),
                "target_id": row.get("target_id", ""),
                "role": row.get("role", ""),
                "evidence_tier": row.get("evidence_tier", ""),
                "total_windows": row.get("total_windows", ""),
                "contamination_risk": row.get("contamination_risk", "unknown_or_not_audited"),
                "paper_safe_use": "benchmark/stress evidence; avoid contamination-free claim",
            }
        )
    write_csv(CONTAM_OUT, rows)
    return rows


def requirement_rows(
    *,
    split_rows: list[dict[str, str]],
    dataset_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
    q9_rows: list[dict[str, object]],
    ltt_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    n_windows = len(split_rows)
    n_models = len({row.get("model", "") for row in split_rows})
    n_families = len({row.get("family", "") for row in split_rows})
    n_datasets = len({row.get("dataset", "") for row in split_rows})
    n_q9 = sum(1 for row in split_rows if row.get("evidence_tier") == "q9_fullgrid")
    families = ", ".join(sorted({row.get("family", "") for row in split_rows}))
    datasets = ", ".join(sorted({row.get("dataset", "") for row in split_rows}))
    baseline_methods = len({row.get("candidate_id", "") for row in baseline_rows})
    selected_ltt = [row for row in ltt_rows if row.get("candidate_id") == SELECTED and row.get("primary_risk") == "1"]
    selected_ltt_ok = selected_ltt and all(row.get("accepted") == "1" for row in selected_ltt)
    oral_source_gaps = read_csv(ORAL_SOURCE_GAPS)
    q9_gap_count = sum(row.get("needs_q9_fullgrid_rerun") == "1" for row in oral_source_gaps)
    history_gap_count = sum(row.get("needs_history_context_export") == "1" for row in oral_source_gaps)
    oral_rerun_commands = read_csv(ORAL_RERUN_COMMANDS)
    ready_commands = sum(command_status(row) == "ready" for row in oral_rerun_commands)
    p0_ready_commands = sum(command_status(row) == "ready" and row.get("priority") == "P0" for row in oral_rerun_commands)
    blocked_commands = sum(command_status(row) != "ready" for row in oral_rerun_commands)
    local_blockers = read_csv(LOCAL_RERUN_BLOCKERS)
    timeout_blockers = sum(row.get("execution_status") == "timeout" for row in local_blockers)
    blocker_families = ", ".join(sorted({row.get("family", "") for row in local_blockers if row.get("family")}))
    remote_plan_rows = read_csv(REMOTE_Q9_RERUN_PLAN)
    remote_plan_sources = len(remote_plan_rows)
    remote_plan_windows = sum(int(row.get("manifest_windows") or 0) for row in remote_plan_rows)
    remote_audit_rows = read_csv(REMOTE_Q9_RERUN_AUDIT)
    remote_complete_sources = sum(row.get("complete_for_ingestion") == "1" for row in remote_audit_rows)
    ingestion_rows = read_csv(REMOTE_Q9_INGESTION)
    ingestion_ready_sources = sum(row.get("ready_for_final_main_refresh") == "1" for row in ingestion_rows)
    native_classical_rows = read_csv(NATIVE_CLASSICAL_SUMMARY)
    native_classical_summary = {row.get("group", ""): row for row in native_classical_rows}
    native_classical_evidence = "sidecar-backed empirical classical interval audit not yet generated"
    if native_classical_summary:
        overall_nc = native_classical_summary["overall"]
        positive_nc = native_classical_summary["role:positive_control"]
        stress_nc = native_classical_summary["role:stress_target"]
        native_classical_evidence = (
            f"sidecar-backed empirical classical interval audit covers {overall_nc['n_windows']} windows; "
            f"overall WQL-RER {float(overall_nc['median_native_classical_wql_rer']):.3f}; "
            f"positive/stress harm {100 * float(positive_nc['native_classical_harm_rate_vs_model']):.1f}%/"
            f"{100 * float(stress_nc['native_classical_harm_rate_vs_model']):.1f}%"
        )
    confirmatory_status = confirmatory_status_json()
    confirmatory_done = (
        confirmatory_status.get("status") == "ok"
        and confirmatory_status.get("confirmatory") is True
        and confirmatory_status.get("selected_candidate_id") == SELECTED
    )

    rows = [
        {
            "area": "Protocol freeze",
            "status": "done",
            "evidence": "final_frozen_protocol.md, candidate_policy_set.yaml, risk_thresholds.yaml, split/model/dataset manifests",
            "gap": "For a stronger formal guarantee, repeat once more with protocol frozen before any candidate development.",
        },
        {
            "area": "Objective baselines",
            "status": "done_current_suite",
            "evidence": f"{baseline_methods} methods in main_baseline_comparison.csv including native, classical, calibrated classical, blend, gates, oracle, DRCR variants; {native_classical_evidence}; feasibility/gap audits show {history_gap_count} source-level history/context export gaps after sidecar reconstruction",
            "gap": "Still missing strict benchmark-native AutoARIMA/AutoETS/Theta interval parity; sidecar-backed empirical classical intervals are now audited but should not be overclaimed as exact native interval reproduction.",
        },
        {
            "area": "Unified metrics",
            "status": "done_current_suite",
            "evidence": "WQL-RER, MAE-RER, RMSE-RER, WAPE-RER, coverage, harm, clipped sensitivity computed on final-main split",
            "gap": "CRPS/MSIS and full MASE coverage are not headline-ready for every merged window.",
        },
        {
            "area": "Paper-faithful metrics",
            "status": "partial",
            "evidence": "Metric mapping and prior paper-faithful report exist; final-main emphasizes Chronos-style WQL/full-grid and point robustness",
            "gap": "Need broader Moirai/GIFT MASE+WQL and TimesFM official metric tables for all expanded datasets.",
        },
        {
            "area": "q9/full-grid evidence",
            "status": "done_current_suite_partial_broad",
            "evidence": f"{n_q9} unique q9/full-grid windows in split manifest; q9_full_grid_results.csv generated",
            "gap": f"More Moirai/TimesFM full-grid or native quantile outputs would make this a stronger main-track claim; current oral gap matrix flags {q9_gap_count or 'remaining'} source-level q9/full-grid reruns, with {p0_ready_commands} P0 rerun commands ready.",
        },
        {
            "area": "Large-scale rerun queue",
            "status": "ready_queue_not_executed",
            "evidence": f"{len(oral_rerun_commands)} source-level rerun commands generated; {ready_commands} ready, {p0_ready_commands} P0 ready, {blocked_commands} blocked; each ready command includes exact window manifest, output slug, and history sidecar export.",
            "gap": f"The command queue is ready, but local execution is resource-limited: {timeout_blockers} timeout blockers recorded across {blocker_families or 'no families yet'}. Execute the 17 P0 q9/full-grid reruns on remote or a larger-memory machine before claiming broader probabilistic evidence is complete.",
        },
        {
            "area": "Remote q9 execution pack",
            "status": "ready_not_executed",
            "evidence": f"remote_q9_rerun_plan.csv and remote_q9_rerun_execution_pack.md define {remote_plan_sources} P0 sources / {remote_plan_windows} windows with expected raw, sidecar, status outputs and validation commands; completion audit currently has {remote_complete_sources}/{remote_plan_sources} sources complete, ingestion manifest has {ingestion_ready_sources}/{len(ingestion_rows)} ready for final-main refresh",
            "gap": "Pack is an execution contract, not completed inference. Run it on remote/larger-memory hardware, rerun the completion audit, backfill features, then refresh final-main source inventory before claiming these gaps are closed.",
        },
        {
            "area": "Model-family breadth",
            "status": "partial",
            "evidence": f"{n_models} model identifiers across {n_families} families: {families}",
            "gap": "Chronos is cleanest; Moirai/TimesFM are present but still less balanced than Chronos.",
        },
        {
            "area": "Dataset breadth",
            "status": "partial",
            "evidence": f"{n_datasets} datasets in current locked suite: {datasets}",
            "gap": "Not yet a full GIFT-Eval/Monash/ETT/M4/M5-scale benchmark expansion.",
        },
        {
            "area": "Protected-window safety",
            "status": "done_current_suite",
            "evidence": "protected_safety_results.csv covers overall, positive controls, stress targets, finance FRED stress",
            "gap": "Needs larger and more balanced protected groups for oral-level breadth.",
        },
        {
            "area": "LTT/CRC-style calibration",
            "status": "done_as_empirical_screen",
            "evidence": f"Selected candidate has {len(selected_ltt)} primary risk rows; accepted={bool(selected_ltt_ok)}",
            "gap": "Do not claim strict distribution-free finite-sample validity under time-series dependence without a stricter pre-freeze rerun.",
        },
        {
            "area": "Denominator fragility",
            "status": "done_current_suite",
            "evidence": "clipped_sensitivity_table.csv and denominator_fragility_report.md generated",
            "gap": "Add raw absolute-delta panels in appendix if space permits.",
        },
        {
            "area": "Main figure",
            "status": "done_draft",
            "evidence": "final_main_figure_draft.{png,pdf,svg} exists with unified color system",
            "gap": "Needs final paper layout polish after deciding whether WidthVeto remains probe.",
        },
        {
            "area": "Confirmatory rerun",
            "status": "done_current_suite" if confirmatory_done else "not_done",
            "evidence": (
                "final_main_confirmatory_* artifacts exist; salted split "
                f"{confirmatory_status.get('n_calibration_windows', 'NA')}/"
                f"{confirmatory_status.get('n_test_windows', 'NA')} selected "
                f"{confirmatory_status.get('selected_candidate_id', 'NA')}"
                if confirmatory_done
                else f"Current suite has {n_windows} locked unique forecast windows, but candidates were developed before final freeze"
            ),
            "gap": (
                "Still needs external new-inference / broader-domain confirmatory evidence for oral-level breadth."
                if confirmatory_done
                else "Run one untouched confirmatory split after all candidates/baselines are frozen."
            ),
        },
    ]
    write_csv(REQ_OUT, rows)
    return rows


def main() -> None:
    split_rows = read_csv(SPLIT)
    summary_rows = read_csv(SUMMARY)
    dataset_rows = read_csv(DATASET_MANIFEST)
    model_rows = read_csv(MODEL_MANIFEST)
    ltt_rows = read_csv(LTT)
    baseline_rows = read_csv(BASELINES)

    coverage_rows = build_coverage(split_rows)
    matrix_rows = build_model_dataset_matrix(split_rows)
    q9_rows = build_q9_results(summary_rows)
    contam_rows = build_contamination(dataset_rows)
    req_rows = requirement_rows(
        split_rows=split_rows,
        dataset_rows=dataset_rows,
        model_rows=model_rows,
        q9_rows=q9_rows,
        ltt_rows=ltt_rows,
        baseline_rows=baseline_rows,
    )

    status = status_json()
    confirm_status = confirmatory_status_json()
    summary_key = {(row["candidate_id"], row["phase"], row["group"]): row for row in summary_rows}
    overall_selected = summary_key.get((SELECTED, "test", "overall"), {})
    q9_selected = summary_key.get((SELECTED, "test", Q9_GROUP), {})
    overall_native = summary_key.get(("native_tsfm", "test", "overall"), {})
    q9_native = summary_key.get(("native_tsfm", "test", Q9_GROUP), {})
    confirm_summary = read_csv(CONFIRMATORY_SUMMARY)
    confirm_key = {(row["candidate_id"], row["phase"], row["group"]): row for row in confirm_summary}
    confirm_overall_selected = confirm_key.get((SELECTED, "test", "overall"), {})
    confirm_q9_selected = confirm_key.get((SELECTED, "test", Q9_GROUP), {})
    confirm_overall_native = confirm_key.get(("native_tsfm", "test", "overall"), {})
    confirm_q9_native = confirm_key.get(("native_tsfm", "test", Q9_GROUP), {})

    status_counts = Counter(row["status"] for row in req_rows)
    family_counts = Counter(row.get("family", "") for row in split_rows)
    role_counts = Counter(row.get("role", "") for row in split_rows)
    tier_counts = Counter(row.get("evidence_tier", "") for row in split_rows)
    oral_source_gaps = read_csv(ORAL_SOURCE_GAPS)
    q9_gap_count = sum(row.get("needs_q9_fullgrid_rerun") == "1" for row in oral_source_gaps)
    history_gap_count = sum(row.get("needs_history_context_export") == "1" for row in oral_source_gaps)
    p0_gap_count = sum(row.get("priority") == "P0" for row in oral_source_gaps)
    oral_rerun_commands = read_csv(ORAL_RERUN_COMMANDS)
    ready_commands = sum(command_status(row) == "ready" for row in oral_rerun_commands)
    p0_ready_commands = sum(command_status(row) == "ready" and row.get("priority") == "P0" for row in oral_rerun_commands)
    blocked_commands = sum(command_status(row) != "ready" for row in oral_rerun_commands)
    remote_plan_rows = read_csv(REMOTE_Q9_RERUN_PLAN)
    remote_plan_windows = sum(int(row.get("manifest_windows") or 0) for row in remote_plan_rows)
    remote_audit_rows = read_csv(REMOTE_Q9_RERUN_AUDIT)
    remote_complete_sources = sum(row.get("complete_for_ingestion") == "1" for row in remote_audit_rows)
    ingestion_rows = read_csv(REMOTE_Q9_INGESTION)
    ingestion_ready_sources = sum(row.get("ready_for_final_main_refresh") == "1" for row in ingestion_rows)

    req_display = req_rows
    coverage_display = sorted(coverage_rows, key=lambda r: (-int(r["total_windows"]), r["family"], r["dataset"]))[:20]
    q9_display = [
        {
            "Method": row["candidate_id"],
            "WQL-RER": fmt_float(row["wql_rer"]),
            "CI": f"[{fmt_float(row['wql_rer_ci_low'])}, {fmt_float(row['wql_rer_ci_high'])}]",
            "Harm": pct(row["wql_harm_rate"]),
            "Coverage": pct(row["coverage_q10_q90"]),
            "MAE-RER": fmt_float(row["mae_rer"]),
        }
        for row in q9_rows
    ]

    lines = [
        "# AAAI Oral Goal Status Dashboard",
        "",
        "This dashboard is generated from the current repository artifacts. It separates what is complete on the locked current suite from what still requires new inference, stronger baselines, or broader external-scale confirmation.",
        "",
        "## Executive Read",
        "",
        f"- Active target: push DRCR from a 930-window paper-readiness prototype toward an AAAI main/oral-level experimental package.",
        f"- Current selected method: `{status.get('selected_candidate_id', SELECTED)}`.",
        f"- Unique forecast windows: `{len(split_rows)}`; expanded method rows in figure windows: `{len(read_csv(WINDOWS))}`.",
        f"- Model families in current suite: `{', '.join(sorted(family_counts))}`.",
        f"- Dataset count in current suite: `{len({row.get('dataset', '') for row in split_rows})}`.",
        f"- Evidence tiers: `{dict(tier_counts)}`.",
        f"- Oral evidence gap queue: `{p0_gap_count}` P0 source reruns, `{q9_gap_count}` q9/full-grid source gaps, `{history_gap_count}` history/context export gaps.",
        f"- Oral rerun command queue: `{ready_commands}` ready commands, `{p0_ready_commands}` P0 ready, `{blocked_commands}` blocked commands.",
        f"- Remote q9 execution pack: `{len(remote_plan_rows)}` P0 sources / `{remote_plan_windows}` windows ready for larger-memory execution; completion audit `{remote_complete_sources}`/`{len(remote_plan_rows)}` complete, ingestion manifest `{ingestion_ready_sources}`/`{len(ingestion_rows)}` ready for final-main refresh.",
        f"- Role mix: `{dict(role_counts)}`.",
        f"- Overall WQL-RER: native `{fmt_float(overall_native.get('repair_median_wql_rer'))}` -> selected DRCR `{fmt_float(overall_selected.get('repair_median_wql_rer'))}`.",
        f"- q9 failure WQL-RER: native `{fmt_float(q9_native.get('repair_median_wql_rer'))}` -> selected DRCR `{fmt_float(q9_selected.get('repair_median_wql_rer'))}`.",
        (
            f"- Confirmatory salted split: `{confirm_status.get('n_calibration_windows')}` / "
            f"`{confirm_status.get('n_test_windows')}`, selected `{confirm_status.get('selected_candidate_id')}`, "
            f"overall WQL-RER `{fmt_float(confirm_overall_native.get('repair_median_wql_rer'))}` -> "
            f"`{fmt_float(confirm_overall_selected.get('repair_median_wql_rer'))}`, q9 failure "
            f"`{fmt_float(confirm_q9_native.get('repair_median_wql_rer'))}` -> "
            f"`{fmt_float(confirm_q9_selected.get('repair_median_wql_rer'))}`."
            if confirm_status.get("status") == "ok"
            else "- Confirmatory salted split: not yet available."
        ),
        f"- Requirement status counts: `{dict(status_counts)}`.",
        "",
        "## What Is Complete vs Still Missing",
        "",
        markdown_table(req_display, [("area", "Area"), ("status", "Status"), ("evidence", "Evidence"), ("gap", "Remaining gap")]),
        "",
        "## q9 Full-Grid Failure Results",
        "",
        markdown_table(q9_display, [("Method", "Method"), ("WQL-RER", "WQL-RER"), ("CI", "95% CI"), ("Harm", "Harm"), ("Coverage", "Coverage"), ("MAE-RER", "MAE-RER")]),
        "",
        "## Benchmark Coverage Snapshot",
        "",
        markdown_table(
            coverage_display,
            [
                ("family", "Family"),
                ("model", "Model"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("evidence_tier", "Tier"),
                ("calibration_windows", "Cal"),
                ("test_windows", "Test"),
                ("total_windows", "Total"),
            ],
        ),
        "",
        "## Paper-Safe Interpretation",
        "",
        "- Safe now: DRCR is a calibration-selected, LTT/CRC-style empirical risk-screened external repair layer that improves the current locked stress/positive-control suite.",
        "- Not safe yet: DRCR has a strict distribution-free finite-sample guarantee for dependent time-series windows.",
        "- Safe now: objective baselines show the method is not merely pure classical fallback, global blending, or a single threshold gate.",
        "- Not safe yet: broad universal TSFM reliability claims across all GIFT-Eval/Monash/ETT/M4/M5 datasets.",
        "",
        "## Generated Artifacts",
        "",
        f"- `{REQ_OUT.relative_to(ROOT)}`",
        f"- `{COVERAGE_OUT.relative_to(ROOT)}`",
        f"- `{MODEL_DATASET_OUT.relative_to(ROOT)}`",
        f"- `{Q9_OUT.relative_to(ROOT)}`",
        f"- `{CONTAM_OUT.relative_to(ROOT)}`",
        f"- `{CLASSICAL_FEASIBILITY.relative_to(ROOT)}`",
        f"- `{ORAL_SOURCE_GAPS.relative_to(ROOT)}`",
        f"- `{ORAL_FAMILY_GAPS.relative_to(ROOT)}`",
        f"- `{ORAL_RERUN_COMMANDS.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_RERUN_PLAN.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_RERUN_AUDIT.relative_to(ROOT)}`",
        f"- `{REMOTE_Q9_INGESTION.relative_to(ROOT)}`",
        f"- `docs/remote_q9_rerun_execution_pack.md`",
        f"- `docs/remote_q9_rerun_completion_audit.md`",
        f"- `docs/remote_q9_ingestion_manifest.md`",
        f"- `{(OUT / 'rerun_manifests').relative_to(ROOT)}/`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
    ]
    DOC_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC_OUT.relative_to(ROOT)}")
    print(f"requirements {dict(status_counts)}")


if __name__ == "__main__":
    main()
