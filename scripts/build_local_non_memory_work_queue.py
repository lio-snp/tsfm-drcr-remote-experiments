#!/usr/bin/env python3
"""Separate external-compute q9 reruns from remaining local non-memory work."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

REMOTE_PLAN = OUT / "remote_q9_rerun_plan.csv"
REMOTE_AUDIT = OUT / "remote_q9_rerun_completion_audit.csv"
INGESTION = OUT / "remote_q9_ingestion_manifest.csv"
DEFERRED_OUT = OUT / "external_compute_deferred_manifest.csv"
QUEUE_OUT = OUT / "local_non_memory_work_queue.csv"
DOC_OUT = DOCS / "local_non_memory_work_queue.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    remote_rows = read_csv(REMOTE_PLAN)
    audit_by_source = {row["source"]: row for row in read_csv(REMOTE_AUDIT)}
    ingestion_by_source = {row["original_source"]: row for row in read_csv(INGESTION)}

    deferred_rows: list[dict[str, object]] = []
    for row in remote_rows:
        source = row["source"]
        audit = audit_by_source.get(source, {})
        ingestion = ingestion_by_source.get(source, {})
        complete = audit.get("complete_for_ingestion") == "1"
        ready = ingestion.get("ready_for_final_main_refresh") == "1"
        deferred_rows.append(
            {
                "source": source,
                "family": row.get("family", ""),
                "dataset": row.get("dataset", ""),
                "role": row.get("role", ""),
                "manifest_windows": row.get("manifest_windows", ""),
                "deferred_status": "deferred_external_compute" if not ready else "ready_for_ingestion",
                "reason": "local_8gb_memory_or_runtime_insufficient_for_model_inference" if not ready else "remote_artifacts_ready",
                "complete_for_ingestion": int(complete),
                "ready_for_final_main_refresh": int(ready),
                "paper_boundary": "do_not_count_as_completed_q9_evidence_until_ready_for_final_main_refresh",
            }
        )
    write_csv(DEFERRED_OUT, deferred_rows)

    queue_rows = [
        {
            "priority": "L0",
            "work_item": "Finalize current-suite claim table and downgrade unsupported claims",
            "blocked_by_memory": 0,
            "status": "ready_local",
            "artifact_or_command": "docs/claim_table_safe_vs_unsafe.md; docs/final_drcr_paper_readiness_summary.md",
            "success_criterion": "Claims distinguish current-suite evidence from deferred external q9 evidence.",
        },
        {
            "priority": "L0",
            "work_item": "Polish final main figure captions and labels for plain-language interpretation",
            "blocked_by_memory": 0,
            "status": "ready_local",
            "artifact_or_command": "docs/caption_plain_language.md; figures/aaai_stress/final_main_figure_draft.*",
            "success_criterion": "Figure captions explain Native/Baselines/DRCR and each axis without internal shorthand.",
        },
        {
            "priority": "L1",
            "work_item": "Complete paper-faithful metric appendix boundary table",
            "blocked_by_memory": 0,
            "status": "ready_local",
            "artifact_or_command": "docs/paper_faithful_metric_robustness.md; docs/metric_mapping_table.md",
            "success_criterion": "Moirai/GIFT, Chronos, and TimesFM metric scopes are explicitly mapped to available evidence.",
        },
        {
            "priority": "L1",
            "work_item": "Add reviewer-facing evidence-scope table",
            "blocked_by_memory": 0,
            "status": "ready_local",
            "artifact_or_command": "docs/aaai_reviewer_audit.md; docs/aaai_oral_goal_status.md",
            "success_criterion": "Table says what is current-suite complete, external-compute deferred, or future-scale.",
        },
        {
            "priority": "L1",
            "work_item": "Strengthen denominator-fragility appendix with absolute-delta wording",
            "blocked_by_memory": 0,
            "status": "ready_local",
            "artifact_or_command": "docs/denominator_fragility_report.md",
            "success_criterion": "RER fragility is framed as sensitivity, not as the only failure evidence.",
        },
        {
            "priority": "L2",
            "work_item": "Strict benchmark-native classical interval parity",
            "blocked_by_memory": 0,
            "status": "local_compute_slow_or_future",
            "artifact_or_command": "docs/native_classical_interval_audit.md",
            "success_criterion": "If not run, keep as future work and avoid calling empirical residual intervals native parity.",
        },
        {
            "priority": "L2",
            "work_item": "Full benchmark breadth expansion beyond six datasets",
            "blocked_by_memory": 0,
            "status": "future_scale_not_required_for_current_local_branch",
            "artifact_or_command": "docs/dataset_inventory.md; docs/benchmark_reproduction_plan.md",
            "success_criterion": "Current paper claims remain bounded; expansion is not implied complete.",
        },
    ]
    write_csv(QUEUE_OUT, queue_rows)

    deferred_count = sum(row["deferred_status"] == "deferred_external_compute" for row in deferred_rows)
    ready_count = sum(row["status"] == "ready_local" for row in queue_rows)
    lines = [
        "# Local Non-Memory Work Queue",
        "",
        "This queue separates q9/full-grid model reruns that are blocked by local 8GB memory/runtime from work that can still be advanced locally. Deferred does not mean completed; it means the local branch should not wait on that inference before polishing the current-suite paper package.",
        "",
        "## Summary",
        "",
        f"- External-compute deferred q9 sources: `{deferred_count}`.",
        f"- Local ready non-memory work items: `{ready_count}`.",
        "- Paper boundary: deferred q9 reruns cannot be counted as completed q9/full-grid evidence.",
        "",
        "## Deferred External-Compute Sources",
        "",
        markdown_table(
            deferred_rows,
            [
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("source", "Source"),
                ("manifest_windows", "Windows"),
                ("deferred_status", "Status"),
                ("complete_for_ingestion", "Complete"),
                ("ready_for_final_main_refresh", "Ready"),
            ],
        ),
        "",
        "## Local Non-Memory Queue",
        "",
        markdown_table(
            queue_rows,
            [
                ("priority", "Priority"),
                ("work_item", "Work item"),
                ("blocked_by_memory", "Memory blocked"),
                ("status", "Status"),
                ("success_criterion", "Success criterion"),
            ],
        ),
        "",
        "## Artifacts",
        "",
        f"- `{DEFERRED_OUT.relative_to(ROOT)}`",
        f"- `{QUEUE_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "deferred_external_compute": deferred_count, "local_ready": ready_count})


if __name__ == "__main__":
    main()
