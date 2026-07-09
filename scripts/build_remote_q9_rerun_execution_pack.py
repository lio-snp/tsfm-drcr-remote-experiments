#!/usr/bin/env python3
"""Build a remote execution pack for the remaining oral q9/full-grid reruns."""

from __future__ import annotations

import csv
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

COMMAND_MANIFEST = OUT / "oral_rerun_command_manifest.csv"
GAP_MATRIX = OUT / "oral_evidence_source_gap_matrix.csv"
EXEC_STATUS = OUT / "oral_rerun_execution_status.csv"
PLAN_OUT = OUT / "remote_q9_rerun_plan.csv"
DOC_OUT = DOCS / "remote_q9_rerun_execution_pack.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
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


def option_value(command: str, option: str) -> str:
    args = shlex.split(command)
    try:
        index = args.index(option)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return args[index + 1]


def expected_outputs(command: str) -> tuple[str, str, str]:
    slug = option_value(command, "--output-slug")
    if not slug:
        return "", "", ""
    raw = f"results/raw_forecasts/{slug}.csv"
    sidecar = f"results/raw_forecasts/{slug}_history_context.csv"
    status = f"results/raw_forecasts/{slug}_status.json"
    return raw, sidecar, status


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def command_block(command: str) -> str:
    return f"```bash\n{command}\n```"


def family_queue_command(family: str, timeout: int = 7200) -> str:
    return (
        "python3 scripts/run_oral_rerun_queue.py "
        f"--priority P0 --family {family} --timeout-seconds {timeout}"
    )


def validation_commands() -> list[str]:
    return [
        "python3 scripts/build_remote_q9_rerun_completion_audit.py",
        "bash scripts/critic_remote_q9_rerun_completion_audit.sh",
        "python3 scripts/build_remote_q9_ingestion_manifest.py",
        "bash scripts/critic_remote_q9_ingestion_manifest.sh",
        "python3 scripts/build_oral_evidence_gap_matrix.py",
        "python3 scripts/build_oral_rerun_command_manifest.py",
        "python3 scripts/build_aaai_oral_goal_status.py",
        "python3 scripts/build_drcr_paper_readiness_reports.py",
        "bash scripts/critic_oral_evidence_gap_matrix.sh",
        "bash scripts/critic_oral_rerun_command_manifest.sh",
        "bash scripts/critic_drcr_paper_readiness_reports.sh",
    ]


def main() -> None:
    command_rows = read_csv(COMMAND_MANIFEST)
    gap_rows = {row.get("source", ""): row for row in read_csv(GAP_MATRIX)}
    status_rows = {row.get("source", ""): row for row in read_csv(EXEC_STATUS)}

    p0_rows = [
        row
        for row in command_rows
        if row.get("priority") == "P0"
        and row.get("needs_q9_fullgrid_rerun") == "1"
        and row.get("command_status") == "ready"
    ]

    plan_rows: list[dict[str, object]] = []
    for index, row in enumerate(p0_rows, start=1):
        raw, sidecar, status = expected_outputs(row.get("command", ""))
        exec_row = status_rows.get(row.get("source", ""), {})
        gap_row = gap_rows.get(row.get("source", ""), {})
        windows = int(row.get("manifest_windows") or 0)
        plan_rows.append(
            {
                "run_order": index,
                "priority": row.get("priority", ""),
                "family": row.get("family", ""),
                "dataset": row.get("dataset", ""),
                "role": row.get("role", ""),
                "source": row.get("source", ""),
                "manifest_windows": windows,
                "manifest_path": row.get("manifest_path", ""),
                "expected_raw_path": raw,
                "expected_history_sidecar_path": sidecar,
                "expected_status_path": status,
                "current_execution_status": exec_row.get("execution_status", "not_attempted"),
                "current_raw_rows": exec_row.get("raw_rows", "0"),
                "current_sidecar_rows": exec_row.get("history_sidecar_rows", "0"),
                "gap_reason": "q9/full-grid source gap" if gap_row.get("needs_q9_fullgrid_rerun") == "1" else "",
                "command": row.get("command", ""),
            }
        )

    write_csv(PLAN_OUT, plan_rows)

    family_rows: list[dict[str, object]] = []
    for family in sorted({str(row["family"]) for row in plan_rows}):
        rows = [row for row in plan_rows if row["family"] == family]
        family_rows.append(
            {
                "family": family,
                "sources": len(rows),
                "windows": sum(int(row["manifest_windows"]) for row in rows),
                "failure_targets": sum(row["role"] == "failure_target" for row in rows),
                "positive_controls": sum(row["role"] == "positive_control" for row in rows),
                "stress_targets": sum(row["role"] == "stress_target" for row in rows),
            }
        )

    role_rows: list[dict[str, object]] = []
    for role in sorted({str(row["role"]) for row in plan_rows}):
        rows = [row for row in plan_rows if row["role"] == role]
        role_rows.append({"role": role, "sources": len(rows), "windows": sum(int(row["manifest_windows"]) for row in rows)})

    total_windows = sum(int(row["manifest_windows"]) for row in plan_rows)
    timeout_blockers = sum(row["current_execution_status"] == "timeout" for row in plan_rows)

    lines = [
        "# Remote q9/Full-Grid Rerun Execution Pack",
        "",
        "This pack is the execution contract for the remaining P0 probabilistic-evidence gaps. It is meant for a remote or larger-memory machine; local 8GB attempts already timed out and should not be treated as failed science.",
        "",
        "## Scope",
        "",
        f"- P0 sources to rerun: `{len(plan_rows)}`.",
        f"- Total forecast windows: `{total_windows}`.",
        f"- Families: `{', '.join(sorted({str(row['family']) for row in plan_rows}))}`.",
        f"- Current local timeout blockers among these sources: `{timeout_blockers}`.",
        "- Required evidence per source: raw forecast CSV, history/context sidecar CSV, and runner status JSON.",
        "- Success criterion: every P0 row in `results/aaai_stress/remote_q9_rerun_plan.csv` has non-empty expected raw and sidecar outputs, then the gap matrix and paper-readiness reports are rebuilt and critics pass.",
        "",
        "## Hardware / Runtime Recommendation",
        "",
        "- Minimum practical target: 16GB RAM.",
        "- Preferred target: 32GB RAM or more, especially for TimesFM 2.5 and Moirai 1.1 large/base runs.",
        "- Keep the same repository checkout and data paths; commands use source-specific window manifests to avoid mixing windows.",
        "- Use family-specific environments if present: `.venv-moirai` for Moirai and `.venv-chronos` for TimesFM/Chronos. The queue runner handles this automatically.",
        "",
        "## Family Summary",
        "",
        markdown_table(
            family_rows,
            [
                ("family", "Family"),
                ("sources", "Sources"),
                ("windows", "Windows"),
                ("failure_targets", "Failure targets"),
                ("positive_controls", "Positive controls"),
                ("stress_targets", "Stress targets"),
            ],
        ),
        "",
        "## Role Summary",
        "",
        markdown_table(role_rows, [("role", "Role"), ("sources", "Sources"), ("windows", "Windows")]),
        "",
        "## Recommended Execution",
        "",
        "Run Moirai and TimesFM separately so failures are isolated and resumable.",
        "",
        "### Dry Run",
        "",
        command_block("python3 scripts/run_oral_rerun_queue.py --priority P0 --dry-run"),
        "",
        "### Moirai",
        "",
        command_block(family_queue_command("moirai")),
        "",
        "### TimesFM",
        "",
        command_block(family_queue_command("timesfm")),
        "",
        "If the remote environment has enough RAM, do not use `--allow-low-memory`; that flag only bypasses local preflight checks and does not make the inference cheaper. If a preflight check is too conservative on a known large machine, rerun the same command with `--allow-low-memory` and a longer timeout.",
        "",
        "## Post-Run Validation",
        "",
        "After the queue finishes, first audit whether the expected raw/sidecar/status artifacts actually arrived, then rebuild the current dashboard reports:",
        "",
        command_block("\n".join(validation_commands())),
        "",
        "Expected validation outcome after a complete successful remote run:",
        "",
        "- `results/aaai_stress/remote_q9_rerun_completion_audit.csv` reports all 17 P0 sources as `complete_for_ingestion=1`.",
        "- Raw files have non-empty `forecast_q10..forecast_q90` columns; sidecar files have one row per manifest window; runner status JSON files report `ok`.",
        "- The current gap matrix may still show q9/full-grid gaps until the final-main source inventory is refreshed to point at the `_oral_sidecar_rerun` slugs. New raw files alone do not update the locked final-main manifests.",
        "- After the ingestion/refresh step, `docs/aaai_oral_goal_status.md` and `docs/final_drcr_paper_readiness_summary.md` should stop describing these Moirai/TimesFM sources as only proxy-level evidence.",
        "- The critic scripts above pass.",
        "",
        "## P0 Source Plan",
        "",
        markdown_table(
            plan_rows,
            [
                ("run_order", "#"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("source", "Source"),
                ("manifest_windows", "Windows"),
                ("current_execution_status", "Current status"),
                ("expected_raw_path", "Expected raw"),
                ("expected_history_sidecar_path", "Expected sidecar"),
            ],
        ),
        "",
        "## Exact Commands",
        "",
    ]
    for row in plan_rows:
        lines.extend(
            [
                f"### {row['run_order']}. {row['source']}",
                "",
                f"- Family: `{row['family']}`; dataset: `{row['dataset']}`; role: `{row['role']}`; windows: `{row['manifest_windows']}`.",
                f"- Expected raw: `{row['expected_raw_path']}`.",
                f"- Expected sidecar: `{row['expected_history_sidecar_path']}`.",
                "",
                command_block(str(row["command"])),
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{PLAN_OUT.relative_to(ROOT)}`",
            f"- `{DOC_OUT.relative_to(ROOT)}`",
            f"- Source command manifest: `{COMMAND_MANIFEST.relative_to(ROOT)}`",
            f"- Gap matrix: `{GAP_MATRIX.relative_to(ROOT)}`",
            "",
        ]
    )
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "p0_sources": len(plan_rows), "windows": total_windows, "doc": str(DOC_OUT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
