#!/usr/bin/env python3
"""Summarize local resource blockers for oral q9/full-grid reruns."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

EXEC_STATUS = OUT / "oral_rerun_execution_status.csv"
SOURCE_GAPS = OUT / "oral_evidence_source_gap_matrix.csv"
CSV_OUT = OUT / "local_rerun_resource_blockers.csv"
DOC_OUT = DOCS / "local_rerun_resource_blockers.md"


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


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def tail(path: Path, n: int = 12) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> None:
    gap_rows = read_csv(SOURCE_GAPS)
    exec_rows = read_csv(EXEC_STATUS)
    blockers: list[dict[str, object]] = []
    for row in exec_rows:
        status = row.get("execution_status", "")
        if status not in {"timeout", "failed_or_missing_artifacts"}:
            continue
        blockers.append(
            {
                "source": row.get("source", ""),
                "family": row.get("family", ""),
                "dataset": row.get("dataset", ""),
                "role": row.get("role", ""),
                "manifest_windows": row.get("manifest_windows", ""),
                "execution_status": status,
                "returncode": row.get("returncode", ""),
                "elapsed_seconds": row.get("elapsed_seconds", ""),
                "raw_rows": row.get("raw_rows", ""),
                "history_sidecar_rows": row.get("history_sidecar_rows", ""),
                "log_path": row.get("log_path", ""),
            }
        )
    write_csv(CSV_OUT, blockers)

    q9_gap_count = sum(row.get("needs_q9_fullgrid_rerun") == "1" for row in gap_rows)
    p0_gap_count = sum(row.get("priority") == "P0" for row in gap_rows)
    attempted_sources = {row["source"] for row in blockers}
    attempted_q9_gaps = sum(row.get("source") in attempted_sources and row.get("needs_q9_fullgrid_rerun") == "1" for row in gap_rows)
    timeout_count = sum(row["execution_status"] == "timeout" for row in blockers)

    log_sections: list[str] = []
    for row in blockers:
        log_path = ROOT / str(row.get("log_path", ""))
        excerpt = tail(log_path)
        if excerpt:
            log_sections.extend(
                [
                    f"### {row['source']}",
                    "",
                    "```text",
                    excerpt,
                    "```",
                    "",
                ]
            )

    lines = [
        "# Local Rerun Resource Blockers",
        "",
        "This report records local attempts to execute the remaining oral q9/full-grid reruns. It distinguishes a ready command queue from what this 8GB local machine can actually complete.",
        "",
        "## Summary",
        "",
        f"- Remaining q9/full-grid source gaps: `{q9_gap_count}`.",
        f"- Remaining P0 source gaps: `{p0_gap_count}`.",
        f"- Local attempted q9/full-grid gap sources: `{attempted_q9_gaps}`.",
        f"- Timeout blocker count: `{timeout_count}`.",
        "- Interpretation: command manifests are ready, but local execution is not sufficient for the remaining Moirai/TimesFM q9 reruns; use remote or a larger-memory machine for the final broad probabilistic evidence.",
        "",
        "## Attempted Sources",
        "",
        markdown_table(
            blockers,
            [
                ("source", "Source"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("manifest_windows", "Windows"),
                ("execution_status", "Status"),
                ("elapsed_seconds", "Seconds"),
                ("raw_rows", "Raw rows"),
                ("history_sidecar_rows", "Sidecar rows"),
                ("log_path", "Log"),
            ],
        )
        if blockers
        else "No local rerun blockers recorded.",
        "",
        "## Log Excerpts",
        "",
        *log_sections,
        "## Artifacts",
        "",
        f"- `{CSV_OUT.relative_to(ROOT)}`",
        f"- `{EXEC_STATUS.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "blockers": len(blockers), "timeouts": timeout_count, "q9_gaps": q9_gap_count})


if __name__ == "__main__":
    main()
