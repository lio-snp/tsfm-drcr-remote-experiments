#!/usr/bin/env python3
"""Audit whether remote q9/full-grid rerun artifacts have actually arrived."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

PLAN = OUT / "remote_q9_rerun_plan.csv"
AUDIT_OUT = OUT / "remote_q9_rerun_completion_audit.csv"
DOC_OUT = DOCS / "remote_q9_rerun_completion_audit.md"

Q9_COLUMNS = [f"forecast_q{level}" for level in range(10, 100, 10)]


def read_csv_optional(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_required(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv_optional(path)


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


def row_count(path: Path) -> int:
    return len(read_csv_optional(path))


def raw_window_count(rows: list[dict[str, str]]) -> int:
    keys = {
        (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("series_id", ""),
            row.get("origin", ""),
            row.get("window_index", ""),
        )
        for row in rows
    }
    keys.discard(("", "", "", "", ""))
    return len(keys)


def nonempty_q9_coverage(rows: list[dict[str, str]], columns: list[str]) -> float:
    if not rows or not columns:
        return 0.0
    ok = 0
    for row in rows:
        if all(str(row.get(column, "")).strip() not in {"", "nan", "None"} for column in columns):
            ok += 1
    return ok / len(rows)


def status_value(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "invalid_json"
    return str(data.get("status", "unknown"))


def slug_from_raw_path(value: str) -> str:
    if not value:
        return ""
    return Path(value).stem


def main() -> None:
    plan_rows = read_csv_required(PLAN)
    audit_rows: list[dict[str, object]] = []
    for row in plan_rows:
        raw_path = ROOT / row["expected_raw_path"]
        sidecar_path = ROOT / row["expected_history_sidecar_path"]
        status_path = ROOT / row["expected_status_path"]
        manifest_windows = int(row.get("manifest_windows") or 0)

        raw_rows = read_csv_optional(raw_path)
        raw_columns = set(raw_rows[0].keys()) if raw_rows else set()
        missing_q9 = [column for column in Q9_COLUMNS if column not in raw_columns]
        raw_windows = raw_window_count(raw_rows)
        sidecar_rows = row_count(sidecar_path)
        q9_nonempty_rate = nonempty_q9_coverage(raw_rows, Q9_COLUMNS)

        raw_ready = raw_path.exists() and len(raw_rows) > 0 and raw_windows >= manifest_windows
        sidecar_ready = sidecar_path.exists() and sidecar_rows == manifest_windows
        q9_ready = not missing_q9 and q9_nonempty_rate >= 0.999
        status_ok = status_value(status_path) == "ok"
        complete = raw_ready and sidecar_ready and q9_ready and status_ok
        audit_rows.append(
            {
                "source": row.get("source", ""),
                "rerun_slug": slug_from_raw_path(row.get("expected_raw_path", "")),
                "family": row.get("family", ""),
                "dataset": row.get("dataset", ""),
                "role": row.get("role", ""),
                "manifest_windows": manifest_windows,
                "raw_exists": int(raw_path.exists()),
                "raw_rows": len(raw_rows),
                "raw_window_count": raw_windows,
                "sidecar_exists": int(sidecar_path.exists()),
                "sidecar_rows": sidecar_rows,
                "status_exists": int(status_path.exists()),
                "runner_status": status_value(status_path),
                "missing_q9_columns": ";".join(missing_q9),
                "q9_nonempty_rate": f"{q9_nonempty_rate:.6f}",
                "raw_ready": int(raw_ready),
                "sidecar_ready": int(sidecar_ready),
                "q9_ready": int(q9_ready),
                "status_ok": int(status_ok),
                "complete_for_ingestion": int(complete),
                "expected_raw_path": row.get("expected_raw_path", ""),
                "expected_history_sidecar_path": row.get("expected_history_sidecar_path", ""),
            }
        )
    write_csv(AUDIT_OUT, audit_rows)

    complete_count = sum(int(row["complete_for_ingestion"]) for row in audit_rows)
    raw_count = sum(int(row["raw_exists"]) for row in audit_rows)
    sidecar_count = sum(int(row["sidecar_exists"]) for row in audit_rows)
    q9_ready_count = sum(int(row["q9_ready"]) for row in audit_rows)
    incomplete = [row for row in audit_rows if not int(row["complete_for_ingestion"])]
    backfill_slugs = [str(row["rerun_slug"]) for row in audit_rows if int(row["complete_for_ingestion"])]
    backfill_command = "python3 scripts/build_missing_predictor_features_from_metrics.py <complete_rerun_slug_1> <complete_rerun_slug_2> ..."
    if backfill_slugs:
        backfill_command = "python3 scripts/build_missing_predictor_features_from_metrics.py " + " ".join(backfill_slugs)

    lines = [
        "# Remote q9/Full-Grid Rerun Completion Audit",
        "",
        "This audit answers a narrower question than the main paper dashboard: have the remote q9/full-grid rerun artifacts actually arrived and are they complete enough to be ingested into the final-main suite?",
        "",
        "## Summary",
        "",
        f"- Planned P0 sources: `{len(audit_rows)}`.",
        f"- Complete for ingestion: `{complete_count}`.",
        f"- Sources with raw CSV present: `{raw_count}`.",
        f"- Sources with sidecar CSV present: `{sidecar_count}`.",
        f"- Sources with all q10..q90 columns non-empty: `{q9_ready_count}`.",
        "- Interpretation: a source is complete only when raw forecasts, exact history/context sidecar, runner status, and nine quantile columns are all present. This audit does not run model inference.",
        "",
        "## Incomplete / Pending Sources",
        "",
        markdown_table(
            incomplete,
            [
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("source", "Source"),
                ("manifest_windows", "Windows"),
                ("raw_exists", "Raw"),
                ("raw_window_count", "Raw windows"),
                ("sidecar_exists", "Sidecar"),
                ("sidecar_rows", "Sidecar rows"),
                ("runner_status", "Status"),
                ("missing_q9_columns", "Missing q cols"),
            ],
        )
        if incomplete
        else "No incomplete P0 sources; all remote q9 reruns are ready for ingestion.",
        "",
        "## Post-Completion Ingestion Steps",
        "",
        "After all 17 sources are complete, backfill predictor features for the rerun slugs, then refresh the final-main source inventory so the `_oral_sidecar_rerun` slugs replace their q3/proxy predecessors as q9/full-grid evidence.",
        "",
        "```bash",
        "# If no slugs are complete yet, rerun this audit after remote outputs arrive.",
        backfill_command,
        "```",
        "",
        "Important boundary: `build_oral_evidence_gap_matrix.py` is still based on the current final-main manifests. It will not clear these gaps merely because new raw files exist. The final-main source inventory must be refreshed or explicitly pointed at the rerun slugs before the main dashboard can claim the q9/full-grid gap is closed.",
        "",
        "## Artifacts",
        "",
        f"- `{AUDIT_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        f"- Source plan: `{PLAN.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "planned": len(audit_rows), "complete_for_ingestion": complete_count, "audit": str(AUDIT_OUT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
