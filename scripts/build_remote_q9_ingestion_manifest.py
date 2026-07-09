#!/usr/bin/env python3
"""Build the final-main ingestion map for completed remote q9 reruns."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"
RAW_DIR = ROOT / "results" / "raw_forecasts"
FEATURE_DIR = ROOT / "results" / "failure_mining"
METRIC_DIR = ROOT / "results" / "window_metrics"

PLAN = OUT / "remote_q9_rerun_plan.csv"
AUDIT = OUT / "remote_q9_rerun_completion_audit.csv"
DATASET_MANIFEST = OUT / "dataset_manifest.csv"
INGESTION_OUT = OUT / "remote_q9_ingestion_manifest.csv"
DOC_OUT = DOCS / "remote_q9_ingestion_manifest.md"


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


def slug_from_raw_path(value: str) -> str:
    return Path(value).stem if value else ""


def main() -> None:
    plan_rows = read_csv(PLAN)
    audit_by_source = {row["source"]: row for row in read_csv(AUDIT)}
    dataset_by_source = {row["source"]: row for row in read_csv(DATASET_MANIFEST)}

    rows: list[dict[str, object]] = []
    for plan in plan_rows:
        source = plan["source"]
        dataset_row = dataset_by_source.get(source, {})
        audit = audit_by_source.get(source, {})
        rerun_slug = slug_from_raw_path(plan.get("expected_raw_path", ""))
        raw_path = RAW_DIR / f"{rerun_slug}.csv"
        metric_path = METRIC_DIR / f"{rerun_slug}_metrics.csv"
        feature_path = FEATURE_DIR / f"{rerun_slug}_predictor_features.csv"
        complete = audit.get("complete_for_ingestion") == "1"
        features_ready = feature_path.exists() and feature_path.stat().st_size > 0
        metrics_ready = metric_path.exists() and metric_path.stat().st_size > 0
        rows.append(
            {
                "original_source": source,
                "rerun_slug": rerun_slug,
                "family": plan.get("family", ""),
                "dataset": plan.get("dataset") or dataset_row.get("dataset", ""),
                "target_id": dataset_row.get("target_id", ""),
                "role": plan.get("role") or dataset_row.get("role", ""),
                "old_evidence_tier": dataset_row.get("evidence_tier", ""),
                "new_evidence_tier": "q9_fullgrid",
                "manifest_windows": plan.get("manifest_windows", ""),
                "raw_path": str(raw_path.relative_to(ROOT)),
                "metrics_path": str(metric_path.relative_to(ROOT)),
                "feature_path": str(feature_path.relative_to(ROOT)),
                "raw_complete": int(complete),
                "metrics_ready": int(metrics_ready),
                "features_ready": int(features_ready),
                "ready_for_final_main_refresh": int(complete and metrics_ready and features_ready),
                "refresh_action": "replace_original_source_with_rerun_slug",
                "paper_boundary": "do not count as q9/fullgrid evidence until ready_for_final_main_refresh=1",
            }
        )
    write_csv(INGESTION_OUT, rows)

    ready = sum(int(row["ready_for_final_main_refresh"]) for row in rows)
    complete = sum(int(row["raw_complete"]) for row in rows)
    metrics = sum(int(row["metrics_ready"]) for row in rows)
    features = sum(int(row["features_ready"]) for row in rows)

    lines = [
        "# Remote q9 Ingestion Manifest",
        "",
        "This manifest is the bridge between successful remote q9/full-grid reruns and a future final-main refresh. It does not claim new evidence is available; it records which locked q3/proxy sources should be replaced once their rerun raw, metrics, and predictor-feature artifacts are present.",
        "",
        "## Summary",
        "",
        f"- Replacement sources: `{len(rows)}`.",
        f"- Raw/sidecar/status complete by completion audit: `{complete}`.",
        f"- Metrics files present: `{metrics}`.",
        f"- Predictor features present: `{features}`.",
        f"- Ready for final-main refresh: `{ready}`.",
        "- Interpretation: a row is eligible for the next final-main rebuild only when `ready_for_final_main_refresh=1`.",
        "- Boundary: do not count as q9/fullgrid evidence until ready_for_final_main_refresh=1.",
        "",
        "## Replacement Map",
        "",
        markdown_table(
            rows,
            [
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("original_source", "Original source"),
                ("rerun_slug", "Rerun slug"),
                ("old_evidence_tier", "Old tier"),
                ("new_evidence_tier", "New tier"),
                ("raw_complete", "Raw complete"),
                ("metrics_ready", "Metrics"),
                ("features_ready", "Features"),
                ("ready_for_final_main_refresh", "Ready"),
            ],
        ),
        "",
        "## Refresh Protocol",
        "",
        "Refresh action: `replace_original_source_with_rerun_slug`.",
        "",
        "1. Run remote q9 reruns until `docs/remote_q9_rerun_completion_audit.md` reports all required sources complete.",
        "2. Backfill predictor features for complete rerun slugs using `scripts/build_missing_predictor_features_from_metrics.py`.",
        "3. Rebuild this ingestion manifest and confirm `ready_for_final_main_refresh=1` for the replacement rows.",
        "4. Refresh the final-main source inventory so each `original_source` is replaced by its `rerun_slug` with `new_evidence_tier=q9_fullgrid`.",
        "5. Rerun final-main figures, paper-readiness reports, and critics before upgrading any q9/full-grid claim.",
        "",
        "## Artifacts",
        "",
        f"- `{INGESTION_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "replacement_sources": len(rows), "ready_for_final_main_refresh": ready})


if __name__ == "__main__":
    main()
