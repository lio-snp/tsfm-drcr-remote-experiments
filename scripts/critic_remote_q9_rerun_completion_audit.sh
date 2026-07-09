#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
audit_path = root / "results/aaai_stress/remote_q9_rerun_completion_audit.csv"
doc_path = root / "docs/remote_q9_rerun_completion_audit.md"
plan_path = root / "results/aaai_stress/remote_q9_rerun_plan.csv"

for path in [audit_path, doc_path, plan_path]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty required artifact: {path}")

audit_rows = list(csv.DictReader(audit_path.open()))
plan_rows = list(csv.DictReader(plan_path.open()))
if len(audit_rows) != len(plan_rows):
    raise SystemExit(f"Audit/plan row mismatch: {len(audit_rows)} vs {len(plan_rows)}")
if len(audit_rows) < 17:
    raise SystemExit(f"Too few audited P0 sources: {len(audit_rows)}")

required_columns = {
    "source",
    "rerun_slug",
    "raw_window_count",
    "sidecar_rows",
    "missing_q9_columns",
    "q9_nonempty_rate",
    "complete_for_ingestion",
}
missing = required_columns - set(audit_rows[0])
if missing:
    raise SystemExit(f"Completion audit missing columns: {sorted(missing)}")

families = {row["family"] for row in audit_rows}
if families != {"moirai", "timesfm"}:
    raise SystemExit(f"Unexpected families in audit: {families}")

doc = doc_path.read_text()
for phrase in [
    "Complete for ingestion",
    "Post-Completion Ingestion Steps",
    "build_missing_predictor_features_from_metrics.py",
    "will not clear these gaps merely because new raw files exist",
]:
    if phrase not in doc:
        raise SystemExit(f"Completion audit doc missing phrase: {phrase}")

print("[remote q9 rerun completion audit critic] PASS")
print({"p0_sources": len(audit_rows), "complete": sum(int(row["complete_for_ingestion"]) for row in audit_rows)})
PY
