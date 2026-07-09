#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
manifest_path = root / "results/aaai_stress/remote_q9_ingestion_manifest.csv"
doc_path = root / "docs/remote_q9_ingestion_manifest.md"
plan_path = root / "results/aaai_stress/remote_q9_rerun_plan.csv"

for path in [manifest_path, doc_path, plan_path]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty required artifact: {path}")

rows = list(csv.DictReader(manifest_path.open()))
plan_rows = list(csv.DictReader(plan_path.open()))
if len(rows) != len(plan_rows):
    raise SystemExit(f"Ingestion manifest row mismatch: {len(rows)} vs {len(plan_rows)}")
if len(rows) < 17:
    raise SystemExit(f"Too few ingestion rows: {len(rows)}")

required = {
    "original_source",
    "rerun_slug",
    "old_evidence_tier",
    "new_evidence_tier",
    "ready_for_final_main_refresh",
    "paper_boundary",
}
missing = required - set(rows[0])
if missing:
    raise SystemExit(f"Ingestion manifest missing columns: {sorted(missing)}")
if any(row["new_evidence_tier"] != "q9_fullgrid" for row in rows):
    raise SystemExit("All remote q9 replacement rows must target q9_fullgrid")
if any(not row["rerun_slug"].endswith("_oral_sidecar_rerun") for row in rows):
    raise SystemExit("Unexpected rerun slug without _oral_sidecar_rerun suffix")

doc = doc_path.read_text()
for phrase in [
    "Ready for final-main refresh",
    "replace_original_source_with_rerun_slug",
    "do not count as q9/fullgrid evidence until ready_for_final_main_refresh=1",
    "Refresh Protocol",
]:
    if phrase not in doc:
        raise SystemExit(f"Ingestion manifest doc missing phrase: {phrase}")

print("[remote q9 ingestion manifest critic] PASS")
print({"replacement_sources": len(rows), "ready": sum(int(row["ready_for_final_main_refresh"]) for row in rows)})
PY
