#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
plan_path = root / "results/aaai_stress/remote_q9_rerun_plan.csv"
doc_path = root / "docs/remote_q9_rerun_execution_pack.md"
manifest_path = root / "results/aaai_stress/oral_rerun_command_manifest.csv"

for path in [plan_path, doc_path, manifest_path]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty required artifact: {path}")

plan_rows = list(csv.DictReader(plan_path.open()))
manifest_rows = list(csv.DictReader(manifest_path.open()))
p0_manifest = [
    row
    for row in manifest_rows
    if row.get("priority") == "P0"
    and row.get("needs_q9_fullgrid_rerun") == "1"
    and row.get("command_status") == "ready"
]

if len(plan_rows) != len(p0_manifest):
    raise SystemExit(f"P0 source count mismatch: plan={len(plan_rows)} manifest={len(p0_manifest)}")
if len(plan_rows) < 17:
    raise SystemExit(f"Too few P0 sources in remote pack: {len(plan_rows)}")

families = {row["family"] for row in plan_rows}
if families != {"moirai", "timesfm"}:
    raise SystemExit(f"Unexpected families in remote pack: {families}")

windows = sum(int(row["manifest_windows"]) for row in plan_rows)
if windows < 400:
    raise SystemExit(f"Too few windows in remote pack: {windows}")

for row in plan_rows:
    source = row["source"]
    for key in ["manifest_path", "expected_raw_path", "expected_history_sidecar_path", "command"]:
        if not row.get(key):
            raise SystemExit(f"{source} missing {key}")
    manifest = root / row["manifest_path"]
    if not manifest.exists():
        raise SystemExit(f"{source} manifest missing: {manifest}")
    mrows = list(csv.DictReader(manifest.open()))
    if len(mrows) != int(row["manifest_windows"]):
        raise SystemExit(f"{source} manifest window mismatch: {len(mrows)} vs {row['manifest_windows']}")
    command = row["command"]
    for phrase in ["--window-manifest", "--output-slug", "--export-history-sidecar"]:
        if phrase not in command:
            raise SystemExit(f"{source} command missing {phrase}")

doc = doc_path.read_text()
for phrase in [
    "Remote q9/Full-Grid Rerun Execution Pack",
    "Success criterion",
    "Post-Run Validation",
    "build_remote_q9_rerun_completion_audit.py",
    "complete_for_ingestion=1",
    "New raw files alone do not update the locked final-main manifests",
    "python3 scripts/run_oral_rerun_queue.py --priority P0 --family moirai",
    "python3 scripts/run_oral_rerun_queue.py --priority P0 --family timesfm",
    "bash scripts/critic_oral_evidence_gap_matrix.sh",
]:
    if phrase not in doc:
        raise SystemExit(f"Remote execution pack missing phrase: {phrase}")

print("[remote q9 rerun execution pack critic] PASS")
print({"p0_sources": len(plan_rows), "windows": windows, "families": sorted(families)})
PY
