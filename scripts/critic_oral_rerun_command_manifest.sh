#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
command_path = root / "results/aaai_stress/oral_rerun_command_manifest.csv"
doc_path = root / "docs/oral_rerun_command_manifest.md"
runner_paths = [
    root / "scripts/run_chronos_bolt_gift_eval_raw.py",
    root / "scripts/run_moirai_gift_eval_raw.py",
    root / "scripts/run_timesfm_gift_eval_raw.py",
]

for path in [command_path, doc_path, *runner_paths]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty required artifact: {path}")

rows = list(csv.DictReader(command_path.open()))
if len(rows) < 20:
    raise SystemExit(f"Too few rerun commands: {len(rows)}")

ready = [row for row in rows if row.get("command_status") == "ready"]
p0_ready = [row for row in ready if row.get("priority") == "P0"]
blocked = [row for row in rows if row.get("command_status") != "ready"]
if len(ready) != len(rows):
    raise SystemExit(f"Expected all source reruns to be ready, got ready={len(ready)} total={len(rows)}")
if len(p0_ready) < 17:
    raise SystemExit(f"Too few ready P0 commands: {len(p0_ready)}")
if blocked:
    raise SystemExit(f"Unexpected blocked commands: {[row.get('source') for row in blocked]}")

for row in ready:
    command = row.get("command", "")
    for required in ["--window-manifest", "--output-slug", "--export-history-sidecar"]:
        if required not in command:
            raise SystemExit(f"Ready command for {row.get('source')} missing {required}: {command}")
    manifest = root / row["manifest_path"]
    if not manifest.exists():
        raise SystemExit(f"Missing source-specific manifest: {manifest}")
    manifest_rows = list(csv.DictReader(manifest.open()))
    if len(manifest_rows) != int(row["manifest_windows"]):
        raise SystemExit(
            f"Manifest row count mismatch for {row.get('source')}: "
            f"{len(manifest_rows)} vs {row['manifest_windows']}"
        )

moirai_runner = (root / "scripts/run_moirai_gift_eval_raw.py").read_text()
for phrase in ["sample_quantile_grid", "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]"]:
    if phrase not in moirai_runner:
        raise SystemExit(f"Moirai runner missing q9 sample export support: {phrase}")

timesfm_runner = (root / "scripts/run_timesfm_gift_eval_raw.py").read_text()
for phrase in ["--window-manifest", "--output-slug", "load_window_manifest"]:
    if phrase not in timesfm_runner:
        raise SystemExit(f"TimesFM runner missing exact-rerun support: {phrase}")

finance_runner = (root / "scripts/run_finance_fred_stress.py").read_text()
for phrase in ["--window-manifest", "--output-slug", "--export-history-sidecar", "history_context_sidecar_row"]:
    if phrase not in finance_runner:
        raise SystemExit(f"Finance runner missing exact-rerun/sidecar support: {phrase}")

doc = doc_path.read_text()
for phrase in ["Ready commands", "P0 ready commands", "Blocked commands", "--export-history-sidecar"]:
    if phrase not in doc:
        raise SystemExit(f"Rerun command report missing phrase: {phrase}")

print("[oral rerun command critic] PASS")
print({"commands": len(rows), "ready": len(ready), "p0_ready": len(p0_ready), "blocked": len(blocked)})
PY
