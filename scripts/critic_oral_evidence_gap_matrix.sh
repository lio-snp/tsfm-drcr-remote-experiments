#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
source_path = root / "results/aaai_stress/oral_evidence_source_gap_matrix.csv"
family_path = root / "results/aaai_stress/oral_evidence_family_gap_matrix.csv"
doc_path = root / "docs/oral_evidence_gap_matrix.md"

for path in [source_path, family_path, doc_path]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty required artifact: {path}")

source_rows = list(csv.DictReader(source_path.open()))
family_rows = list(csv.DictReader(family_path.open()))
if len(source_rows) < 20:
    raise SystemExit(f"Too few audited sources: {len(source_rows)}")
if len(family_rows) < 8:
    raise SystemExit(f"Too few family/dataset groups: {len(family_rows)}")

q9_gap_count = sum(row.get("needs_q9_fullgrid_rerun") == "1" for row in source_rows)
history_gap_count = sum(row.get("needs_history_context_export") == "1" for row in source_rows)
timesfm_q3 = sum(
    row.get("family") == "timesfm" and row.get("needs_q9_fullgrid_rerun") == "1"
    for row in source_rows
)
moirai_q3 = sum(
    row.get("family") == "moirai" and row.get("needs_q9_fullgrid_rerun") == "1"
    for row in source_rows
)
p0_count = sum(row.get("priority") == "P0" for row in source_rows)

if q9_gap_count < 10:
    raise SystemExit(f"q9/full-grid gap count unexpectedly low: {q9_gap_count}")
if history_gap_count != 0:
    raise SystemExit(f"history/context sidecar recovery should leave no source-level export gaps, got {history_gap_count}")
if timesfm_q3 < 5:
    raise SystemExit(f"TimesFM q3/full-grid gap count unexpectedly low: {timesfm_q3}")
if moirai_q3 < 5:
    raise SystemExit(f"Moirai q3/full-grid gap count unexpectedly low: {moirai_q3}")
if p0_count < 10:
    raise SystemExit(f"P0 rerun queue too small to reflect current oral gaps: {p0_count}")

doc = doc_path.read_text()
required_phrases = [
    "Sources needing q9/full-grid rerun",
    "TimesFM q9/full-grid rerun gaps",
    "Moirai q9/full-grid rerun gaps",
    "Exporter Contract For The Next Rerun",
    "context_values",
    "baseline_context_values",
    "For sidecar-backed sources, the next step is fitting and auditing native classical interval baselines",
]
for phrase in required_phrases:
    if phrase not in doc:
        raise SystemExit(f"Gap matrix report missing phrase: {phrase}")

print("[oral evidence gap critic] PASS")
print(
    {
        "sources": len(source_rows),
        "q9_gap_count": q9_gap_count,
        "history_gap_count": history_gap_count,
        "timesfm_q3_gaps": timesfm_q3,
        "moirai_q3_gaps": moirai_q3,
        "p0_count": p0_count,
    }
)
PY
