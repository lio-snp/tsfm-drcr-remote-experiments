#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

root = Path.cwd()
config = json.loads((root / "configs" / "benefit_selective_drcr_external_protocol.json").read_text())
rows = list(csv.DictReader((root / "results" / "aaai_stress" / "benefit_selective_external_manifest.csv").open()))
if len(rows) != 42:
    raise SystemExit(f"[external-protocol critic] FAIL: expected 42 model-domain jobs, found {len(rows)}")
domains = {(row["domain"], row["dataset"]) for row in rows}
if len(domains) != 14:
    raise SystemExit("[external-protocol critic] FAIL: expected 14 independent dataset variants")
counts = Counter(row["domain"] for _, row in {(row["dataset"], row["domain"]): row for row in rows}.items())
if set(counts.values()) != {2}:
    raise SystemExit(f"[external-protocol critic] FAIL: domain stratification changed: {counts}")
if set(config["domain_sampling"]["selection_fields_only"]) != {"dataset", "domain"}:
    raise SystemExit("[external-protocol critic] FAIL: outcome field allowed in domain selection")
excluded_bases = set(config["domain_sampling"]["exclude_development_base_datasets"])
selected_bases = {row["dataset"].split("/", 1)[0] for row in rows}
if excluded_bases & selected_bases:
    raise SystemExit(f"[external-protocol critic] FAIL: development base leaked into external manifest: {excluded_bases & selected_bases}")
doc = (root / "docs" / "benefit_selective_external_confirmation_protocol.md").read_text()
for phrase in ["whole-domain transfer", "no labels from the external domains", "Failure of any co-primary rule blocks", "Model/configuration sources are not treated as independent domains", "algorithmically outcome-blind"]:
    if phrase not in doc:
        raise SystemExit(f"[external-protocol critic] FAIL: missing protocol guard: {phrase}")
print("[external-protocol critic] PASS: frozen whole-domain protocol is outcome-blind and fail-closed")
PY
