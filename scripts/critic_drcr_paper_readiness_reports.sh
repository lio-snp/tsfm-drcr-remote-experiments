#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path.cwd()
required = [
    root / "docs" / "ltt_crc_pvalue_report.md",
    root / "docs" / "protected_group_safety_table.md",
    root / "docs" / "denominator_fragility_report.md",
    root / "docs" / "paper_faithful_metric_robustness.md",
    root / "docs" / "no_test_tuning_statement.md",
    root / "docs" / "caption_plain_language.md",
    root / "docs" / "final_drcr_paper_readiness_summary.md",
    root / "docs" / "selector_ablation_baselines.md",
    root / "results" / "aaai_stress" / "ltt_crc_pvalue_table.csv",
    root / "results" / "aaai_stress" / "protected_safety_results.csv",
    root / "results" / "aaai_stress" / "clipped_sensitivity_table.csv",
    root / "results" / "aaai_stress" / "selector_ablation_baselines.csv",
    root / "results" / "aaai_stress" / "split_manifest.csv",
    root / "results" / "aaai_stress" / "dataset_manifest.csv",
    root / "results" / "aaai_stress" / "model_manifest.csv",
    root / "results" / "aaai_stress" / "final_main_results.csv",
]
for path in required:
    if not path.exists():
        raise SystemExit(f"missing paper-readiness artifact: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"empty paper-readiness artifact: {path}")

status = json.loads((root / "results" / "aaai_stress" / "final_main_figure_status.json").read_text())
if status.get("selected_candidate_id") != "drcr_expert_pull_1.25_cap_1.10":
    raise SystemExit("selected candidate drifted")
if int(status.get("n_candidates", 0)) != 14:
    raise SystemExit("objective baselines are not included in final-main status")

with (root / "results" / "aaai_stress" / "ltt_crc_pvalue_table.csv").open(newline="") as handle:
    ltt = list(csv.DictReader(handle))
selected_primary = [
    row for row in ltt
    if row["candidate_id"] == "drcr_expert_pull_1.25_cap_1.10" and row["primary_risk"] == "1"
]
if len(selected_primary) != 3:
    raise SystemExit(f"expected 3 selected primary risk rows, got {len(selected_primary)}")
if any(row["accepted"] != "1" for row in selected_primary):
    raise SystemExit("selected method failed a primary LTT risk row")

with (root / "results" / "aaai_stress" / "protected_safety_results.csv").open(newline="") as handle:
    safety = list(csv.DictReader(handle))
finance = [
    row for row in safety
    if row["candidate_id"] == "drcr_expert_pull_1.25_cap_1.10" and row["group"] == "target_id:finance_fred_stress"
]
if not finance or float(finance[0]["wql_harm_rate"]) != 0.0:
    raise SystemExit("selected finance harm should be zero in protected safety report")

with (root / "results" / "aaai_stress" / "split_manifest.csv").open(newline="") as handle:
    split = list(csv.DictReader(handle))
if len(split) != 1076:
    raise SystemExit(f"split manifest should have one row per window, got {len(split)}")

with (root / "results" / "aaai_stress" / "selector_ablation_baselines.csv").open(newline="") as handle:
    selectors = {row["selector_rule"]: row for row in csv.DictReader(handle)}
for rule in ["no_risk_highest_utility", "wql_only", "harm_only", "undercoverage_only", "absolute_coverage_only", "dual_tri_risk"]:
    if rule not in selectors:
        raise SystemExit(f"missing selector ablation rule: {rule}")
if selectors["dual_tri_risk"]["selected_candidate_id"] != "drcr_expert_pull_1.25_cap_1.10":
    raise SystemExit("dual-tri-risk selector should match current selected method")
if selectors["absolute_coverage_only"]["tri_risk_accepted"] == "1":
    raise SystemExit("absolute coverage-only selector should expose a tri-risk failure")

summary_text = (root / "docs" / "final_drcr_paper_readiness_summary.md").read_text()
for phrase in [
    "Simple score gate",
    "Oracle upper bound",
    "可以安全写的 claim",
    "仍需补强",
    "LTT/CRC-style empirical risk screening",
]:
    if phrase not in summary_text:
        raise SystemExit(f"final summary missing phrase: {phrase}")

print("[critic] PASS: DRCR paper-readiness reports are present and internally consistent")
print("[critic] artifacts:", len(required))
print("[critic] ltt selected primary risks:", selected_primary)
PY
