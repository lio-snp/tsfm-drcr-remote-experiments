#!/usr/bin/env python3
"""Build a machine-readable gap matrix for the remaining AAAI/oral evidence work.

This script does not run model inference.  It audits the current final-main
artifacts and writes the next rerun/exporter requirements in a form that can be
handed to a runner or reviewer.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

SPLIT = OUT / "split_manifest.csv"
DATASET_MANIFEST = OUT / "dataset_manifest.csv"
CLASSICAL_FEASIBILITY = OUT / "classical_probabilistic_baseline_feasibility.csv"

SOURCE_GAPS_OUT = OUT / "oral_evidence_source_gap_matrix.csv"
FAMILY_GAPS_OUT = OUT / "oral_evidence_family_gap_matrix.csv"
DOC_OUT = DOCS / "oral_evidence_gap_matrix.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def priority_for(*, family: str, role: str, evidence_tier: str, total_windows: int) -> str:
    if evidence_tier != "q9_fullgrid" and family in {"moirai", "timesfm"}:
        return "P0"
    if family in {"moirai", "timesfm"} and role in {"positive_control", "stress_target"} and total_windows < 64:
        return "P1"
    if evidence_tier != "q9_fullgrid":
        return "P1"
    return "P2"


def reason_for(*, family: str, role: str, evidence_tier: str, total_windows: int, has_history_context: bool) -> str:
    reasons: list[str] = []
    if evidence_tier != "q9_fullgrid":
        reasons.append("probabilistic result is q3/proxy rather than native q9/full-grid")
    if not has_history_context:
        reasons.append("raw artifacts lack history/context values for native classical interval baselines")
    if family in {"moirai", "timesfm"} and total_windows < 64:
        reasons.append("non-Chronos slice is small relative to the clean Chronos 64-window blocks")
    if role in {"positive_control", "stress_target"} and family in {"moirai", "timesfm"}:
        reasons.append("protected/safety evidence is less balanced outside Chronos")
    return "; ".join(reasons) if reasons else "current source is usable as current-suite evidence"


def recommended_action(*, family: str, source: str, evidence_tier: str, has_history_context: bool) -> str:
    actions: list[str] = []
    if evidence_tier != "q9_fullgrid":
        if family == "timesfm":
            actions.append("rerun TimesFM exporter and persist all quantile columns returned by the continuous quantile head")
        elif family == "moirai":
            actions.append("rerun Moirai2 or Moirai1; Moirai1 now exports sample-derived q10..q90 in the raw runner")
        else:
            actions.append("rerun with --quantile-levels 0.1,0.2,...,0.9")
    if not has_history_context:
        actions.append("export context_values and baseline_context_values per window or as sidecar files before fitting native intervals")
    if not actions:
        actions.append("keep as current-suite evidence; broaden with new domains if compute is available")
    return "; ".join(actions)


def main() -> None:
    split_rows = read_csv(SPLIT)
    dataset_rows = read_csv(DATASET_MANIFEST)
    feasibility_rows = read_csv(CLASSICAL_FEASIBILITY)
    feasibility_by_source = {
        Path(row.get("raw_file", "")).stem: row
        for row in feasibility_rows
        if row.get("raw_file")
    }

    family_by_source: dict[str, str] = {}
    qlevels_by_source: dict[str, set[str]] = defaultdict(set)
    counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        source = row.get("source", "")
        family_by_source[source] = row.get("family", "")
        qlevels_by_source[source].add(row.get("quantile_grid_n_levels", ""))
        counts_by_source[source][row.get("split", "")] += 1

    source_gap_rows: list[dict[str, object]] = []
    sources_by_group: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    history_gap_by_source: dict[str, int] = {}
    for row in dataset_rows:
        source = row.get("source", "")
        family = family_by_source.get(source, "unknown")
        total = int(row.get("total_windows") or 0)
        evidence_tier = row.get("evidence_tier", "")
        role = row.get("role", "")
        q9_ready = evidence_tier == "q9_fullgrid"
        feasibility = feasibility_by_source.get(source, {})
        has_history_context = feasibility.get("has_history_values") == "1" or feasibility.get("has_history_sidecar") == "1"
        row_priority = priority_for(family=family, role=role, evidence_tier=evidence_tier, total_windows=total)
        history_gap = int(not has_history_context)
        history_gap_by_source[source] = history_gap
        sources_by_group[(family, row.get("dataset", ""), role, evidence_tier)].append(source)
        source_gap_rows.append(
            {
                "priority": row_priority,
                "source": source,
                "family": family,
                "dataset": row.get("dataset", ""),
                "role": role,
                "current_evidence_tier": evidence_tier,
                "quantile_grid_levels_seen": ";".join(sorted(qlevels_by_source.get(source, []))),
                "calibration_windows": row.get("calibration_windows", ""),
                "test_windows": row.get("test_windows", ""),
                "total_windows": total,
                "needs_q9_fullgrid_rerun": int(not q9_ready),
                "needs_history_context_export": history_gap,
                "reason": reason_for(
                    family=family,
                    role=role,
                    evidence_tier=evidence_tier,
                    total_windows=total,
                    has_history_context=has_history_context,
                ),
                "recommended_action": recommended_action(
                    family=family,
                    source=source,
                    evidence_tier=evidence_tier,
                    has_history_context=has_history_context,
                ),
            }
        )
    source_gap_rows.sort(key=lambda row: (str(row["priority"]), str(row["family"]), str(row["dataset"]), str(row["source"])))
    write_csv(SOURCE_GAPS_OUT, source_gap_rows)

    grouped: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        key = (row.get("family", ""), row.get("dataset", ""), row.get("role", ""), row.get("evidence_tier", ""))
        grouped[key][row.get("split", "")] += 1
    family_rows: list[dict[str, object]] = []
    for (family, dataset, role, evidence_tier), counts in sorted(grouped.items()):
        total = sum(counts.values())
        group_sources = sources_by_group.get((family, dataset, role, evidence_tier), [])
        group_needs_history = any(history_gap_by_source.get(source, 1) for source in group_sources)
        family_rows.append(
            {
                "family": family,
                "dataset": dataset,
                "role": role,
                "evidence_tier": evidence_tier,
                "calibration_windows": counts.get("calibration", 0),
                "test_windows": counts.get("test", 0),
                "total_windows": total,
                "needs_q9_fullgrid_rerun": int(evidence_tier != "q9_fullgrid"),
                "small_non_chronos_slice": int(family in {"moirai", "timesfm"} and total < 64),
                "needs_history_context_export": int(group_needs_history),
            }
        )
    write_csv(FAMILY_GAPS_OUT, family_rows)

    p0_rows = [row for row in source_gap_rows if row["priority"] == "P0"]
    p1_rows = [row for row in source_gap_rows if row["priority"] == "P1"]
    q9_gap_count = sum(int(row["needs_q9_fullgrid_rerun"]) for row in source_gap_rows)
    history_gap_count = sum(int(row["needs_history_context_export"]) for row in source_gap_rows)
    timesfm_q3 = sum(1 for row in source_gap_rows if row["family"] == "timesfm" and row["needs_q9_fullgrid_rerun"])
    moirai_q3 = sum(1 for row in source_gap_rows if row["family"] == "moirai" and row["needs_q9_fullgrid_rerun"])

    doc = [
        "# Oral Evidence Gap Matrix",
        "",
        "This audit turns the remaining AAAI/oral-scale evidence gaps into a rerun/exporter checklist. It is generated from the current final-main manifests and raw-artifact feasibility audit; it does not add new model inference.",
        "",
        "## Executive Read",
        "",
        f"- Sources audited: `{len(source_gap_rows)}`.",
        f"- Sources needing q9/full-grid rerun: `{q9_gap_count}`.",
        f"- TimesFM q9/full-grid rerun gaps: `{timesfm_q3}`.",
        f"- Moirai q9/full-grid rerun gaps: `{moirai_q3}`.",
        f"- Sources needing history/context export for native classical interval baselines: `{history_gap_count}`.",
        "- Interpretation: current-suite DRCR evidence is strong enough for a bounded paper claim, but oral-level breadth still needs new inference/exporter work.",
        "",
        "## P0 Rerun/Exporter Items",
        "",
        markdown_table(
            p0_rows[:20],
            [
                ("priority", "Priority"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("current_evidence_tier", "Tier"),
                ("total_windows", "Windows"),
                ("reason", "Reason"),
                ("recommended_action", "Action"),
            ],
        )
        if p0_rows
        else "No P0 gaps detected.",
        "",
        "## P1 Items",
        "",
        markdown_table(
            p1_rows[:20],
            [
                ("priority", "Priority"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("role", "Role"),
                ("current_evidence_tier", "Tier"),
                ("total_windows", "Windows"),
                ("reason", "Reason"),
            ],
        )
        if p1_rows
        else "No P1 gaps detected.",
        "",
        "## Exporter Contract For The Next Rerun",
        "",
        "Every new raw forecast file should include either per-row repeated serialized values or a stable sidecar keyed by `(run_id, series_id, window_index)`:",
        "",
        "- Run raw exporters with `--export-history-sidecar`.",
        "- `context_values`: the exact pre-origin values passed to the TSFM.",
        "- `baseline_context_values`: the exact history used by AutoETS/AutoARIMA/Theta or other classical baseline.",
        "- `context_time_index` and `target_time_index` when available.",
        "- `baseline_model_class`, `baseline_season_length`, `baseline_context_cap`, and transformation metadata.",
        "- native or reconstructed classical interval quantiles: `classical_q10..classical_q90` when fitted.",
        "- full TSFM quantile grid whenever the model API exposes it: `forecast_q10, q20, ..., q90`.",
        "",
        "For sources without sidecars, native classical interval baselines cannot be fitted fairly from the current artifacts, even though point baselines and residual-calibrated classical intervals are available. For sidecar-backed sources, the next step is fitting and auditing native classical interval baselines.",
        "",
        "## Generated Artifacts",
        "",
        f"- `{SOURCE_GAPS_OUT.relative_to(ROOT)}`",
        f"- `{FAMILY_GAPS_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(doc))
    print(
        {
            "status": "ok",
            "sources": len(source_gap_rows),
            "q9_gap_count": q9_gap_count,
            "history_gap_count": history_gap_count,
            "timesfm_q3_gaps": timesfm_q3,
            "moirai_q3_gaps": moirai_q3,
        }
    )


if __name__ == "__main__":
    main()
