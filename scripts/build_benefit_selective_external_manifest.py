#!/usr/bin/env python3
"""Freeze an outcome-blind, domain-stratified external evaluation manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benefit_selective_drcr_external_protocol.json"
CATALOG_PATH = ROOT / "results" / "failure_mining" / "gift_eval_second_family_failures.csv"
CSV_OUT = ROOT / "results" / "aaai_stress" / "benefit_selective_external_manifest.csv"
DOC_OUT = ROOT / "docs" / "benefit_selective_external_confirmation_protocol.md"


def rank(seed: str, dataset: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}".encode("utf-8")).hexdigest()


def base_dataset(dataset: str) -> str:
    return dataset.split("/", 1)[0]


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sampling = config["domain_sampling"]
    allowed = set(sampling["selection_fields_only"])
    if allowed != {"dataset", "domain"}:
        raise ValueError("External domain selection may use only dataset and domain metadata")
    excluded = set(sampling["exclude_development_datasets"])
    excluded_bases = set(sampling["exclude_development_base_datasets"])

    catalog: dict[str, str] = {}
    with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset, domain = row["dataset"], row["domain"]
            if dataset not in excluded and base_dataset(dataset) not in excluded_bases:
                catalog[dataset] = domain
    by_domain: dict[str, list[str]] = defaultdict(list)
    for dataset, domain in catalog.items():
        by_domain[domain].append(dataset)

    selected: list[tuple[str, str]] = []
    seed = sampling["seed"]
    for domain in sorted(by_domain):
        chosen_bases: set[str] = set()
        for dataset in sorted(by_domain[domain], key=lambda item: rank(seed, item)):
            base = base_dataset(dataset)
            if base in chosen_bases:
                continue
            selected.append((domain, dataset))
            chosen_bases.add(base)
            if len(chosen_bases) == 2:
                break
        if len(chosen_bases) != 2:
            raise ValueError(f"Domain {domain} does not have two distinct base datasets")

    models = config["evaluation"]["models"]
    families = {
        "chronos_bolt_small": "chronos",
        "moirai_1_1_small": "moirai",
        "timesfm_2_5": "timesfm",
    }
    rows: list[dict[str, object]] = []
    for domain, dataset in selected:
        for model in models:
            rows.append({
                "domain": domain,
                "dataset": dataset,
                "family": families[model],
                "model": model,
                "windows_requested": config["evaluation"]["windows_per_model_domain"],
                "quantile_grid": "q10,q20,q30,q40,q50,q60,q70,q80,q90",
                "baseline_selection": config["evaluation"]["classical_reference_selection"],
                "status": "pending_remote_execution",
            })
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Benefit-Selective DRCR External Confirmation Protocol",
        "",
        "Date frozen: 2026-07-11",
        "",
        "## Purpose",
        "",
        "The development result shows strong source/domain routing but only modest extra value over controls that preserve per-source intervention counts. This protocol therefore tests whole-domain transfer with no labels from the external domains. It is the required experiment for upgrading the method from an internal prototype to a paper-facing selective-repair claim.",
        "",
        "## Frozen Method",
        "",
        "- Candidate: `benefit_lcb_leaf20_q40_noleak16`.",
        "- Native is the default action.",
        "- Fixed actions: point-only DRCR and frozen EP1.25/cap1.10 DRCR.",
        "- Fixed 16-feature external contract: `configs/benefit_selective_drcr_external_protocol.json`; realized-target `flatness_score` is excluded.",
        "- No feature, threshold, action, or candidate change is allowed after an external outcome is inspected.",
        "",
        "## Outcome-Blind Domain Sample",
        "",
        "The sampler reads only dataset names and broad domain labels, excludes every development base dataset, SHA256-ranks names with the frozen seed, and selects two different base datasets per broad domain. It does not read benchmark error or failure fields for selection.",
        "",
        "The source catalog contains historical aggregate outcomes. Those columns are forbidden to the sampler, but this is algorithmically outcome-blind domain selection rather than a claim that researchers never had access to catalog outcomes.",
        "",
        "| Broad domain | Dataset 1 | Dataset 2 |",
        "| --- | --- | --- |",
    ]
    selected_by_domain: dict[str, list[str]] = defaultdict(list)
    for domain, dataset in selected:
        selected_by_domain[domain].append(dataset)
    for domain in sorted(selected_by_domain):
        first, second = selected_by_domain[domain]
        lines.append(f"| {domain} | `{first}` | `{second}` |")
    lines.extend([
        "",
        f"This gives **{len(selected)} data domains**, three model families, and up to **{len(rows) * int(config['evaluation']['windows_per_model_domain'])} source-specific windows**. Repeated model evaluations of one target window are paired measurements, not new domains.",
        "",
        "## Baselines",
        "",
        "Native TSFM, deterministic rolling-selected classical reference, always repair, fixed 50/50 blend, matched random gate, history-only HCR gate, Benefit-LCB, and oracle. Every method uses the same q9 grid and forecast cases. The deterministic classical expert is not described as a calibrated probabilistic baseline.",
        "",
        "## Co-Primary Pass/Fail Rules",
        "",
        "1. Overall domain-macro q9-WQL paired 95% CI upper bound is at most zero.",
        "2. Ex-ante low-structure stratum q9-WQL paired 95% CI upper bound is at most zero.",
        "3. Structured-complement q9-WQL harm upper bound is at most 0.002 and coverage-gap harm upper bound is at most 0.02.",
        "4. Structured-complement intervention is at most 25%.",
        "5. Benefit-LCB outperforms always repair, fixed blend, and intervention-matched random gate.",
        "6. Leave-one-domain-out sensitivity does not reverse the effect direction.",
        "",
        "Failure of any co-primary rule blocks a broad safe-repair claim. The paper must then remain a scoped reliability/limits study or redesign the selector before collecting another untouched endpoint.",
        "",
        "## Statistical Unit",
        "",
        "Primary inference is hierarchical and paired: domain, then series, then non-overlapping forecast origin. Domain-level sign-flip and per-domain forest plots accompany the hierarchical bootstrap. Model/configuration sources are not treated as independent domains.",
        "",
        f"Execution manifest: `{CSV_OUT.relative_to(ROOT)}`. Frozen machine-readable protocol: `{CONFIG_PATH.relative_to(ROOT)}`.",
        "",
    ])
    DOC_OUT.write_text("\n".join(lines), encoding="utf-8")
    print({"status": "ok", "domains": len(selected), "jobs": len(rows), "windows": len(rows) * int(config["evaluation"]["windows_per_model_domain"]), "manifest": str(CSV_OUT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
