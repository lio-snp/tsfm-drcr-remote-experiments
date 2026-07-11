#!/usr/bin/env python3
"""Write the immutable file-hash contract for external DRCR confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress" / "benefit_selective_external_freeze_hashes.json"
REPLACEMENTS = ROOT / "results" / "aaai_stress" / "remote_q9_final_main_replacements.csv"
CORE_FILES = [
    "configs/benefit_selective_drcr_external_protocol.json",
    "results/aaai_stress/benefit_selective_external_manifest.csv",
    "results/aaai_stress/benefit_selective_external_execution_jobs.csv",
    "results/aaai_stress/final_main_figure_windows.csv",
    "results/aaai_stress/remote_q9_final_main_replacements.csv",
    "src/low_snr_tsfm/baselines.py",
    "src/low_snr_tsfm/benefit_selective.py",
    "scripts/evaluate_benefit_selective_external.py",
    "scripts/run_chronos_bolt_gift_eval_raw.py",
    "scripts/run_moirai_gift_eval_raw.py",
    "scripts/run_timesfm_gift_eval_raw.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = list(CORE_FILES)
    with REPLACEMENTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ready_for_final_main_refresh"] == "1":
                files.append(row["feature_path"])
    unique = sorted(set(files))
    missing = [relative for relative in unique if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing files: {missing}")
    protocol = json.loads((ROOT / CORE_FILES[0]).read_text(encoding="utf-8"))
    payload = {
        "status": "frozen_before_external_outcomes",
        "protocol_id": protocol["protocol_id"],
        "external_candidate_id": protocol["method"]["external_candidate_id"],
        "n_files": len(unique),
        "sha256": {relative: sha256(ROOT / relative) for relative in unique},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "n_files": payload["n_files"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
