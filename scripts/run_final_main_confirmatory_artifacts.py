#!/usr/bin/env python
"""Run a frozen-candidate confirmatory DRCR final-main artifact pass.

This wrapper intentionally reuses the final-main candidate set, metrics,
calibration screen, and plotting code, but swaps the calibration/test split to
an untouched salted hash split and writes to separate confirmatory artifacts.
It should not overwrite the paper-facing exploratory/current-suite main figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_final_main_figure_artifacts as final_main

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "aaai_stress"
FIG_DIR = ROOT / "figures" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "final_main_confirmatory_results_report.md"
WINDOW_OUT = OUT_DIR / "final_main_confirmatory_windows.csv"
SUMMARY_OUT = OUT_DIR / "final_main_confirmatory_summary.csv"
CANDIDATE_OUT = OUT_DIR / "final_main_confirmatory_candidates.csv"
CALIBRATION_OUT = OUT_DIR / "final_main_confirmatory_calibration_tests.csv"
STATUS_OUT = OUT_DIR / "final_main_confirmatory_status.json"
FIGURE_OUT = FIG_DIR / "final_main_confirmatory_draft.png"

SPLIT_PROTOCOL = "confirmatory_salted_hash"
SPLIT_ID = "confirmatory_v1_2026_07_08"


def confirmatory_split_bucket(window: dict[str, object]) -> int:
    key = (
        f"{SPLIT_ID}|{window['source']}|{window['series_id']}|"
        f"{window['window_index']}"
    ).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % 2


def main() -> None:
    final_main.DOC_PATH = DOC_PATH
    final_main.WINDOW_OUT = WINDOW_OUT
    final_main.SUMMARY_OUT = SUMMARY_OUT
    final_main.CANDIDATE_OUT = CANDIDATE_OUT
    final_main.CALIBRATION_OUT = CALIBRATION_OUT
    final_main.STATUS_OUT = STATUS_OUT
    final_main.FIGURE_OUT = FIGURE_OUT
    final_main.smooth.crc.split_bucket = confirmatory_split_bucket

    final_main.main()

    status = json.loads(STATUS_OUT.read_text())
    status.update(
        {
            "split_protocol": SPLIT_PROTOCOL,
            "split_id": SPLIT_ID,
            "confirmatory": True,
            "confirmatory_note": (
                "Uses the frozen final-main candidate/risk machinery with an "
                "untouched salted hash split and separate output paths."
            ),
        }
    )
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")

    report = DOC_PATH.read_text()
    insert = (
        "\n## Confirmatory Split\n\n"
        f"- Split protocol: `{SPLIT_PROTOCOL}`.\n"
        f"- Split id: `{SPLIT_ID}`.\n"
        "- Candidate set, risk thresholds, metrics, bootstrap, and plotting code "
        "are inherited from `scripts/run_final_main_figure_artifacts.py`.\n"
        "- This run is confirmatory relative to the current final-main suite; it "
        "does not by itself add new model inference or new benchmark domains.\n"
    )
    DOC_PATH.write_text(report + insert)


if __name__ == "__main__":
    main()
