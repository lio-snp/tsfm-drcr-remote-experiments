#!/usr/bin/env python3
"""Build the frozen-job recovery queue from the lead-side reconciliation."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "aaai_stress"
RECONCILIATION = RESULTS / "benefit_selective_external_endpoint_reconciliation.csv"
JOBS = RESULTS / "benefit_selective_external_execution_jobs.csv"
OUT = RESULTS / "benefit_selective_external_recovery_jobs.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    reconciliation = read_csv(RECONCILIATION)
    jobs = {row["job_id"]: row for row in read_csv(JOBS)}
    if len(reconciliation) != 42 or len(jobs) != 42:
        raise SystemExit(
            f"Frozen inventory mismatch: reconciliation={len(reconciliation)}, jobs={len(jobs)}"
        )

    recovery_rows: list[dict[str, object]] = []
    for row in reconciliation:
        if row["outcome_valid"] == "1":
            continue
        job_id = row["job_id"]
        job = jobs[job_id]
        recovery_rows.append(
            {
                "job_id": job_id,
                "family": row["family"],
                "domain": row["domain"],
                "dataset": row["dataset"],
                "category": row["category"],
                "runner_status": row["runner_status"],
                "windows_run": row["windows_run"],
                "windows_requested": row["windows_requested"],
                "force_required": 1,
                "timeout_seconds": 28800,
                "status_path": job["status_path"],
                "recovery_command": (
                    "python scripts/run_benefit_selective_external_queue.py "
                    f"--job-id {job_id} --download-missing --force "
                    "--timeout-seconds 28800"
                ),
            }
        )

    if len(recovery_rows) != 33:
        raise SystemExit(f"Expected 33 recovery jobs, found {len(recovery_rows)}")

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(recovery_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(recovery_rows)

    by_family: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in recovery_rows:
        family = str(row["family"])
        category = str(row["category"])
        by_family[family] = by_family.get(family, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
    print({"jobs": len(recovery_rows), "families": by_family, "categories": by_category})


if __name__ == "__main__":
    main()
