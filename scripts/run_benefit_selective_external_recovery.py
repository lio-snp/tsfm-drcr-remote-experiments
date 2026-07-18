#!/usr/bin/env python3
"""Rerun the frozen invalid/incomplete jobs serially and continue after failures."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "aaai_stress"
MANIFEST = RESULTS / "benefit_selective_external_recovery_jobs.csv"
STATUS_OUT = RESULTS / "benefit_selective_external_recovery_status.csv"
QUEUE = ROOT / "scripts" / "run_benefit_selective_external_queue.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def artifact_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except (json.JSONDecodeError, OSError):
        return "invalid_json"


def write_status(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with STATUS_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["chronos", "moirai", "timesfm"])
    parser.add_argument("--category")
    parser.add_argument("--job-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-swap-preflight", action="store_true")
    args = parser.parse_args()

    jobs = read_csv(MANIFEST)
    if args.family:
        jobs = [row for row in jobs if row["family"] == args.family]
    if args.category:
        jobs = [row for row in jobs if row["category"] == args.category]
    if args.job_id:
        jobs = [row for row in jobs if row["job_id"] == args.job_id]
    if args.limit is not None:
        jobs = jobs[: max(args.limit, 0)]
    if not jobs:
        raise SystemExit("No recovery jobs matched the requested filters")

    statuses: list[dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        command = [
            sys.executable,
            str(QUEUE),
            "--job-id",
            job["job_id"],
            "--download-missing",
            "--force",
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.ignore_swap_preflight:
            command.append("--ignore-swap-preflight")
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if args.dry_run:
            statuses.append(
                {
                    "job_id": job["job_id"],
                    "family": job["family"],
                    "category": job["category"],
                    "queue_exit_code": "",
                    "artifact_status": "planned",
                    "status_path": job["status_path"],
                }
            )
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        status_path = ROOT / job["status_path"]
        status = artifact_status(status_path)
        statuses.append(
            {
                "job_id": job["job_id"],
                "family": job["family"],
                "category": job["category"],
                "queue_exit_code": completed.returncode,
                "artifact_status": status,
                "status_path": job["status_path"],
            }
        )
        write_status(statuses)

    ok = sum(row["artifact_status"] == "ok" for row in statuses)
    planned = sum(row["artifact_status"] == "planned" for row in statuses)
    print(
        {
            "attempted": len(statuses),
            "planned": planned,
            "artifact_status_ok": ok,
            "status": str(STATUS_OUT),
        }
    )
    return 0 if args.dry_run or ok == len(statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
