#!/usr/bin/env python3
"""Execute frozen external jobs serially with resource and artifact checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.system_memory import available_ram_gb, windows_commit_fraction


JOBS = ROOT / "results" / "aaai_stress" / "benefit_selective_external_execution_jobs.csv"
STATUS = ROOT / "results" / "aaai_stress" / "benefit_selective_external_execution_status.csv"
LOG_DIR = ROOT / "results" / "aaai_stress" / "external_logs"
MIN_RAM_GB = {"chronos": 1.5, "moirai": 2.0, "timesfm": 1.56}


def swap_fraction() -> float | None:
    if sys.platform == "win32":
        return windows_commit_fraction()
    if sys.platform != "darwin":
        return None
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"total = ([0-9.]+)M\s+used = ([0-9.]+)M", output)
    if not match or float(match.group(1)) <= 0:
        return None
    return float(match.group(2)) / float(match.group(1))


def load_status_rows() -> dict[str, dict[str, str]]:
    if not STATUS.exists():
        return {}
    with STATUS.open(newline="") as handle:
        return {row["job_id"]: row for row in csv.DictReader(handle)}


def write_status_rows(rows: dict[str, dict[str, object]]) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id", "family", "dataset", "status", "started_at", "finished_at",
        "elapsed_seconds", "exit_code", "available_ram_gb", "swap_fraction",
        "disk_free_gb", "log_path", "detail",
    ]
    tmp = STATUS.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job_id in sorted(rows):
            writer.writerow({field: rows[job_id].get(field, "") for field in fields})
    tmp.replace(STATUS)


def resource_snapshot() -> dict[str, float | None]:
    return {
        "available_ram_gb": round(available_ram_gb(), 3),
        "swap_fraction": swap_fraction(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / (1024**3), 3),
    }


def resource_block(job: dict[str, str], snapshot: dict[str, float | None], ignore_swap: bool) -> str | None:
    if float(snapshot["disk_free_gb"] or 0) < 10.0:
        return "disk_free_below_10_gib"
    swap = snapshot["swap_fraction"]
    if not ignore_swap and swap is not None and swap > 0.85:
        return "swap_use_above_85_percent"
    if float(snapshot["available_ram_gb"] or 0) < MIN_RAM_GB[job["family"]]:
        return f"available_ram_below_{MIN_RAM_GB[job['family']]:.2f}_gib"
    return None


def status_is_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def resolve_data_path(job: dict[str, str]) -> Path | None:
    frequency_path = ROOT / job["data_path"]
    root_path = ROOT / job["data_root_path"]
    for candidate in (frequency_path, root_path):
        if candidate.exists() and (candidate / "state.json").exists():
            return candidate
    return None


def split_job_command(raw_command: str) -> list[str]:
    command = shlex.split(raw_command)
    if sys.platform != "win32" or not command:
        return command
    executable = command[0].replace("\\", "/")
    match = re.fullmatch(r"(?P<venv>\.venv[^/]*)/bin/python(?:3)?", executable)
    if match:
        command[0] = str(ROOT / match.group("venv") / "Scripts" / "python.exe")
    return command


def command_with_data_path(job: dict[str, str], data_path: Path) -> list[str]:
    command = split_job_command(job["command"])
    index = command.index("--data-path") + 1
    command[index] = str(data_path.relative_to(ROOT))
    return command


def run_logged(command: list[str], log_path: Path, timeout: int) -> tuple[int, str]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return completed.returncode, "completed"
        except subprocess.TimeoutExpired:
            return 124, f"timeout_after_{timeout}_seconds"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["chronos", "moirai", "timesfm"])
    parser.add_argument("--job-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-swap-preflight", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    with JOBS.open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if args.family:
        jobs = [job for job in jobs if job["family"] == args.family]
    if args.job_id:
        jobs = [job for job in jobs if job["job_id"] == args.job_id]
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    if not jobs:
        raise SystemExit("No jobs matched the requested filters")

    statuses: dict[str, dict[str, object]] = load_status_rows()
    for job in jobs:
        job_id = job["job_id"]
        output_status = ROOT / job["status_path"]
        if status_is_ok(output_status) and not args.force:
            statuses[job_id] = {
                **job,
                "status": "skipped_existing_ok",
                "detail": str(output_status),
            }
            write_status_rows(statuses)
            continue

        snapshot = resource_snapshot()
        blocked = resource_block(job, snapshot, args.ignore_swap_preflight)
        if args.dry_run or blocked:
            status = "dry_run_ready" if args.dry_run and not blocked else "blocked_resource_preflight"
            statuses[job_id] = {
                **job,
                **snapshot,
                "status": status,
                "detail": blocked or job["command"],
            }
            write_status_rows(statuses)
            print(f"{job_id}: {status}: {blocked or job['command']}")
            if blocked and not args.dry_run:
                break
            continue

        data_path = resolve_data_path(job)
        if data_path is None:
            if not args.download_missing:
                statuses[job_id] = {
                    **job, **snapshot, "status": "blocked_missing_data",
                    "detail": f"{job['data_path']} or {job['data_root_path']}",
                }
                write_status_rows(statuses)
                break
            download_log = LOG_DIR / f"{job_id}_download.log"
            code, detail = run_logged(split_job_command(job["download_command"]), download_log, args.timeout_seconds)
            data_path = resolve_data_path(job)
            if code != 0 or data_path is None:
                statuses[job_id] = {
                    **job, **snapshot, "status": "blocked_download_failed", "exit_code": code,
                    "log_path": str(download_log.relative_to(ROOT)), "detail": detail,
                }
                write_status_rows(statuses)
                break

        started = int(time.time())
        log_path = LOG_DIR / f"{job_id}.log"
        code, detail = run_logged(command_with_data_path(job, data_path), log_path, args.timeout_seconds)
        finished = int(time.time())
        final_status = "ok" if code == 0 and status_is_ok(output_status) else "failed_artifact_contract"
        statuses[job_id] = {
            **job,
            **snapshot,
            "status": final_status,
            "started_at": started,
            "finished_at": finished,
            "elapsed_seconds": finished - started,
            "exit_code": code,
            "log_path": str(log_path.relative_to(ROOT)),
            "detail": detail,
        }
        write_status_rows(statuses)
        print(f"{job_id}: {final_status} ({finished - started}s)")
        if final_status != "ok":
            break


if __name__ == "__main__":
    main()
