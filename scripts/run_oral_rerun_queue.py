#!/usr/bin/env python3
"""Run the oral evidence rerun command queue with resumable status logs."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
MANIFEST = OUT / "oral_rerun_command_manifest.csv"
STATUS_OUT = OUT / "oral_rerun_execution_status.csv"
LOG_DIR = OUT / "oral_rerun_logs"
RAW_DIR = ROOT / "results" / "raw_forecasts"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def option_value(args: list[str], option: str) -> str:
    try:
        index = args.index(option)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return args[index + 1]


def command_args(
    command: str,
    family: str,
    python_override: str | None = None,
    allow_low_memory: bool = False,
) -> list[str]:
    args = shlex.split(command)
    if not args:
        return args
    if args[0] not in {"python", "python3"}:
        return args
    replacement = python_override or default_python(family, args)
    if replacement:
        args[0] = replacement
    if allow_low_memory and "--skip-memory-preflight" not in args:
        script = args[1] if len(args) > 1 else ""
        if script.endswith(
            (
                "run_timesfm_gift_eval_raw.py",
                "run_moirai_gift_eval_raw.py",
                "run_chronos_bolt_gift_eval_raw.py",
            )
        ):
            args.extend(["--skip-memory-preflight", "--min-available-ram-gb", "0.2"])
    return args


def default_python(family: str, args: list[str]) -> str:
    if family == "moirai":
        candidate = ROOT / ".venv-moirai" / "bin" / "python"
    else:
        candidate = ROOT / ".venv-chronos" / "bin" / "python"
    return str(candidate.relative_to(ROOT)) if candidate.exists() else ""


def expected_outputs(command: str) -> tuple[Path | None, Path | None]:
    args = shlex.split(command)
    slug = option_value(args, "--output-slug")
    if not slug:
        return None, None
    return RAW_DIR / f"{slug}.csv", RAW_DIR / f"{slug}_history_context.csv"


def count_rows(path: Path | None) -> int:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def artifact_status(command: str) -> dict[str, object]:
    raw_path, sidecar_path = expected_outputs(command)
    status_path = None
    if raw_path is not None:
        status_path = raw_path.with_name(f"{raw_path.stem}_status.json")
    raw_rows = count_rows(raw_path)
    sidecar_rows = count_rows(sidecar_path)
    return {
        "raw_path": str(raw_path.relative_to(ROOT)) if raw_path else "",
        "history_sidecar_path": str(sidecar_path.relative_to(ROOT)) if sidecar_path else "",
        "runner_status_path": str(status_path.relative_to(ROOT)) if status_path else "",
        "raw_exists": int(bool(raw_path and raw_path.exists())),
        "history_sidecar_exists": int(bool(sidecar_path and sidecar_path.exists())),
        "runner_status_exists": int(bool(status_path and status_path.exists())),
        "raw_rows": raw_rows,
        "history_sidecar_rows": sidecar_rows,
    }


def already_complete(command: str) -> bool:
    status = artifact_status(command)
    return bool(status["raw_exists"] and status["history_sidecar_exists"] and status["raw_rows"] and status["history_sidecar_rows"])


def existing_status() -> dict[str, dict[str, str]]:
    return {row["source"]: row for row in read_csv(STATUS_OUT) if row.get("source")}


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("command_status") == "ready"]
    if args.priority:
        priorities = set(args.priority)
        selected = [row for row in selected if row.get("priority") in priorities]
    if args.family:
        families = set(args.family)
        selected = [row for row in selected if row.get("family") in families]
    if args.source:
        sources = set(args.source)
        selected = [row for row in selected if row.get("source") in sources]
    if args.max_windows is not None:
        selected = [row for row in selected if int(row.get("manifest_windows") or 0) <= args.max_windows]
    return selected[: args.limit] if args.limit else selected


def run_one(
    row: dict[str, str],
    timeout: int,
    dry_run: bool,
    python_override: str | None,
    allow_low_memory: bool,
) -> dict[str, object]:
    source = row["source"]
    command = row["command"]
    args = command_args(command, row.get("family", ""), python_override, allow_low_memory)
    executed_command = " ".join(shlex.quote(arg) for arg in args)
    artifact_before = artifact_status(command)
    if already_complete(command):
        return {
            **row,
            **artifact_before,
            "execution_status": "skipped_existing_artifacts",
            "executed_command": executed_command,
            "returncode": 0,
            "started_at": "",
            "finished_at": "",
            "elapsed_seconds": 0,
            "log_path": "",
        }
    if dry_run:
        return {
            **row,
            **artifact_before,
            "execution_status": "dry_run_ready",
            "executed_command": executed_command,
            "returncode": "",
            "started_at": "",
            "finished_at": "",
            "elapsed_seconds": 0,
            "log_path": "",
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{source}.log"
    started = int(time.time())
    with log_path.open("w") as log_handle:
        log_handle.write(f"$ {executed_command}\n\n")
        log_handle.flush()
        try:
            proc = subprocess.run(
                args,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            returncode = proc.returncode
            execution_status = "ok" if returncode == 0 and already_complete(command) else "failed_or_missing_artifacts"
        except subprocess.TimeoutExpired as exc:
            log_handle.write(f"\nTIMEOUT after {timeout}s: {exc}\n")
            returncode = 124
            execution_status = "timeout"
    finished = int(time.time())
    artifact_after = artifact_status(command)
    return {
        **row,
        **artifact_after,
        "execution_status": execution_status,
        "executed_command": executed_command,
        "returncode": returncode,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": finished - started,
        "log_path": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST.relative_to(ROOT)))
    parser.add_argument("--priority", action="append", choices=["P0", "P1", "P2"])
    parser.add_argument("--family", action="append")
    parser.add_argument("--source", action="append")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--python-override", default=None)
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="Append the runner's explicit low-memory override for tiny local reruns.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    rows = select_rows(read_csv(manifest_path), args)
    prior = existing_status()
    status_rows: list[dict[str, object]] = []
    for row in rows:
        if not args.rerun_completed and prior.get(row["source"], {}).get("execution_status") in {
            "ok",
            "skipped_existing_artifacts",
        }:
            status_rows.append({**row, **prior[row["source"]]})
            continue
        result = run_one(
            row,
            timeout=args.timeout_seconds,
            dry_run=args.dry_run,
            python_override=args.python_override,
            allow_low_memory=args.allow_low_memory,
        )
        status_rows.append(result)

    untouched = [row for source, row in prior.items() if source not in {item["source"] for item in status_rows}]
    combined = untouched + status_rows
    if combined:
        write_csv(STATUS_OUT, combined)

    counts: dict[str, int] = {}
    for row in combined:
        status = str(row.get("execution_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    print({"selected": len(rows), "dry_run": args.dry_run, "status_counts": counts, "status_path": str(STATUS_OUT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
