#!/usr/bin/env python3
"""Generate concrete rerun commands for the oral evidence gap queue."""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"
RAW_DIR = ROOT / "results" / "raw_forecasts"

SOURCE_GAPS = OUT / "oral_evidence_source_gap_matrix.csv"
SPLIT = OUT / "split_manifest.csv"
COMMAND_OUT = OUT / "oral_rerun_command_manifest.csv"
MANIFEST_DIR = OUT / "rerun_manifests"
DOC_OUT = DOCS / "oral_rerun_command_manifest.md"

Q9_LEVELS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
MOIRAI1_MODEL_IDS = {
    "small": "Salesforce/moirai-1.1-R-small",
    "base": "Salesforce/moirai-1.1-R-base",
    "large": "Salesforce/moirai-1.1-R-large",
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def clean_dataset_name(dataset: str) -> str:
    return dataset.split("/")[0].lower()


def term_from_dataset(dataset: str) -> str:
    parts = dataset.split("/")
    return parts[-1] if parts else "short"


def resolve_data_path(dataset: str) -> str:
    parts = dataset.split("/")
    if not parts:
        return ""
    base = parts[0]
    freq = parts[1] if len(parts) >= 3 else ""
    candidates = []
    for base_variant in [base, base.lower(), base.upper()]:
        if freq:
            candidates.append(ROOT / "data" / "gift-eval" / base_variant / freq)
        candidates.append(ROOT / "data" / "gift-eval" / base_variant)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(ROOT))
    return str((Path("data/gift-eval") / base).as_posix())


def first_raw_row(source: str) -> dict[str, str]:
    path = RAW_DIR / f"{source}.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return next(reader, {})


def status_json(source: str) -> dict[str, object]:
    path = RAW_DIR / f"{source}_status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def source_manifest(source: str, split_rows: list[dict[str, str]]) -> tuple[Path, int]:
    rows = [
        {
            "source": row["source"],
            "dataset": row["dataset"],
            "dataset_name": row["dataset"].split("/")[0],
            "term": row["dataset"].split("/")[-1],
            "series_id": row["series_id"],
            "window_index": row["window_index"],
            "split": row["split"],
        }
        for row in split_rows
        if row.get("source") == source
    ]
    if not rows:
        raise ValueError(f"no split rows for source {source}")
    path = MANIFEST_DIR / f"{source}_windows.csv"
    write_csv(path, rows)
    return path, len(rows)


def infer_moirai_family(model_id: str, source: str) -> str:
    if "moirai-1.1" in model_id or source.startswith("moirai_1_1"):
        return "moirai1"
    return "moirai2"


def command_for(row: dict[str, str], manifest_path: Path, n_windows: int) -> tuple[str, str, str]:
    source = row["source"]
    family = row["family"]
    dataset = row["dataset"]
    raw = first_raw_row(source)
    status = status_json(source)
    if not raw:
        return "", "blocked_missing_raw_metadata", "Raw source CSV is missing; cannot infer model/baseline args."
    if "FRED finance stress" in dataset:
        args = [
            "python3",
            "scripts/run_finance_fred_stress.py",
            "--backend",
            str(status.get("backend") or "timesfm"),
            "--model-id",
            str(raw.get("model_id") or status.get("model_id") or ""),
            "--model-name",
            str(raw.get("model") or source),
            "--start-date",
            str(status.get("start_date") or "2016-01-01"),
            "--end-date",
            str(status.get("end_date") or ""),
            "--context-length",
            str(raw.get("context_length") or status.get("context_length") or 256),
            "--horizon",
            str(raw.get("horizon") or status.get("horizon") or 10),
            "--window-manifest",
            str(manifest_path.relative_to(ROOT)),
            "--output-slug",
            f"{source}_oral_sidecar_rerun",
            "--export-history-sidecar",
        ]
        return " ".join(shlex.quote(str(item)) for item in args), "ready", f"finance source-specific manifest has {n_windows} windows"

    dataset_name = clean_dataset_name(dataset)
    term = term_from_dataset(dataset)
    output_slug = f"{source}_oral_sidecar_rerun"
    model_name = raw.get("model") or source
    model_id = raw.get("model_id") or status.get("model_id") or ""
    baseline_mode = raw.get("baseline_mode") or "best_simple"
    context_cap = raw.get("context_length") or status.get("context_cap") or ""
    data_path = resolve_data_path(dataset)

    if family == "timesfm":
        script = "scripts/run_timesfm_gift_eval_raw.py"
        args = [
            "python3",
            script,
            "--dataset-name",
            dataset_name,
            "--data-path",
            data_path,
            "--term",
            term,
            "--model-id",
            str(model_id),
            "--model-name",
            model_name,
            "--baseline-mode",
            baseline_mode,
            "--window-manifest",
            str(manifest_path.relative_to(ROOT)),
            "--output-slug",
            output_slug,
            "--export-history-sidecar",
        ]
        if context_cap:
            args.extend(["--context-cap", str(context_cap)])
    elif family == "moirai":
        script = "scripts/run_moirai_gift_eval_raw.py"
        model_family = infer_moirai_family(str(model_id), source)
        if not model_id and source.startswith("moirai_1_1"):
            for key, value in MOIRAI1_MODEL_IDS.items():
                if f"_{key}_" in source:
                    model_id = value
                    break
        args = [
            "python3",
            script,
            "--dataset-name",
            dataset_name,
            "--data-path",
            data_path,
            "--term",
            term,
            "--model-id",
            str(model_id),
            "--model-name",
            model_name,
            "--model-family",
            model_family,
            "--baseline-mode",
            baseline_mode,
            "--window-manifest",
            str(manifest_path.relative_to(ROOT)),
            "--output-slug",
            output_slug,
            "--export-history-sidecar",
        ]
        if context_cap:
            args.extend(["--context-cap", str(context_cap)])
        if model_family == "moirai1":
            args.extend(["--num-samples", str(status.get("num_samples") or 100)])
    elif family == "chronos":
        script = "scripts/run_chronos_bolt_gift_eval_raw.py"
        args = [
            "python3",
            script,
            "--dataset-name",
            dataset_name,
            "--data-path",
            data_path,
            "--term",
            term,
            "--model-id",
            str(model_id),
            "--model-name",
            model_name,
            "--baseline-mode",
            baseline_mode,
            "--quantile-levels",
            Q9_LEVELS,
            "--window-manifest",
            str(manifest_path.relative_to(ROOT)),
            "--output-slug",
            output_slug,
            "--export-history-sidecar",
        ]
        if context_cap:
            args.extend(["--context-cap", str(context_cap)])
    else:
        return "", "blocked_unknown_family", f"Unknown model family: {family}"

    baseline_context = raw.get("baseline_context_length")
    baseline_season = raw.get("baseline_season_length")
    if baseline_context:
        args.extend(["--baseline-context-cap", baseline_context])
    if baseline_season:
        args.extend(["--baseline-season-length", baseline_season])
    return " ".join(shlex.quote(str(item)) for item in args), "ready", f"source-specific manifest has {n_windows} windows"


def main() -> None:
    gap_rows = read_csv(SOURCE_GAPS)
    split_rows = read_csv(SPLIT)
    command_rows: list[dict[str, object]] = []
    for gap in gap_rows:
        source = gap["source"]
        manifest_path, n_windows = source_manifest(source, split_rows)
        command, status, note = command_for(gap, manifest_path, n_windows)
        command_rows.append(
            {
                "priority": gap["priority"],
                "source": source,
                "family": gap["family"],
                "dataset": gap["dataset"],
                "role": gap["role"],
                "current_evidence_tier": gap["current_evidence_tier"],
                "needs_q9_fullgrid_rerun": gap["needs_q9_fullgrid_rerun"],
                "needs_history_context_export": gap["needs_history_context_export"],
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                "manifest_windows": n_windows,
                "command_status": status,
                "command": command,
                "note": note,
            }
        )
    write_csv(COMMAND_OUT, command_rows)

    ready = [row for row in command_rows if row["command_status"] == "ready"]
    blocked = [row for row in command_rows if row["command_status"] != "ready"]
    p0_ready = [row for row in ready if row["priority"] == "P0"]
    doc = [
        "# Oral Rerun Command Manifest",
        "",
        "This file turns the oral evidence gap matrix into source-specific rerun commands. Commands use source-specific window manifests so a rerun does not accidentally mix windows from another model/source on the same dataset.",
        "",
        "## Summary",
        "",
        f"- Total source commands: `{len(command_rows)}`.",
        f"- Ready commands: `{len(ready)}`.",
        f"- P0 ready commands: `{len(p0_ready)}`.",
        f"- Blocked commands: `{len(blocked)}`.",
        "- All ready commands include `--export-history-sidecar`.",
        "",
        "## Ready P0 Commands",
        "",
        markdown_table(
            p0_ready[:20],
            [
                ("priority", "Priority"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("source", "Source"),
                ("manifest_windows", "Windows"),
                ("command", "Command"),
            ],
        )
        if p0_ready
        else "No ready P0 commands.",
        "",
        "## Blocked Items",
        "",
        markdown_table(
            blocked,
            [
                ("priority", "Priority"),
                ("family", "Family"),
                ("dataset", "Dataset"),
                ("source", "Source"),
                ("command_status", "Status"),
                ("note", "Note"),
            ],
        )
        if blocked
        else "No blocked commands.",
        "",
        "## Generated Artifacts",
        "",
        f"- `{COMMAND_OUT.relative_to(ROOT)}`",
        f"- `{MANIFEST_DIR.relative_to(ROOT)}/`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(doc))
    print(
        {
            "status": "ok",
            "commands": len(command_rows),
            "ready": len(ready),
            "p0_ready": len(p0_ready),
            "blocked": len(blocked),
        }
    )


if __name__ == "__main__":
    main()
