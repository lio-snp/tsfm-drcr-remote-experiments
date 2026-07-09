#!/usr/bin/env python3
"""Reconstruct history/context sidecars from existing raw forecast artifacts.

This is a no-model-inference recovery path.  Raw forecast CSVs already contain
the dataset, series id, origin, horizon, context length, model id, and baseline
metadata.  For native classical interval baselines we only need the exact
history/target values, which can be reconstructed from the local GIFT-Eval/FRED
data for deterministic rolling windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from low_snr_tsfm.forecast_export import history_context_sidecar_row
from low_snr_tsfm.gift_eval_windowing import forward_fill_nan, iter_univariate_targets

RAW_DIR = ROOT / "results" / "raw_forecasts"
METRIC_DIR = ROOT / "results" / "window_metrics"
OUT = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

SOURCE_GAPS = OUT / "oral_evidence_source_gap_matrix.csv"
STATUS_OUT = OUT / "history_sidecar_reconstruction_status.csv"
DOC_OUT = DOCS / "history_sidecar_reconstruction_report.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def repo_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def resolve_gift_path(dataset: str) -> Path:
    parts = dataset.split("/")
    base = parts[0]
    freq = parts[1] if len(parts) >= 3 else ""
    candidates: list[Path] = []
    for base_variant in [base, base.lower(), base.upper()]:
        if freq:
            candidates.append(ROOT / "data" / "gift-eval" / base_variant / freq)
        candidates.append(ROOT / "data" / "gift-eval" / base_variant)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / "data" / "gift-eval" / base


def metric_lookup(source: str) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(METRIC_DIR / f"{source}_metrics.csv")
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        try:
            key = (str(row["series_id"]), int(row["window_index"]))
        except (KeyError, ValueError):
            continue
        lookup[key] = row
    return lookup


def grouped_windows(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        try:
            key = (str(row["series_id"]), int(row["window_index"]))
        except (KeyError, ValueError):
            continue
        grouped[key].append(row)
    windows = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row.get("horizon_index") or 0))
        first = rows[0].copy()
        first["_actual_values"] = json.dumps([float(row["actual"]) for row in rows])
        windows.append(first)
    return windows


def int_field(*values: object, default: int) -> int:
    for value in values:
        try:
            parsed = int(float(str(value)))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return default


def gift_series_map(dataset: str) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    data_path = resolve_gift_path(dataset)
    series: dict[str, np.ndarray] = {}
    arrow_files = sorted(data_path.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files found in {data_path}")
    item_idx = 0
    for arrow_file in arrow_files:
        with pa.memory_map(str(arrow_file), "r") as source:
            table = ipc.open_stream(source).read_all()
        for item in table.to_pylist():
            item_idx += 1
            item_id = str(item.get("item_id", f"item_{item_idx}"))
            for series_id, values in iter_univariate_targets(item["target"], item_id):
                series[series_id] = np.asarray(values, dtype=float)
    return series


def finance_series_map() -> dict[str, np.ndarray]:
    from run_finance_fred_stress import DATA_DIR, DEFAULT_SPECS, read_fred_values, transform_series

    series: dict[str, np.ndarray] = {}
    for spec in DEFAULT_SPECS:
        path = DATA_DIR / f"{spec.series_id}.csv"
        if not path.exists():
            continue
        dates, values = read_fred_values(path, spec.series_id, "1900-01-01", "2999-12-31")
        _, transformed = transform_series(dates, values, spec.target)
        series[spec.label] = np.asarray(transformed, dtype=float)
    return series


def is_finance(dataset: str) -> bool:
    return "FRED finance stress" in dataset


def source_commit_for(dataset: str) -> str:
    if is_finance(dataset):
        return "fred-public-csv"
    return f"gift-eval:{repo_commit(ROOT / 'external' / 'gift-eval')}"


def build_for_source(source: str, *, overwrite: bool) -> dict[str, object]:
    raw_path = RAW_DIR / f"{source}.csv"
    sidecar_path = RAW_DIR / f"{source}_history_context.csv"
    if not raw_path.exists():
        return {"source": source, "status": "missing_raw", "sidecar": str(sidecar_path.relative_to(ROOT)), "rows": 0}
    if sidecar_path.exists() and not overwrite:
        return {
            "source": source,
            "status": "skipped_existing",
            "sidecar": str(sidecar_path.relative_to(ROOT)),
            "rows": len(read_csv(sidecar_path)),
        }

    raw_rows = read_csv(raw_path)
    if not raw_rows:
        return {"source": source, "status": "empty_raw", "sidecar": str(sidecar_path.relative_to(ROOT)), "rows": 0}
    windows = grouped_windows(raw_rows)
    dataset = windows[0].get("dataset", "")
    series_values = finance_series_map() if is_finance(dataset) else gift_series_map(dataset)
    metrics = metric_lookup(source)
    run_id = f"{source}_reconstructed_sidecar_{int(time.time())}"
    sidecar_rows: list[dict[str, object]] = []
    missing_series = 0
    mismatched_actual = 0

    for window in windows:
        series_id = str(window["series_id"])
        values = series_values.get(series_id)
        if values is None:
            missing_series += 1
            continue
        origin = int(float(window["origin"]))
        horizon = int_field(window.get("horizon"), default=len(json.loads(window["_actual_values"])))
        full_context = forward_fill_nan(values[:origin])
        context_length = int_field(window.get("context_length"), default=int(full_context.size))
        metric = metrics.get((series_id, int(float(window["window_index"]))), {})
        baseline_context_length = int_field(
            window.get("baseline_context_length"),
            metric.get("baseline_context_length"),
            default=context_length,
        )
        baseline_season_length = int_field(
            window.get("baseline_season_length"),
            metric.get("baseline_season_length"),
            default=1,
        )
        target = np.asarray(values[origin : origin + horizon], dtype=float)
        actual = np.asarray(json.loads(window["_actual_values"]), dtype=float)
        if target.shape == actual.shape and not np.allclose(target, actual, equal_nan=True):
            mismatched_actual += 1
        context = full_context[-context_length:]
        baseline_context = full_context[-baseline_context_length:]
        sidecar_rows.append(
            history_context_sidecar_row(
                run_id=run_id,
                dataset=dataset,
                series_id=series_id,
                model=window.get("model", source),
                baseline_family=window.get("baseline_family") or metric.get("baseline") or "",
                baseline_mode=window.get("baseline_mode") or metric.get("baseline_mode") or "",
                origin=origin,
                window_index=int(float(window["window_index"])),
                context=context,
                full_context=full_context,
                baseline_context=baseline_context,
                target=target,
                baseline_season_length=baseline_season_length,
                baseline_context_cap=baseline_context_length,
                source_commit=source_commit_for(dataset),
                model_id=window.get("model_id", ""),
            )
        )

    if sidecar_rows:
        write_csv(sidecar_path, sidecar_rows)
        status = "ok" if missing_series == 0 and mismatched_actual == 0 else "partial_with_warnings"
    else:
        status = "failed_no_rows"
    return {
        "source": source,
        "status": status,
        "dataset": dataset,
        "raw_rows": len(raw_rows),
        "windows": len(windows),
        "rows": len(sidecar_rows),
        "missing_series": missing_series,
        "mismatched_actual": mismatched_actual,
        "sidecar": str(sidecar_path.relative_to(ROOT)),
    }


def source_list(args: argparse.Namespace) -> list[str]:
    if args.source:
        return args.source
    if args.all_raw:
        return sorted(path.stem for path in RAW_DIR.glob("*.csv") if not path.name.endswith("_history_context.csv"))
    rows = read_csv(Path(args.source_gap_csv))
    return [row["source"] for row in rows if row.get("source")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append")
    parser.add_argument("--source-gap-csv", default=str(SOURCE_GAPS.relative_to(ROOT)))
    parser.add_argument("--all-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = [build_for_source(source, overwrite=args.overwrite) for source in source_list(args)]
    write_csv(STATUS_OUT, results)
    counts: dict[str, int] = {}
    for row in results:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1

    lines = [
        "# History Sidecar Reconstruction Report",
        "",
        "This report reconstructs exact per-window history/context sidecars from existing raw forecast artifacts and local GIFT-Eval/FRED data. It does not rerun TSFM inference.",
        "",
        "## Summary",
        "",
        f"- Sources requested: `{len(results)}`.",
        f"- Status counts: `{counts}`.",
        f"- Total sidecar rows: `{sum(int(row.get('rows') or 0) for row in results)}`.",
        "",
        "## Artifact",
        "",
        f"- `{STATUS_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
    ]
    DOC_OUT.write_text("\n".join(lines) + "\n")
    print({"status": "ok", "sources": len(results), "counts": counts, "rows": sum(int(row.get("rows") or 0) for row in results)})


if __name__ == "__main__":
    main()
