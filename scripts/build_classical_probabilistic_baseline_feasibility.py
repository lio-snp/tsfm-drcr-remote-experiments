#!/usr/bin/env python3
"""Audit whether native classical probabilistic baselines can be fit locally."""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw_forecasts"
OUT_DIR = ROOT / "results" / "aaai_stress"
DOCS = ROOT / "docs"

CSV_OUT = OUT_DIR / "classical_probabilistic_baseline_feasibility.csv"
DOC_OUT = DOCS / "classical_probabilistic_baseline_feasibility.md"

HISTORY_TOKENS = ("history", "context_values", "train_values", "insample", "past_values")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
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


def main() -> None:
    rows: list[dict[str, object]] = []
    baseline_modes: Counter[str] = Counter()
    q3_files = 0
    q9_files = 0
    history_files = 0
    history_sidecar_files = 0

    sidecars = {path.name.removesuffix("_history_context.csv"): path for path in RAW_DIR.glob("*_history_context.csv")}
    for path in sorted(item for item in RAW_DIR.glob("*.csv") if not item.name.endswith("_history_context.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            first = next(reader, {})
        history_columns = [
            column
            for column in columns
            if any(token in column.lower() for token in HISTORY_TOKENS)
        ]
        baseline_mode = first.get("baseline_mode", "")
        baseline_modes[baseline_mode or "<missing>"] += 1
        has_q3 = all(column in columns for column in ["forecast_q10", "forecast_q50", "forecast_q90"])
        has_q9 = all(f"forecast_q{level}" in columns for level in range(10, 100, 10))
        sidecar_path = sidecars.get(path.stem)
        has_history_sidecar = sidecar_path is not None
        q3_files += int(has_q3)
        q9_files += int(has_q9)
        history_sidecar_files += int(has_history_sidecar)
        history_files += int(bool(history_columns) or has_history_sidecar)
        rows.append(
            {
                "raw_file": str(path.relative_to(ROOT)),
                "n_columns": len(columns),
                "baseline_mode": baseline_mode or "<missing>",
                "has_baseline_forecast": int("baseline_forecast" in columns),
                "has_q3": int(has_q3),
                "has_q9": int(has_q9),
                "has_history_values": int(bool(history_columns) or has_history_sidecar),
                "has_history_sidecar": int(has_history_sidecar),
                "history_sidecar": str(sidecar_path.relative_to(ROOT)) if sidecar_path else "",
                "history_like_columns": ";".join(history_columns),
            }
        )

    write_csv(CSV_OUT, rows)

    statsmodels_available = importlib.util.find_spec("statsmodels") is not None
    mode_rows = [
        {"baseline_mode": mode, "n_files": count}
        for mode, count in baseline_modes.most_common()
    ]
    example_rows = rows[:10]
    lines = [
        "# Classical Probabilistic Baseline Feasibility Audit",
        "",
        "## Conclusion",
        "",
        "- `statsmodels` is available locally, so model fitting libraries are not the immediate blocker.",
        f"- Raw forecast CSV files audited: `{len(rows)}`.",
        f"- Files with q10/q50/q90 TSFM intervals: `{q3_files}`.",
        f"- Files with full q10..q90 grid: `{q9_files}`.",
        f"- Files with history/context value columns: `{history_files}`.",
        f"- Files with history/context sidecars: `{history_sidecar_files}`.",
        (
            "- Current raw artifacts contain classical point forecasts and TSFM quantiles, but not the actual historical context values needed to fit native AutoETS/AutoARIMA/Theta prediction intervals."
            if history_files == 0
            else "- At least one raw artifact now has history/context values or a sidecar; native classical interval fitting should be re-audited for those files."
        ),
        (
            "- Therefore the present `classical_residual_calibrated` baseline is the strongest fair probabilistic classical baseline available from current artifacts; native classical interval baselines require a rerun/exporter change."
            if history_files == 0
            else "- The historical-context blocker is no longer universal; fit native classical interval baselines for the sidecar-backed files before final claims."
        ),
        "",
        "## Environment",
        "",
        f"- statsmodels importable: `{statsmodels_available}`.",
        "",
        "## Baseline Mode Distribution",
        "",
        markdown_table(mode_rows, [("baseline_mode", "Baseline mode"), ("n_files", "Raw files")]),
        "",
        "## Example Raw-File Audit Rows",
        "",
        markdown_table(
            example_rows,
            [
                ("raw_file", "Raw file"),
                ("baseline_mode", "Baseline"),
                ("has_baseline_forecast", "Point"),
                ("has_q3", "q3"),
                ("has_q9", "q9"),
                ("has_history_values", "History values"),
            ],
        ),
        "",
        "## Required Exporter Change",
        "",
        "To add native probabilistic AutoETS/AutoARIMA/Theta baselines, each raw window should export:",
        "",
        "- the pre-origin history/context values used by the TSFM and classical baseline, preferably via `--export-history-sidecar`;",
        "- frequency/season length and any transformation metadata;",
        "- the baseline model class selected by the original benchmark runner;",
        "- native prediction interval quantiles or enough fitted residual/state information to reconstruct them.",
        "",
        "Until then, paper wording should say that stronger native classical interval baselines are a remaining evidence gap rather than an omitted completed experiment.",
    ]
    DOC_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"wrote {DOC_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
