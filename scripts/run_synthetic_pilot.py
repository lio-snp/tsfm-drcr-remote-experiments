#!/usr/bin/env python
"""Run a baseline-only synthetic pilot for protocol sanity checks."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.evaluation import evaluate_baselines_records
from low_snr_tsfm.features import feature_vector
from low_snr_tsfm.synthetic import SyntheticSpec, generate


DEFAULT_CONFIG = {
    "seed": 7,
    "length": 1024,
    "context_length": 128,
    "horizon": 24,
    "step": 24,
    "season_length": 24,
    "ar_lags": 12,
    "processes": {
        "ar1_phi": {"values": [0.0, 0.1, 0.3, 0.6, 0.9]},
        "seasonal_amplitude": {"values": [0.0, 0.25, 0.5, 1.0, 2.0]},
        "local_level_ratio": {"values": [0.01, 0.05, 0.10, 0.25, 0.50]},
    },
    "failure_thresholds": {"delta": [0.0, 0.05, 0.10], "severe": 0.25},
}


def build_specs(config: dict) -> list[SyntheticSpec]:
    seed = int(config["seed"])
    length = int(config["length"])
    specs: list[SyntheticSpec] = [
        SyntheticSpec("white_noise", length, seed + 1, {"sigma": 1.0}),
        SyntheticSpec(
            "garch11",
            length,
            seed + 2,
            {"omega": 0.05, "alpha": 0.08, "beta": 0.90},
        ),
        SyntheticSpec(
            "regime_switching_ar",
            length,
            seed + 3,
            {"phi_low": 0.05, "phi_high": 0.8, "switch_prob": 0.03, "sigma": 1.0},
        ),
    ]
    for idx, phi in enumerate(config["processes"]["ar1_phi"]["values"]):
        specs.append(SyntheticSpec("ar1", length, seed + 100 + idx, {"phi": float(phi), "sigma": 1.0}))
    for idx, amp in enumerate(config["processes"]["seasonal_amplitude"]["values"]):
        specs.append(
            SyntheticSpec(
                "seasonal_ar",
                length,
                seed + 200 + idx,
                {
                    "phi": 0.2,
                    "seasonal_amplitude": float(amp),
                    "period": int(config["season_length"]),
                    "sigma": 1.0,
                },
            )
        )
    for idx, ratio in enumerate(config["processes"]["local_level_ratio"]["values"]):
        specs.append(
            SyntheticSpec(
                "local_level",
                length,
                seed + 300 + idx,
                {"level_sigma": float(ratio), "obs_sigma": 1.0},
            )
        )
    return specs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["process"]), str(row["model"]))].append(row)
    summary = []
    for (process, model), group in grouped.items():
        summary.append(
            {
                "process": process,
                "model": model,
                "mae": sum(float(row["mae"]) for row in group) / len(group),
                "rmse": sum(float(row["rmse"]) for row in group) / len(group),
                "n_windows": len(group),
            }
        )
    return sorted(summary, key=lambda row: (row["process"], row["mae"]))


def write_ar1_svg(path: Path, rows: list[dict]) -> None:
    ar_rows = [row for row in rows if row["process"] == "ar1" and "phi" in row]
    if not ar_rows:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='700' height='400'></svg>")
        return
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in ar_rows:
        grouped[(str(row["model"]), float(row["phi"]))].append(float(row["mae"]))
    model_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (model, phi), values in grouped.items():
        model_points[model].append((phi, sum(values) / len(values)))

    all_phi = [phi for points in model_points.values() for phi, _ in points]
    all_mae = [mae for points in model_points.values() for _, mae in points]
    min_x, max_x = min(all_phi), max(all_phi)
    min_y, max_y = min(all_mae), max(all_mae)
    width, height = 760, 430
    left, top, plot_w, plot_h = 60, 30, 560, 320

    def sx(x: float) -> float:
        return left + (x - min_x) / max(max_x - min_x, 1e-12) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / max(max_y - min_y, 1e-12) * plot_h

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='20' font-family='Arial' font-size='15'>Synthetic pilot: baseline MAE vs AR(1) phi</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='black'/>",
        f"<text x='{left + plot_w / 2 - 30}' y='{height - 35}' font-family='Arial' font-size='12'>AR(1) phi</text>",
        f"<text x='10' y='{top + plot_h / 2}' font-family='Arial' font-size='12' transform='rotate(-90 10 {top + plot_h / 2})'>Mean MAE</text>",
    ]
    for idx, (model, points) in enumerate(sorted(model_points.items())):
        color = colors[idx % len(colors)]
        points = sorted(points)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in points:
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='{color}'/>")
        legend_y = top + 18 * idx
        parts.append(f"<line x1='640' y1='{legend_y}' x2='660' y2='{legend_y}' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<text x='666' y='{legend_y + 4}' font-family='Arial' font-size='11'>{model}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    config = DEFAULT_CONFIG
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rows = []
    feature_rows = []
    for spec in build_specs(config):
        series = generate(spec)
        frame_rows = evaluate_baselines_records(
            series,
            context_length=int(config["context_length"]),
            horizon=int(config["horizon"]),
            step=int(config["step"]),
            season_length=int(config["season_length"]),
            ar_lags=int(config["ar_lags"]),
        )
        for row in frame_rows:
            row.update(spec.params)
            row["process"] = spec.process
            row["seed"] = spec.seed
        rows.extend(frame_rows)
        feature_rows.append(
            {
                "process": spec.process,
                "seed": spec.seed,
                **spec.params,
                **feature_vector(
                    series,
                    horizon=int(config["horizon"]),
                    context_length=int(config["context_length"]),
                    period=int(config["season_length"]),
                ),
            }
        )

    summary = summarize(rows)
    write_csv(results_dir / "synthetic_pilot_baselines.csv", rows)
    write_csv(results_dir / "synthetic_pilot_summary.csv", summary)
    write_csv(results_dir / "synthetic_pilot_features.csv", feature_rows)
    (results_dir / "synthetic_pilot_config.json").write_text(json.dumps(config, indent=2))
    write_ar1_svg(figures_dir / "synthetic_pilot_ar1_baselines.svg", rows)

    print(f"Wrote {len(rows)} forecast-window rows to {results_dir}")
    for row in summary[:20]:
        print(f"{row['process']:22s} {row['model']:20s} MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
