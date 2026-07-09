#!/usr/bin/env python
"""Apply the frozen DRCR-Smooth protocol to broader evidence slices.

This script is intentionally not a tuning/search pass.  It loads the frozen
DRCR-Smooth protocol and applies the already selected smooth candidate to
available Moirai full-grid reruns plus TimesFM/Moirai q10-q50-q90 interval
artifacts.  Full nine-quantile slices are reported separately from q3 interval
proxy slices so the probabilistic claim boundary stays explicit.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402
import run_factorized_failure_family_goal as base  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402
from low_snr_tsfm.quantile_artifacts import quantile_matrix_from_rows  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_smooth_frozen_expansion_report.md"
STATUS_OUT = OUT_DIR / "drcr_smooth_frozen_expansion_status.json"
WINDOW_OUT = OUT_DIR / "drcr_smooth_frozen_expansion_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_smooth_frozen_expansion_summary.csv"
INVENTORY_OUT = OUT_DIR / "drcr_smooth_frozen_expansion_inventory.csv"

PHASE = "frozen_expansion"
STRATEGY_ID = "drcr_smooth_frozen_expansion"

SOURCES = [
    {
        "slug": "moirai2_fullgrid_ctx1680_m12_covid_deaths_short_auto_ets",
        "family": "moirai",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q9_fullgrid",
        "expansion_set": "new_moirai_q9",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai2_fullgrid_ctx1680_solar_m16_solar_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "solar_10t_short",
        "evidence_tier": "q9_fullgrid",
        "expansion_set": "new_moirai_q9",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai2_fullgrid_ctx1680_m64_covid_deaths_short_auto_ets",
        "family": "moirai",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q9_fullgrid",
        "expansion_set": "reference_overlap_moirai_q9",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai2_fullgrid_ctx1680_solar_m64_solar_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "solar_10t_short",
        "evidence_tier": "q9_fullgrid",
        "expansion_set": "reference_overlap_moirai_q9",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai2_loop_m8_loop_seattle_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_q3_domain",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_small_scaling_covid_deaths_d_short_auto_ets",
        "family": "moirai",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-small",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_base_scaling_covid_deaths_d_short_auto_ets",
        "family": "moirai",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-base",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_large_scaling_covid_deaths_d_short_auto_ets",
        "family": "moirai",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-large",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_small_scaling_loop_seattle_h_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-small",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_base_scaling_loop_seattle_h_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-base",
        "params_m": "",
    },
    {
        "slug": "moirai_1_1_large_scaling_loop_seattle_h_short_seasonal_naive",
        "family": "moirai",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_moirai_1_1_q3_scaling",
        "size": "1.1-large",
        "params_m": "",
    },
    {
        "slug": "timesfm_2_5_m128_covid_deaths_short_auto_ets",
        "family": "timesfm",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_domain",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_m64_covid_deaths_short_auto_ets",
        "family": "timesfm",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_context_ablation",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_m16_covid_deaths_short_auto_ets",
        "family": "timesfm",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_context_ablation",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_m8_covid_deaths_short_auto_ets",
        "family": "timesfm",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_context_ablation",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_finance_fred_finance_fred_stress",
        "family": "timesfm",
        "role": "stress_target",
        "target_id": "finance_fred_stress",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_domain",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_bizitobs_m30_bizitobs_application_short_auto_arima",
        "family": "timesfm",
        "role": "stress_target",
        "target_id": "bizitobs_application_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_domain",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_bizitobs_m8_bizitobs_application_short_auto_arima",
        "family": "timesfm",
        "role": "stress_target",
        "target_id": "bizitobs_application_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_context_ablation",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive",
        "family": "timesfm",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_domain",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_loop_m8_loop_seattle_short_seasonal_naive",
        "family": "timesfm",
        "role": "positive_control",
        "target_id": "loop_seattle_h_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_context_ablation",
        "size": "2.5-200m",
        "params_m": 200,
    },
    {
        "slug": "timesfm_2_5_solar_m8_solar_short_seasonal_naive",
        "family": "timesfm",
        "role": "positive_control",
        "target_id": "solar_10t_short",
        "evidence_tier": "q3_interval_proxy",
        "expansion_set": "new_timesfm_q3_domain",
        "size": "2.5-200m",
        "params_m": 200,
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("dataset", ""), row.get("series_id", ""), str(row.get("window_index", "")))


def load_expansion_windows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for source in SOURCES:
        slug = str(source["slug"])
        raw_path = ROOT / "results" / "raw_forecasts" / f"{slug}.csv"
        feature_path = ROOT / "results" / "failure_mining" / f"{slug}_predictor_features.csv"
        status_path = ROOT / "results" / "raw_forecasts" / f"{slug}_status.json"
        if not raw_path.exists() or not feature_path.exists():
            inventory.append(
                {
                    **source,
                    "n_windows": 0,
                    "skipped_missing_features": 0,
                    "observed_quantile_level_counts": "",
                    "status_quantile_levels": "",
                    "status": "missing_raw_or_features",
                }
            )
            continue

        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        feature_rows = {feature_key(row): row for row in read_csv(feature_path)}
        raw_groups = base.raw_window_map(raw_path)
        matched = 0
        skipped = 0
        observed_level_counts: set[int] = set()
        for raw_key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = raw_key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(
                ("", series_id, window_index)
            )
            if feature is None:
                skipped += 1
                continue
            levels, quantile_grid = quantile_matrix_from_rows(rows)
            observed_level_counts.add(len(levels))
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = base.raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q50 = np.asarray([finite_float(row.get("forecast_q50")) for row in rows], dtype=float)
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            windows.append(
                {
                    "family": source["family"],
                    "source": slug,
                    "role": source["role"],
                    "target_id": source["target_id"],
                    "evidence_tier": source["evidence_tier"],
                    "expansion_set": source["expansion_set"],
                    "size": source["size"],
                    "params_m": source["params_m"],
                    "dataset": dataset,
                    "model": model_name,
                    "series_id": series_id,
                    "origin": origin,
                    "window_index": window_index,
                    "feature": feature,
                    "actual": actual,
                    "model_forecast": model,
                    "baseline_forecast": baseline,
                    "quantile_levels": levels,
                    "quantile_grid": quantile_grid,
                    "quantile_grid_n_levels": len(levels),
                    "q10": q10,
                    "q50": q50,
                    "q90": q90,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "model_rer": model_rer,
                    "model_failure": int(model_rer > 1.05),
                }
            )
            matched += 1
        inventory.append(
            {
                **source,
                "n_windows": matched,
                "skipped_missing_features": skipped,
                "observed_quantile_level_counts": ";".join(str(value) for value in sorted(observed_level_counts)),
                "status_quantile_levels": ";".join(str(value) for value in status.get("quantile_levels", [])),
                "raw_path": str(raw_path.relative_to(ROOT)),
                "feature_path": str(feature_path.relative_to(ROOT)),
                "status": "ok",
            }
        )
    return windows, inventory


def apply_frozen_candidate(windows: list[dict[str, object]]) -> list[dict[str, object]]:
    smooth.validate_protocol_config()
    policy = smooth.crc.fullgrid.common_cpr_policy()
    candidates = {str(candidate["candidate_id"]): candidate for candidate in smooth.candidate_configs()}
    selected = candidates[str(smooth.PROTOCOL_CONFIG["selected_candidate_id"])]
    rows: list[dict[str, object]] = []
    for window in windows:
        row = smooth.apply_candidate_to_window(
            window,
            policy,
            selected,
            split_protocol=smooth.SPLIT_PROTOCOL,
            split_id=smooth.SPLIT_ID,
            phase=PHASE,
        )
        row.update(
            {
                "strategy_id": STRATEGY_ID,
                "protocol_id": smooth.PROTOCOL_ID,
                "protocol_sha256": smooth.protocol_sha256(),
                "evidence_tier": window["evidence_tier"],
                "expansion_set": window["expansion_set"],
                "size": window["size"],
                "params_m": window["params_m"],
                "quantile_grid_n_levels": window["quantile_grid_n_levels"],
                "model_mae_rer": window["model_rer"],
            }
        )
        rows.append(row)
    return rows


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", rows)]
    for key in ["evidence_tier", "expansion_set", "family", "role", "target_id", "source"]:
        for value in sorted({str(row[key]) for row in rows}):
            groups.append((f"{key}:{value}", key, [row for row in rows if str(row[key]) == value]))
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            groups.append(
                (
                    f"family:{family}|role:{role}",
                    "family_role",
                    [row for row in rows if row["family"] == family and row["role"] == role],
                )
            )
    for tier in sorted({str(row["evidence_tier"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["evidence_tier"] == tier}):
            groups.append(
                (
                    f"evidence_tier:{tier}|role:{role}",
                    "tier_role",
                    [row for row in rows if row["evidence_tier"] == tier and row["role"] == role],
                )
            )

    summary: list[dict[str, object]] = []
    for group, group_type, subset in groups:
        if not subset:
            continue
        base_row = smooth.summarize_candidate(subset, group, group_type)
        base_row.update(
            {
                "protocol_id": smooth.PROTOCOL_ID,
                "protocol_sha256": smooth.protocol_sha256(),
                "phase": PHASE,
                "candidate_id": subset[0]["candidate_id"],
                "candidate_type": subset[0]["candidate_type"],
                "quantile_grid_n_levels": ";".join(
                    str(value) for value in sorted({int(row["quantile_grid_n_levels"]) for row in subset})
                ),
            }
        )
        summary.append(base_row)
    return summary


def report_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = [
        "overall",
        "evidence_tier:q9_fullgrid",
        "evidence_tier:q9_fullgrid|role:failure_target",
        "evidence_tier:q9_fullgrid|role:positive_control",
        "expansion_set:new_moirai_q9",
        "evidence_tier:q3_interval_proxy",
        "family:timesfm",
        "family:timesfm|role:failure_target",
        "family:timesfm|role:stress_target",
        "family:timesfm|role:positive_control",
    ]
    rows: list[dict[str, object]] = []
    by_group = {str(row["group"]): row for row in summary}
    for group in wanted:
        row = by_group.get(group)
        if row is None:
            continue
        rows.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "Levels": row["quantile_grid_n_levels"],
                "ModelWQL": num(row["model_median_wql_rer"]),
                "RepairWQL": num(row["repair_median_wql_rer"]),
                "FailRed": pct(row["wql_failure_reduction_vs_model"]),
                "Coverage": num(row["repair_mean_coverage"]),
                "WQLHarm": pct(row["repair_wql_noninferiority_harm_rate"]),
                "SafetyHarm": pct(row["repair_safety_harm_rate"]),
                "Score": num(row["smooth_width_score_mean"]),
            }
        )
    return rows


def write_report(summary: list[dict[str, object]], inventory: list[dict[str, object]], status: dict[str, object]) -> None:
    inventory_rows = [
        {
            "Source": row["slug"],
            "Family": row["family"],
            "Tier": row["evidence_tier"],
            "Set": row["expansion_set"],
            "Role": row["role"],
            "N": row["n_windows"],
            "Levels": row["observed_quantile_level_counts"],
            "Status": row["status"],
        }
        for row in inventory
    ]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR-Smooth Frozen Expansion",
                "",
                "## Purpose",
                "",
                "This applies the frozen DRCR-Smooth selected head to broader Moirai and TimesFM slices without adding or retuning candidates.",
                "",
                "## Protocol Boundary",
                "",
                f"- Frozen protocol: `{status['protocol_id']}`.",
                f"- Protocol config: `{status['protocol_config']}`.",
                f"- Protocol SHA-256: `{status['protocol_sha256']}`.",
                f"- Selected candidate: `{status['selected_candidate_id']}`.",
                "- `q9_fullgrid` rows are full nine-quantile WQL evidence.",
                "- `q3_interval_proxy` rows use only q10/q50/q90 artifacts; they are useful interval-readiness evidence but should not be framed as full paper-faithful WQL.",
                "",
                "## Result",
                "",
                markdown_table(
                    report_rows(summary),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Levels", "Q levels"),
                        ("ModelWQL", "Model WQL"),
                        ("RepairWQL", "Repair WQL"),
                        ("FailRed", "Fail red."),
                        ("Coverage", "Coverage"),
                        ("WQLHarm", "WQL harm"),
                        ("SafetyHarm", "Safety harm"),
                        ("Score", "Score"),
                    ],
                ),
                "",
                "## Inventory",
                "",
                markdown_table(
                    inventory_rows,
                    [
                        ("Source", "Source"),
                        ("Family", "Family"),
                        ("Tier", "Tier"),
                        ("Set", "Set"),
                        ("Role", "Role"),
                        ("N", "N"),
                        ("Levels", "Levels"),
                        ("Status", "Status"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The frozen head transfers positively on the failure side: q9 Moirai covid and q3 TimesFM covid both reduce WQL-RER without any expansion-set retuning.",
                "- The same frozen head is not universally safe: q9 solar positive controls and q3 finance stress slices show WQL harm. This should be framed as failure-side repair evidence, not a universal safe-repair claim.",
                "- A follow-up component ablation shows this harm is driven mostly by interval scaling, not the CPR point shift. This motivates an intervention safety gate that can keep point repair while vetoing risky interval scaling.",
                "- TimesFM currently contributes q3 interval-proxy evidence only; full-grid TimesFM reruns remain required before upgrading TimesFM to a main probabilistic robustness claim.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{INVENTORY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    windows, inventory = load_expansion_windows()
    if not windows:
        raise SystemExit("No expansion windows available")
    rows = apply_frozen_candidate(windows)
    summary = build_summary(rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "protocol_id": smooth.PROTOCOL_ID,
        "protocol_config": str(smooth.CONFIG_PATH.relative_to(ROOT)),
        "protocol_sha256": smooth.protocol_sha256(),
        "selected_candidate_id": smooth.SMOOTH_CANDIDATE_ID,
        "phase": PHASE,
        "strategy_id": STRATEGY_ID,
        "n_sources": len({row["source"] for row in rows}),
        "n_windows": len(rows),
        "n_families": len({row["family"] for row in rows}),
        "n_targets": len({row["target_id"] for row in rows}),
        "evidence_tiers": sorted({row["evidence_tier"] for row in rows}),
        "quantile_grid_n_levels": sorted({int(row["quantile_grid_n_levels"]) for row in rows}),
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "inventory": str(INVENTORY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(INVENTORY_OUT, inventory)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(summary, inventory, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
