#!/usr/bin/env python
"""Generate paper-facing main result CSV and LaTeX tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "tables"
CSV_PATH = OUT_DIR / "main_results_table.csv"
TEX_PATH = OUT_DIR / "main_results_table.tex"
REPORT_PATH = ROOT / "docs" / "main_results_table_report.md"


def pct(value: float) -> float:
    return 100.0 * float(value)


def fmt_pct(value: float) -> str:
    return f"{pct(value):.1f}"


def fmt_signed_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{pct(value):.1f}"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def add_row(rows: list[dict[str, object]], **kwargs: object) -> None:
    rows.append(kwargs)


def load_bootstrap(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def build_rows() -> list[dict[str, object]]:
    robustness = pd.read_csv(ROOT / "results/failure_family/multimetric_robustness_summary.csv")
    synthetic = pd.read_csv(ROOT / "results/failure_family/tsfm_synthetic_ablation_summary.csv")
    repair = pd.read_csv(ROOT / "results/repair/selective_gate_strategy_comparison.csv")
    cross = pd.read_csv(ROOT / "results/repair/cross_family_selective_repair_strategy_summary.csv")
    chronos_boot = load_bootstrap("results/scaling/chronos_bolt_scaling_bootstrap.json")
    moirai_boot = load_bootstrap("results/scaling/moirai_1_1_scaling_bootstrap.json")

    rows: list[dict[str, object]] = []
    rob = {row["group"]: row for _, row in robustness.iterrows()}
    factorized = rob["factorized_regime_n_ge_2"]
    outside = rob["outside_factorized_regime"]
    denom = rob["denominator_fragile"]
    add_row(
        rows,
        block="Failure family",
        evidence="Factorized regime vs. outside",
        scope="700 real windows",
        n="231 / 469",
        metric_a="MASE fail",
        before=fmt_pct(outside["mase_rer_failure_rate"]),
        after=fmt_pct(factorized["mase_rer_failure_rate"]),
        delta=fmt_signed_pct(factorized["mase_rer_failure_rate"] - outside["mase_rer_failure_rate"]),
        gate_or_ci="--",
        takeaway="Low-local-structure family is clearer under scale-normalized error than raw MAE.",
    )
    add_row(
        rows,
        block="Evaluation caveat",
        evidence="Denominator-fragile windows",
        scope="Real benchmark audit",
        n=str(int(denom["n_windows"])),
        metric_a="MAE/MASE fail",
        before="--",
        after=f"{fmt_pct(denom['mae_rer_failure_rate'])}/{fmt_pct(denom['mase_rer_failure_rate'])}",
        delta="--",
        gate_or_ci="fragile=100.0",
        takeaway="RER spikes must be separated from substantive forecast failure.",
    )

    synth = {(str(row["factor"]), float(row["value"])): row for _, row in synthetic.iterrows()}
    ctx24 = synth[("context_length", 24.0)]
    ctx96 = synth[("context_length", 96.0)]
    decay0 = synth[("decay_rate", 0.0)]
    decay005 = synth[("decay_rate", 0.05)]
    add_row(
        rows,
        block="Controlled ablation",
        evidence="Short context: 24 vs. 96",
        scope="Chronos-on-synthetic",
        n=f"{int(ctx24['n_windows'])}+{int(ctx96['n_windows'])}",
        metric_a="Failure",
        before=fmt_pct(ctx96["failure_rate_delta_005"]),
        after=fmt_pct(ctx24["failure_rate_delta_005"]),
        delta=fmt_signed_pct(ctx24["failure_rate_delta_005"] - ctx96["failure_rate_delta_005"]),
        gate_or_ci="paired factors",
        takeaway="Short history sharply increases failure under controlled generation.",
    )
    add_row(
        rows,
        block="Controlled ablation",
        evidence="Decay dynamics: 0 vs. 0.05",
        scope="Chronos-on-synthetic",
        n=f"{int(decay0['n_windows'])}+{int(decay005['n_windows'])}",
        metric_a="Coverage",
        before=fmt_pct(decay0["mean_empirical_coverage_90"]),
        after=fmt_pct(decay005["mean_empirical_coverage_90"]),
        delta=fmt_signed_pct(decay005["mean_empirical_coverage_90"] - decay0["mean_empirical_coverage_90"]),
        gate_or_ci="paired factors",
        takeaway="Decay is mostly a calibration/coverage collapse, not just an MAE story.",
    )

    def lookup(df: pd.DataFrame, strategy: str, group: str) -> pd.Series:
        subset = df[(df["strategy_id"] == strategy) & (df["group"] == group)]
        if subset.empty:
            raise KeyError((strategy, group))
        return subset.iloc[0]

    sel = lookup(repair, "selective_margin_pareto", "overall")
    sel_pc = lookup(repair, "selective_margin_pareto", "positive_control")
    add_row(
        rows,
        block="Selective repair",
        evidence="Chronos Pareto gate",
        scope="5 held-out sources",
        n=str(int(sel["n_windows"])),
        metric_a="Failure",
        before=fmt_pct(sel["model_failure_rate_delta_005"]),
        after=fmt_pct(sel["repair_failure_rate_delta_005"]),
        delta=fmt_signed_pct(sel["failure_rate_reduction"]),
        gate_or_ci=f"gate={fmt_pct(sel['gate_rate'])}; PC dRER={fmt_num(sel_pc['median_rer_delta'], 3)}",
        takeaway="Selective repair reduces failure without positive-control median RER cost.",
    )

    cf = lookup(cross, "risk_controlled_leave_family_out", "overall")
    cf_pc = lookup(cross, "risk_controlled_leave_family_out", "positive_control")
    cf_timesfm_pc = lookup(cross, "risk_controlled_leave_family_out", "family:timesfm|role:positive_control")
    cf_tail = lookup(cross, "cross_family_tail_safe_pareto", "overall")
    cf_tail_pc = lookup(cross, "cross_family_tail_safe_pareto", "positive_control")
    cf_high = lookup(cross, "previous_high_recall", "overall")
    cf_global = lookup(cross, "global_blend_w0.75", "positive_control")
    cf_moirai = lookup(cross, "cross_family_tail_safe_pareto", "family:moirai")
    cf_timesfm = lookup(cross, "cross_family_tail_safe_pareto", "family:timesfm")
    cf_timesfm_failure = lookup(cross, "cross_family_tail_safe_pareto", "family:timesfm|role:failure_target")
    tail_gate_delta = cf_high["gate_rate"] - cf_tail["gate_rate"]
    add_row(
        rows,
        block="Cross-family repair",
        evidence="Risk-controlled LFO conflict-shield gate",
        scope="Chronos+Moirai+TimesFM",
        n=str(int(cf["n_windows"])),
        metric_a="Failure",
        before=fmt_pct(cf["model_failure_rate_delta_005"]),
        after=fmt_pct(cf["repair_failure_rate_delta_005"]),
        delta=fmt_signed_pct(cf["failure_rate_reduction"]),
        gate_or_ci=(
            f"gate={fmt_pct(cf['gate_rate'])}; PC p90 dRER={fmt_num(cf_pc['p90_rer_delta'], 3)}; "
            f"TimesFM PC p90={fmt_num(cf_timesfm_pc['p90_rer_delta'], 3)}"
        ),
        takeaway="A clean leave-family-out protocol uses an expert-conflict interval shield to preserve positive-control tail safety while beating global blending.",
    )
    add_row(
        rows,
        block="Cross-family repair",
        evidence="Tuned tail-safe frontier",
        scope="Chronos+Moirai+TimesFM",
        n=str(int(cf_tail["n_windows"])),
        metric_a="Failure",
        before=fmt_pct(cf_tail["model_failure_rate_delta_005"]),
        after=fmt_pct(cf_tail["repair_failure_rate_delta_005"]),
        delta=fmt_signed_pct(cf_tail["failure_rate_reduction"]),
        gate_or_ci=(
            f"gate={fmt_pct(cf_tail['gate_rate'])}; PC p90 dRER={fmt_num(cf_tail_pc['p90_rer_delta'], 3)}; "
            f"high-recall gate={fmt_pct(cf_high['gate_rate'])}"
        ),
        takeaway=f"The tuned frontier gives stronger repair with {fmt_pct(tail_gate_delta)}pp fewer gates than high-recall, but it is not the cleanest held-out protocol.",
    )
    add_row(
        rows,
        block="Cross-family repair",
        evidence="Family transfer profile",
        scope="Moirai / TimesFM family / TimesFM covid",
        n=f"{int(cf_moirai['n_windows'])}/{int(cf_timesfm['n_windows'])}/{int(cf_timesfm_failure['n_windows'])}",
        metric_a="Failure reduction",
        before="--",
        after=(
            f"{fmt_pct(cf_moirai['failure_rate_reduction'])}/"
            f"{fmt_pct(cf_timesfm['failure_rate_reduction'])}/"
            f"{fmt_pct(cf_timesfm_failure['failure_rate_reduction'])}"
        ),
        delta="--",
        gate_or_ci=(
            f"PC median/p90 dRER={fmt_num(cf_tail_pc['median_rer_delta'], 3)}/{fmt_num(cf_tail_pc['p90_rer_delta'], 3)}; "
            f"global PC dRER={fmt_num(cf_global['median_rer_delta'], 3)}"
        ),
        takeaway="TimesFM transfer is now nontrivial under the shield, but residual covid failure keeps the claim at partial repair rather than universal success.",
    )

    chronos_interactions = chronos_boot["robust_continuous_primary_sensitivity"]["interactions"]
    moirai_interaction = moirai_boot["robust_continuous_primary_sensitivity"]["interactions"][
        "covid_deaths_d_short_minus_loop_seattle_h_short"
    ]
    chronos_loop = chronos_interactions["covid_deaths_d_short_minus_loop_seattle_h_short"]
    chronos_solar = chronos_interactions["covid_deaths_d_short_minus_solar_10t_short"]
    add_row(
        rows,
        block="Scaling mechanism",
        evidence="Capacity interaction",
        scope="Chronos 4 sizes; Moirai 3 sizes",
        n="768 / 48 rows",
        metric_a="log(variance) slope diff",
        before="--",
        after=(
            f"Chronos {fmt_num(chronos_loop['point_slope_difference'], 2)}/"
            f"{fmt_num(chronos_solar['point_slope_difference'], 2)}; "
            f"Moirai {fmt_num(moirai_interaction['point_slope_difference'], 2)}"
        ),
        delta="--",
        gate_or_ci=(
            f"CIs below 0: "
            f"[{fmt_num(chronos_loop['bootstrap_ci']['high'], 2)}, "
            f"{fmt_num(chronos_solar['bootstrap_ci']['high'], 2)}, "
            f"{fmt_num(moirai_interaction['bootstrap_ci']['high'], 2)}]"
        ),
        takeaway="Pretrained zero-shot scaling does not simply follow ERM variance-explosion intuition.",
    )
    return rows


def write_csv_table(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex_table(rows: list[dict[str, object]]) -> None:
    header = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Main empirical evidence for low-local-structure TSFM degeneration and selective repair. Failure is RER$>1.05$ unless otherwise noted; lower failure and gate rates are better, while coverage should stay high.}
\label{tab:main_results}
\begin{tabular}{lllrllllp{4.8cm}}
\toprule
Block & Evidence & Scope & $n$ & Metric & Before & After & Gate / CI & Takeaway \\
\midrule"""
    body = []
    last_block = None
    for row in rows:
        if last_block is not None and row["block"] != last_block:
            body.append(r"\midrule")
        last_block = row["block"]
        body.append(
            " & ".join(
                [
                    latex_escape(row["block"]),
                    latex_escape(row["evidence"]),
                    latex_escape(row["scope"]),
                    latex_escape(row["n"]),
                    latex_escape(row["metric_a"]),
                    latex_escape(row["before"]),
                    latex_escape(row["after"]),
                    latex_escape(row["gate_or_ci"]),
                    latex_escape(row["takeaway"]),
                ]
            )
            + r" \\"
        )
    footer = r"""\bottomrule
\end{tabular}
\end{table*}
"""
    TEX_PATH.write_text("\n".join([header, *body, footer]))


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Main Results Table Report",
        "",
        "Generated artifacts:",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{TEX_PATH.relative_to(ROOT)}`",
        "",
        "The table is organized as a single paper-facing narrative: failure-family evidence, controlled ablation, selective repair, cross-family transfer, and scaling mechanism. It intentionally preserves the negative/boundary results: raw MAE does not alone define the failure family, global blending has positive-control margin cost, and the TimesFM covid slice remains a residual failure regime even though the conflict-shielded repair now gives nontrivial improvement.",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- **{row['block']} / {row['evidence']}**: {row['metric_a']} {row['before']} -> {row['after']} ({row['gate_or_ci']})."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    write_csv_table(rows)
    write_tex_table(rows)
    write_report(rows)
    print(CSV_PATH.relative_to(ROOT))
    print(TEX_PATH.relative_to(ROOT))
    print(REPORT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
