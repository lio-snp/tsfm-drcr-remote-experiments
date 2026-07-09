# AAAI Oral Goal Status Dashboard

This dashboard is generated from the current repository artifacts. It separates what is complete on the locked current suite from what still requires new inference, stronger baselines, or broader external-scale confirmation.

## Executive Read

- Active target: push DRCR from a 930-window paper-readiness prototype toward an AAAI main/oral-level experimental package.
- Current selected method: `drcr_expert_pull_1.25_cap_1.10`.
- Unique forecast windows: `1076`; expanded method rows in figure windows: `15064`.
- Model families in current suite: `chronos, moirai, timesfm`.
- Dataset count in current suite: `6`.
- Evidence tiers: `{'q9_fullgrid': 668, 'q3_interval_proxy': 408}`.
- Oral evidence gap queue: `17` P0 source reruns, `17` q9/full-grid source gaps, `0` history/context export gaps.
- Oral rerun command queue: `29` ready commands, `17` P0 ready, `0` blocked commands.
- Remote q9 execution pack: `17` P0 sources / `408` windows ready for larger-memory execution; completion audit `0`/`17` complete, ingestion manifest `0`/`17` ready for final-main refresh.
- Role mix: `{'failure_target': 572, 'positive_control': 416, 'stress_target': 88}`.
- Overall WQL-RER: native `0.998` -> selected DRCR `0.834`.
- q9 failure WQL-RER: native `3.411` -> selected DRCR `1.563`.
- Confirmatory salted split: `526` / `550`, selected `drcr_expert_pull_1.25_cap_1.10`, overall WQL-RER `1.040` -> `0.874`, q9 failure `2.854` -> `1.545`.
- Requirement status counts: `{'done': 1, 'done_current_suite': 5, 'partial': 3, 'done_current_suite_partial_broad': 1, 'ready_queue_not_executed': 1, 'ready_not_executed': 1, 'done_as_empirical_screen': 1, 'done_draft': 1}`.

## What Is Complete vs Still Missing

| Area | Status | Evidence | Remaining gap |
| --- | --- | --- | --- |
| Protocol freeze | done | final_frozen_protocol.md, candidate_policy_set.yaml, risk_thresholds.yaml, split/model/dataset manifests | For a stronger formal guarantee, repeat once more with protocol frozen before any candidate development. |
| Objective baselines | done_current_suite | 12 methods in main_baseline_comparison.csv including native, classical, calibrated classical, blend, gates, oracle, DRCR variants; sidecar-backed empirical classical interval audit covers 1076 windows; overall WQL-RER 0.922; positive/stress harm 57.9%/73.9%; feasibility/gap audits show 0 source-level history/context export gaps after sidecar reconstruction | Still missing strict benchmark-native AutoARIMA/AutoETS/Theta interval parity; sidecar-backed empirical classical intervals are now audited but should not be overclaimed as exact native interval reproduction. |
| Unified metrics | done_current_suite | WQL-RER, MAE-RER, RMSE-RER, WAPE-RER, coverage, harm, clipped sensitivity computed on final-main split | CRPS/MSIS and full MASE coverage are not headline-ready for every merged window. |
| Paper-faithful metrics | partial | Metric mapping and prior paper-faithful report exist; final-main emphasizes Chronos-style WQL/full-grid and point robustness | Need broader Moirai/GIFT MASE+WQL and TimesFM official metric tables for all expanded datasets. |
| q9/full-grid evidence | done_current_suite_partial_broad | 668 unique q9/full-grid windows in split manifest; q9_full_grid_results.csv generated | More Moirai/TimesFM full-grid or native quantile outputs would make this a stronger main-track claim; current oral gap matrix flags 17 source-level q9/full-grid reruns, with 17 P0 rerun commands ready. |
| Large-scale rerun queue | ready_queue_not_executed | 29 source-level rerun commands generated; 29 ready, 17 P0 ready, 0 blocked; each ready command includes exact window manifest, output slug, and history sidecar export. | The command queue is ready, but local execution is resource-limited: 2 timeout blockers recorded across moirai, timesfm. Execute the 17 P0 q9/full-grid reruns on remote or a larger-memory machine before claiming broader probabilistic evidence is complete. |
| Remote q9 execution pack | ready_not_executed | remote_q9_rerun_plan.csv and remote_q9_rerun_execution_pack.md define 17 P0 sources / 408 windows with expected raw, sidecar, status outputs and validation commands; completion audit currently has 0/17 sources complete, ingestion manifest has 0/17 ready for final-main refresh | Pack is an execution contract, not completed inference. Run it on remote/larger-memory hardware, rerun the completion audit, backfill features, then refresh final-main source inventory before claiming these gaps are closed. |
| Model-family breadth | partial | 22 model identifiers across 3 families: chronos, moirai, timesfm | Chronos is cleanest; Moirai/TimesFM are present but still less balanced than Chronos. |
| Dataset breadth | partial | 6 datasets in current locked suite: FRED finance stress seed, LOOP_SEATTLE/H/short, bizitobs_application/10S/short, covid_deaths/D/short, loop_seattle/H/short, solar/10T/short | Not yet a full GIFT-Eval/Monash/ETT/M4/M5-scale benchmark expansion. |
| Protected-window safety | done_current_suite | protected_safety_results.csv covers overall, positive controls, stress targets, finance FRED stress | Needs larger and more balanced protected groups for oral-level breadth. |
| LTT/CRC-style calibration | done_as_empirical_screen | Selected candidate has 3 primary risk rows; accepted=True | Do not claim strict distribution-free finite-sample validity under time-series dependence without a stricter pre-freeze rerun. |
| Denominator fragility | done_current_suite | clipped_sensitivity_table.csv and denominator_fragility_report.md generated | Add raw absolute-delta panels in appendix if space permits. |
| Main figure | done_draft | final_main_figure_draft.{png,pdf,svg} exists with unified color system | Needs final paper layout polish after deciding whether WidthVeto remains probe. |
| Confirmatory rerun | done_current_suite | final_main_confirmatory_* artifacts exist; salted split 526/550 selected drcr_expert_pull_1.25_cap_1.10 | Still needs external new-inference / broader-domain confirmatory evidence for oral-level breadth. |

## q9 Full-Grid Failure Results

| Method | WQL-RER | 95% CI | Harm | Coverage | MAE-RER |
| --- | --- | --- | --- | --- | --- |
| classical_deterministic | 1.000 | [1.000, 1.000] | 21.7% | 30.9% | 1.000 |
| classical_residual_calibrated | 3.007 | [1.587, 244.077] | 66.2% | 83.4% | 1.000 |
| drcr_cap_1.10 | 1.649 | [1.018, 3.762] | 16.6% | 82.7% | 1.000 |
| drcr_expert_pull_1.25_cap_1.10 | 1.563 | [1.021, 3.119] | 17.2% | 82.9% | 1.000 |
| drcr_expert_pull_1.50_cap_1.10 | 1.545 | [1.018, 3.119] | 17.8% | 82.6% | 1.000 |
| drcr_full | 1.649 | [1.021, 2.793] | 16.6% | 82.7% | 1.000 |
| drcr_point | 1.683 | [1.050, 3.420] | 15.9% | 83.9% | 1.000 |
| drcr_score_floor_0.60_cap_1.00 | 1.545 | [1.021, 3.119] | 15.9% | 83.4% | 1.000 |
| drcr_width_veto_expert_pull_1.50_cap_1.10 | 1.545 | [1.002, 3.119] | 15.3% | 81.7% | 1.000 |
| global_blend_w0.50 | 1.995 | [1.305, 6.877] | 12.7% | 62.7% | 2.424 |
| native_tsfm | 3.411 | [1.717, 12.895] | 0.0% | 76.2% | 3.894 |
| oracle_native_classical_drcr | 0.702 | [0.592, 0.776] | 0.0% | 61.4% | 0.977 |
| smooth_score_gate_t0.50_w1.00 | 1.167 | [1.044, 1.717] | 8.3% | 66.0% | 1.466 |
| width_gate_t0.10_w1.00 | 1.000 | [1.000, 1.000] | 13.4% | 41.2% | 1.000 |

## Benchmark Coverage Snapshot

| Family | Model | Dataset | Role | Tier | Cal | Test | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timesfm | timesfm_2_5_m128 | covid_deaths/D/short | failure_target | q3_interval_proxy | 60 | 68 | 128 |
| chronos | chronos_bolt_base | covid_deaths/D/short | failure_target | q9_fullgrid | 35 | 29 | 64 |
| chronos | chronos_bolt_mini | covid_deaths/D/short | failure_target | q9_fullgrid | 32 | 32 | 64 |
| chronos | chronos_bolt_small | covid_deaths/D/short | failure_target | q9_fullgrid | 36 | 28 | 64 |
| chronos | chronos_bolt_tiny | covid_deaths/D/short | failure_target | q9_fullgrid | 34 | 30 | 64 |
| chronos | chronos_bolt_base | solar/10T/short | positive_control | q9_fullgrid | 34 | 30 | 64 |
| chronos | chronos_bolt_mini | solar/10T/short | positive_control | q9_fullgrid | 38 | 26 | 64 |
| chronos | chronos_bolt_small | solar/10T/short | positive_control | q9_fullgrid | 32 | 32 | 64 |
| chronos | chronos_bolt_tiny | solar/10T/short | positive_control | q9_fullgrid | 31 | 33 | 64 |
| moirai | moirai2_fullgrid_ctx1680_m64 | covid_deaths/D/short | failure_target | q9_fullgrid | 31 | 33 | 64 |
| moirai | moirai2_fullgrid_ctx1680_solar_m64 | solar/10T/short | positive_control | q9_fullgrid | 32 | 32 | 64 |
| timesfm | timesfm_2_5_m64 | covid_deaths/D/short | failure_target | q3_interval_proxy | 34 | 30 | 64 |
| timesfm | timesfm_2_5_finance_fred | FRED finance stress seed | stress_target | q3_interval_proxy | 28 | 28 | 56 |
| timesfm | timesfm_2_5_loop_m32 | LOOP_SEATTLE/H/short | positive_control | q3_interval_proxy | 17 | 15 | 32 |
| timesfm | timesfm_2_5_bizitobs_m30 | bizitobs_application/10S/short | stress_target | q3_interval_proxy | 14 | 16 | 30 |
| moirai | moirai2_fullgrid_ctx1680_solar_m16 | solar/10T/short | positive_control | q9_fullgrid | 10 | 6 | 16 |
| timesfm | timesfm_2_5_m16 | covid_deaths/D/short | failure_target | q3_interval_proxy | 7 | 9 | 16 |
| moirai | moirai2_fullgrid_ctx1680_m12 | covid_deaths/D/short | failure_target | q9_fullgrid | 7 | 5 | 12 |
| moirai | moirai2_loop_m8 | LOOP_SEATTLE/H/short | positive_control | q3_interval_proxy | 4 | 4 | 8 |
| moirai | moirai_1_1_base | covid_deaths/D/short | failure_target | q3_interval_proxy | 3 | 5 | 8 |

## Paper-Safe Interpretation

- Safe now: DRCR is a calibration-selected, LTT/CRC-style empirical risk-screened external repair layer that improves the current locked stress/positive-control suite.
- Not safe yet: DRCR has a strict distribution-free finite-sample guarantee for dependent time-series windows.
- Safe now: objective baselines show the method is not merely pure classical fallback, global blending, or a single threshold gate.
- Not safe yet: broad universal TSFM reliability claims across all GIFT-Eval/Monash/ETT/M4/M5 datasets.

## Generated Artifacts

- `results/aaai_stress/aaai_oral_requirement_matrix.csv`
- `results/aaai_stress/benchmark_coverage_table.csv`
- `results/aaai_stress/model_dataset_matrix.csv`
- `results/aaai_stress/q9_full_grid_results.csv`
- `results/aaai_stress/contamination_risk_table.csv`
- `results/aaai_stress/classical_probabilistic_baseline_feasibility.csv`
- `results/aaai_stress/oral_evidence_source_gap_matrix.csv`
- `results/aaai_stress/oral_evidence_family_gap_matrix.csv`
- `results/aaai_stress/oral_rerun_command_manifest.csv`
- `results/aaai_stress/remote_q9_rerun_plan.csv`
- `results/aaai_stress/remote_q9_rerun_completion_audit.csv`
- `results/aaai_stress/remote_q9_ingestion_manifest.csv`
- `docs/remote_q9_rerun_execution_pack.md`
- `docs/remote_q9_rerun_completion_audit.md`
- `docs/remote_q9_ingestion_manifest.md`
- `results/aaai_stress/rerun_manifests/`
- `docs/aaai_oral_goal_status.md`
