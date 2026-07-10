# Remote q9 DRCR Repair Evaluation Addendum

This addendum fixes the evaluation hierarchy for the frozen DRCR refresh after the 17 remote q9/full-grid sources are ingested. It distinguishes already-completed locked-suite evidence from the still-pending repair evaluation on the new q9 artifacts.

## Evidence Boundary

- Completed remote work: native q9 forecasts, history sidecars, metrics, and predictor features for 17 sources / 408 windows.
- Completed earlier DRCR work: the same 408 windows entered the locked suite as q3 interval proxies, alongside 668 existing q9 windows.
- Not completed yet: applying the frozen selected DRCR policy to the 17 replacement q9 slugs and rebuilding final-main results.
- Therefore the current remote q9 table is a native-model table, not a Native-vs-DRCR repair table.

## Existing Locked-Suite Evidence

These values predate the remote q9 replacement and must not be presented as results on the new 17 q9 sources.

| Evaluation | Native WQL-RER | Selected DRCR WQL-RER | Relative reduction | Additional read |
| --- | ---: | ---: | ---: | --- |
| Current-suite overall | 0.998 | 0.834 | 16.4% | All 1,076 locked windows; includes 408 q3 proxies. |
| q9 failure test slice | 3.411 | 1.563 | 54.2% | Coverage 76.2% to 82.9%; MAE-RER 3.894 to 1.000. |
| Confirmatory salted overall | 1.040 | 0.874 | 16.0% | Frozen candidates on the separate 526/550 split. |
| Confirmatory salted q9 failure | 2.854 | 1.545 | 45.9% | Direction replicates, but repaired WQL-RER remains above 1. |

Interpretation: DRCR substantially mitigates severe failure, but does not eliminate it. MAE-RER 1.000 means recovery to the classical baseline on the reported median, not recovery to an oracle or irreducible-error limit.

## Frozen Remote-q9 Estimands

The refreshed paper table must make all-window performance the primary endpoint. Failure repair and non-failure safety are secondary, mutually exclusive decompositions of the same 408-window population.

| Scope | Sources | All windows | Calibration | Test | Role in claim |
| --- | ---: | ---: | ---: | ---: | --- |
| Overall | 17 | 408 | 196 | 212 | Primary deployable performance endpoint. |
| Failure target | 7 | 240 | 115 | 125 | Severe-failure repair strength. |
| Non-failure complement | 10 | 168 | 81 | 87 | Aggregate protected-window safety endpoint. |
| Positive control | 7 | 80 | 39 | 41 | Stable-case preservation within the complement. |
| Stress target | 3 | 88 | 42 | 46 | Non-failure stress behavior within the complement. |

The `failure target` row is a pre-specified benchmark role, not an ex-post selection of windows where native TSFM happened to lose. An additional actual-native-failure diagnostic may be reported, but it must be labeled conditional/descriptive and cannot replace the pre-specified partition.

## Required Main Result Table

| Scope | N | Native RelMAE | DRCR RelMAE | Native RelRMSE | DRCR RelRMSE | Native q9-WQL | DRCR q9-WQL | Native coverage | DRCR coverage | WQL harm | Intervention rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 408 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Failure target | 240 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Non-failure complement | 168 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Positive control | 80 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Stress target | 88 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Do not fill these cells by copying the earlier locked-suite numbers. They must be generated after source replacement while keeping `drcr_expert_pull_1.25_cap_1.10`, thresholds, split rules, and policies frozen.

## Aggregation And Inference

- Report a micro overall view over all eligible forecast points/windows. This estimates total operational loss but can be dominated by large sources and long horizons.
- Report a macro overall view that first aggregates within each source and then gives the 17 sources equal weight. This prevents the 128-window TimesFM source from overwhelming two- or eight-window sources.
- Use q9-WQL, RelMAE, RelRMSE, AvgRelMAE where defined, MASE with valid counts, q10-q90 coverage, WQL harm, and intervention rate. Raw MAE/RMSE should only be compared within common units/scales.
- Compute paired Native-vs-DRCR deltas on identical windows. Confidence intervals should resample at the series/source level so overlapping horizons are not treated as independent observations.
- Report both failure improvement and complement harm. Overall performance alone can dilute severe repair; failure-only performance can hide damage to already-good forecasts.

## Paper-Safe Claim

The intended hierarchy is:

> Frozen DRCR improves performance over the complete remote-q9 window population; the gain is concentrated in pre-specified failure targets while the mutually exclusive non-failure complement shows limited harm.

Until the pending table is populated, the safe wording remains:

> Earlier locked-suite experiments show substantial DRCR mitigation of severe failure, while the new Moirai/TimesFM q9 artifacts are ready but have not yet undergone the frozen all-window repair refresh.

Machine-readable partition: `results/aaai_stress/remote_q9_drcr_evaluation_partition.csv`.
