# Benefit-Selective DRCR External Confirmation Protocol

Date frozen: 2026-07-11

## Purpose

The development result shows strong source/domain routing but only modest extra value over controls that preserve per-source intervention counts. This protocol therefore tests whole-domain transfer with no labels from the external domains. It is the required experiment for upgrading the method from an internal prototype to a paper-facing selective-repair claim.

## Frozen Method

- Candidate: `benefit_lcb_leaf20_q40_noleak16`.
- Native is the default action.
- Fixed actions: point-only DRCR and frozen EP1.25/cap1.10 DRCR.
- Fixed 16-feature external contract: `configs/benefit_selective_drcr_external_protocol.json`; realized-target `flatness_score` is excluded.
- No feature, threshold, action, or candidate change is allowed after an external outcome is inspected.

## Outcome-Blind Domain Sample

The sampler reads only dataset names and broad domain labels, excludes every development base dataset, SHA256-ranks names with the frozen seed, and selects two different base datasets per broad domain. It does not read benchmark error or failure fields for selection.

The source catalog contains historical aggregate outcomes. Those columns are forbidden to the sampler, but this is algorithmically outcome-blind domain selection rather than a claim that researchers never had access to catalog outcomes.

| Broad domain | Dataset 1 | Dataset 2 |
| --- | --- | --- |
| Econ/Fin | `m4_yearly/A/short` | `m4_monthly/M/short` |
| Energy | `ett1/W/short` | `electricity/15T/short` |
| Healthcare | `hospital/M/short` | `us_births/M/short` |
| Nature | `kdd_cup_2018/H/medium` | `temperature_rain/D/short` |
| Sales | `restaurant/D/short` | `car_parts/M/short` |
| Transport | `m_dense/H/short` | `sz_taxi/15T/short` |
| Web/CloudOps | `bizitobs_l2c/5T/medium` | `bitbrains_fast_storage/5T/short` |

This gives **14 data domains**, three model families, and up to **672 source-specific windows**. Repeated model evaluations of one target window are paired measurements, not new domains.

## Baselines

Native TSFM, deterministic rolling-selected classical reference, always repair, fixed 50/50 blend, matched random gate, history-only HCR gate, Benefit-LCB, and oracle. Every method uses the same q9 grid and forecast cases. The deterministic classical expert is not described as a calibrated probabilistic baseline.

## Co-Primary Pass/Fail Rules

1. Overall domain-macro q9-WQL paired 95% CI upper bound is at most zero.
2. Ex-ante low-structure stratum q9-WQL paired 95% CI upper bound is at most zero.
3. Structured-complement q9-WQL harm upper bound is at most 0.002 and coverage-gap harm upper bound is at most 0.02.
4. Structured-complement intervention is at most 25%.
5. Benefit-LCB outperforms always repair, fixed blend, and intervention-matched random gate.
6. Leave-one-domain-out sensitivity does not reverse the effect direction.

Failure of any co-primary rule blocks a broad safe-repair claim. The paper must then remain a scoped reliability/limits study or redesign the selector before collecting another untouched endpoint.

## Statistical Unit

Primary inference is hierarchical and paired: domain, then series, then non-overlapping forecast origin. Domain-level sign-flip and per-domain forest plots accompany the hierarchical bootstrap. Model/configuration sources are not treated as independent domains.

Execution manifest: `results/aaai_stress/benefit_selective_external_manifest.csv`. Frozen machine-readable protocol: `configs/benefit_selective_drcr_external_protocol.json`.
