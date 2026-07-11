#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path('.')
jobs = list(csv.DictReader((root / 'results/aaai_stress/benefit_selective_external_execution_jobs.csv').open()))
assert len(jobs) == 42, len(jobs)
assert len({row['dataset'] for row in jobs}) == 14
assert {row['family'] for row in jobs} == {'chronos', 'moirai', 'timesfm'}
assert all(int(row['windows_requested']) == 16 for row in jobs)
assert all(int(row['series_requested']) == 16 for row in jobs)
assert all(int(row['windows_per_series']) == 1 for row in jobs)
assert all(int(row['context_cap']) == 512 for row in jobs)
assert all('rolling_pre_origin' in row['command'] for row in jobs)
assert all('best_simple' not in row['command'] for row in jobs)
assert all('q10,q20,q30,q40,q50,q60,q70,q80,q90' == row['quantile_grid'] for row in jobs)
assert all(row['status'] == 'ready_not_run' for row in jobs)
assert all('--num-samples 100' in row['command'] for row in jobs if row['family'] == 'moirai')
assert all(int(row['moirai_samples']) == 100 for row in jobs if row['family'] == 'moirai')
hub_names = {row['dataset_name']: row['hub_dataset_name'] for row in jobs}
assert hub_names['car_parts'] == 'car_parts_with_missing'
assert hub_names['kdd_cup_2018'] == 'kdd_cup_2018_with_missing'
assert hub_names['m_dense'] == 'M_DENSE'
assert hub_names['temperature_rain'] == 'temperature_rain_with_missing'

protocol = json.loads((root / 'configs/benefit_selective_drcr_external_protocol.json').read_text())
evaluation = protocol['evaluation']
method = protocol['method']
assert evaluation['context_cap'] == 512
assert evaluation['classical_validation_folds'] == 3
assert evaluation['execution_policy'] == 'strictly_serial_one_model_process_per_job'
assert 'before_external_outcomes' in protocol['execution_contract_frozen_at']
assert method['development_candidate_id'] == 'benefit_lcb_leaf20_q40'
assert method['min_samples_leaf'] == 20
assert method['residual_quantile'] == 0.4
assert 'flatness_score' not in method['features']
assert len(method['features']) == 16
assert method['base_cpr_policy']['min_active'] == 2
assert method['smooth_interval_head']['low_scale'] == 0.906
assert protocol['low_structure_taxonomy']['minimum_active_factors'] == 2

doc = (root / 'docs/benefit_selective_external_execution_pack.md').read_text()
assert 'one model process at a time' in doc
assert 'pre-origin rolling validation' in doc
print('[external-execution-pack critic] PASS: 42 locked serial jobs use q9 and non-leaky pre-origin baseline selection')
PY
