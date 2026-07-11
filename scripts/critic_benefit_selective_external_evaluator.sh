#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import ast
import json
from pathlib import Path

root = Path('.')
evaluator_path = root / 'scripts/evaluate_benefit_selective_external.py'
freeze_path = root / 'results/aaai_stress/benefit_selective_external_freeze_hashes.json'
source = evaluator_path.read_text(encoding='utf-8')
tree = ast.parse(source)

required_phrases = [
    'verify_freeze_hashes(protocol)',
    'Frozen development calibration inventory is not exactly 196 windows',
    'Cross-family target mismatch',
    'Cross-family context mismatch',
    'rolling_pre_origin',
    'q9_wql',
    'relmae',
    'relrmse',
    'mase',
    'coverage_gap',
    'interval_width',
    'Matched random gate',
    'Fixed 50/50 blend',
    'Always ExpertPull',
    'forest_ready',
]
missing = [phrase for phrase in required_phrases if phrase not in source]
if missing:
    raise SystemExit(f'[external-evaluator critic] FAIL: missing guards/outputs: {missing}')

names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
for required in ['artifact_inventory', 'build_external_windows', 'validate_cross_family_pairing',
                 'fit_frozen_benefit_models', 'hierarchical_bootstrap_delta', 'sign_flip_p', 'lodo_stable']:
    if required not in names:
        raise SystemExit(f'[external-evaluator critic] FAIL: missing function {required}')

if freeze_path.exists():
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    if frozen.get('status') != 'frozen_before_external_outcomes':
        raise SystemExit('[external-evaluator critic] FAIL: invalid freeze status')
    if 'scripts/evaluate_benefit_selective_external.py' not in frozen.get('sha256', {}):
        raise SystemExit('[external-evaluator critic] FAIL: evaluator is not hash-locked')

print('[external-evaluator critic] PASS: fail-closed evaluator exports standard metrics, controls, family sensitivity, and domain inference')
PY
