"""Run a dev-only two-stage RSI gate; never dispatch the held-out split."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from rsi_guards import RSI_RULES

ROOT = Path(__file__).resolve().parents[1]
DEV_REFERENCES = ('v4_dev_lora_v4_r1', 'v4_dev_lora_autodoc_full_r1', 'v4fix_dev_lora_r1')
FAIL_REFERENCES = ('v4_dev_fail25_lora_v4_r1', 'v4fix_dev_fail25_lora_r1')


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def load_records(folder):
    rows = [json.loads(p.read_text()) for p in folder.glob('*.json')]
    mapping = {row['task_id']: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f'Duplicate tasks in {folder}')
    return mapping


def load_splits(root):
    dev = (root / 'data/datasets/dev.txt').read_text().splitlines()
    failed = json.loads((root / 'dev_fail25.json').read_text())
    if (len(dev) != 57 or len(set(dev)) != 57 or not isinstance(failed, list) or len(failed) != 25
            or len(set(failed)) != 25 or not set(failed) <= set(dev)
            or any(not re.fullmatch(r'[0-9a-f]+_\d', task) for task in failed)):
        raise ValueError('Expected the complete 57-task dev split and 25 unique dev failures')
    return {'dev': dev, 'dev_fail25': failed}


def make_case(split, variant):
    if split not in ('dev_fail25', 'dev') or variant not in ('core', 'autodoc'):
        raise ValueError('Only dev_fail25/dev core/autodoc cases are permitted')
    return {'stage': 1 if split == 'dev_fail25' else 2, 'split': split, 'variant': variant,
            'condition': f'v5_{split}_lora_{variant}'}


def build_command(root, case, splits):
    if case['split'] not in ('dev_fail25', 'dev'):
        raise ValueError('Held-out evaluation is not permitted by this gate')
    tasks = ','.join(splits['dev_fail25']) if case['split'] == 'dev_fail25' else 'dev'
    return [sys.executable, str(root / 'scripts/run_lesson.py'), '--cond', case['condition'],
            '--tasks', tasks, '--runs', 'r1', '--np', '8', '--model', 'ace20',
            '--injection-mode', 'none', '--guardrails', 'full', '--retry-mode', 'none',
            '--max-tokens', '3000', '--auto-doc', 'none' if case['variant'] == 'core' else 'full',
            '--rsi-guards', 'v5']


def select_dev_variants(core_passed, autodoc_passed, policy='auto'):
    winner = 'core' if core_passed >= autodoc_passed else 'autodoc'
    # Predeclared dev-only choice: a gap of three tasks also validates the other
    # configuration on the complete dev split; a tie always prefers less doc noise.
    both = policy == 'both' or (policy == 'auto' and abs(core_passed - autodoc_passed) >= 3)
    return [winner, 'autodoc' if winner == 'core' else 'core'] if both else [winner]


def compare(current, reference):
    common = sorted(current.keys() & reference.keys())
    up = [task for task in common if current[task]['success'] and not reference[task]['success']]
    down = [task for task in common if not current[task]['success'] and reference[task]['success']]
    return {'reference_passed': sum(row['success'] for row in reference.values()),
            'compared': len(common), 'rescued': up, 'regressed': down, 'net_gain': len(up) - len(down)}


def summarize_case(root, case, splits):
    expected = set(splits[case['split']])
    rows = load_records(root / 'experiments/gate1' / (case['condition'] + '_r1'))
    if not rows.keys() <= expected:
        raise ValueError(f'Unexpected tasks in {case["condition"]}')
    refs = FAIL_REFERENCES if case['split'] == 'dev_fail25' else DEV_REFERENCES
    comparisons = {}
    for label in refs:
        ref = load_records(root / 'experiments/gate1' / label)
        if ref.keys() != expected:
            raise ValueError(f'Reference {label} must have exactly the same task set')
        comparisons[label] = compare(rows, ref)
    totals = {rule: sum(row.get('rsi_triggers', {}).get(rule, 0) for row in rows.values()) for rule in RSI_RULES}
    clusters = {}
    for rule in RSI_RULES:
        triggered = sorted(task for task, row in rows.items() if row.get('rsi_triggers', {}).get(rule, 0))
        clusters[rule] = {'triggered': triggered, 'passed': [t for t in triggered if rows[t]['success']],
                          'failed': [t for t in triggered if not rows[t]['success']], 'by_reference': {}}
        for label, delta in comparisons.items():
            clusters[rule]['by_reference'][label] = {
                'rescued': [t for t in delta['rescued'] if t in triggered],
                'regressed': [t for t in delta['regressed'] if t in triggered],
            }
    passed = sum(row['success'] for row in rows.values())
    return {**case, 'completed': len(rows), 'expected': len(expected), 'passed': passed,
            'complete': rows.keys() == expected, 'tgc': 100 * passed / len(expected) if rows.keys() == expected else None,
            'runner_errors': sum(bool(row.get('error')) for row in rows.values()),
            'auto_doc_injections': sum(row.get('guardrails', {}).get('auto_doc_injections', 0) for row in rows.values()),
            'rsi_totals': totals, 'clusters': clusters, 'comparisons': comparisons}


def publish(root, cases, splits, status, error=None):
    results = [summarize_case(root, case, splits) for case in cases]
    dev = [row for row in results if row['stage'] == 2 and row['complete']]
    best = max(dev, key=lambda row: (row['passed'], row['variant'] == 'core'), default=None)
    gate = {'decision': 'PENDING', 'recommend_heldout_three_runs': False}
    if status == 'COMPLETE' and best:
        primary = best['comparisons']['v4fix_dev_lora_r1']
        associated = sorted({t for cluster in best['clusters'].values()
                             for t in cluster['by_reference']['v4fix_dev_lora_r1']['regressed']})
        baseline = max(c['reference_passed'] for c in best['comparisons'].values())
        keep = best['passed'] > baseline and not associated
        gate = {'decision': 'KEEP' if keep else 'HOLD_FOR_DEV_REVIEW',
                'best_condition': best['condition'], 'net_gain_vs_v4fix': primary['net_gain'],
                'rsi_associated_regressions': associated, 'recommend_heldout_three_runs': keep,
                'reason': 'Gain above all references with no RSI-associated regressions' if keep else
                          'No gain above the strongest reference, or RSI-associated regressions require dev review'}
    report = {'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': status, 'error': error,
              'scope': ['dev_fail25', 'dev'], 'heldout_run': False, 'results': results, 'gate': gate,
              'attribution_note': 'Rule-cluster rescue/regression is co-occurrence, not causal proof; actor outputs can vary.'}
    atomic_json(root / 'v5_summary.json', report)
    lines = ['# V5 RSI dev gate', '', f'Status: {status}', '', f'Updated: {report["updated_at"]}', '',
             'Scope: dev_fail25 and dev only. No held-out evaluation is scheduled.', '',
             '| Stage | Condition | Progress | Passed | TGC | Auto-doc |', '| --- | --- | --- | --- | --- | --- |']
    for row in results:
        tgc = f'{row["tgc"]:.2f}%' if row['tgc'] is not None else 'pending'
        lines.append(f'| {row["stage"]} | {row["condition"]} | {row["completed"]}/{row["expected"]} | {row["passed"]} | {tgc} | {row["auto_doc_injections"]} |')
    lines += ['', '## Rule hits', '', '| Condition | ' + ' | '.join(RSI_RULES) + ' |',
              '| --- | ' + ' | '.join('---' for _ in RSI_RULES) + ' |']
    for row in results:
        lines.append('| ' + row['condition'] + ' | ' + ' | '.join(str(row['rsi_totals'][rule]) for rule in RSI_RULES) + ' |')
    for row in results:
        lines += ['', '## ' + row['condition'], '']
        for label, delta in row['comparisons'].items():
            lines += [f'- Reference `{label}` ({delta["reference_passed"]}/{row["expected"]}): '
                      f'rescued {len(delta["rescued"])}, regressed {len(delta["regressed"])}, net {delta["net_gain"]:+d}.',
                      '  Rescued: ' + (', '.join(delta['rescued']) or 'none') + '.',
                      '  Regressed: ' + (', '.join(delta['regressed']) or 'none') + '.']
        lines += ['', 'Cluster outcomes (association, not causal attribution):', '']
        primary = 'v4fix_dev_fail25_lora_r1' if row['stage'] == 1 else 'v4fix_dev_lora_r1'
        for rule, cluster in row['clusters'].items():
            changes = cluster['by_reference'][primary]
            lines.append(f'- `{rule}`: triggered {len(cluster["triggered"])} tasks; rescued '
                         + (', '.join(changes['rescued']) or 'none') + '; regressed '
                         + (', '.join(changes['regressed']) or 'none') + '.')
    lines += ['', '## Keep-or-revert recommendation', '', json.dumps(gate, ensure_ascii=False), '', report['attribution_note'], '']
    tmp = root / 'v5_summary.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(root / 'v5_summary.md')
    return report


def other_evaluations():
    output = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    return [line.strip() for line in output.splitlines() if 'run_lesson.py' in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dev-policy', choices=('auto', 'winner', 'both'), default='auto')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    splits = load_splits(root)
    cases = [make_case('dev_fail25', variant) for variant in ('core', 'autodoc')]
    if args.dry_run:
        print(json.dumps({'stage1': [build_command(root, case, splits) for case in cases],
                          'stage2': 'Winner on dev; tie -> core; auto policy also runs both if gap >= 3 tasks'}, indent=2))
        return
    if args.summary_only:
        manifest = json.loads((root / 'artifacts/rsi_gate/manifest.json').read_text())
        print(json.dumps(publish(root, manifest['cases'], splits, manifest['status']), ensure_ascii=False))
        return
    output = root / 'artifacts/rsi_gate'
    output.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if other_evaluations():
            raise RuntimeError('Another evaluation is active; do not overlap GPU experiments')
        with urlopen('http://127.0.0.1:8000/v1/models', timeout=10) as response:
            models = json.load(response)
        if not {row['id'] for row in models['data']} >= {'ace20', 'qwen3.5-9b'}:
            raise ValueError('Required served models are not available')
        names = ['scripts/harness_modules.py', 'scripts/rsi_guards.py', 'scripts/run_lesson.py',
                 'scripts/run_rsi_gate.py', 'configs/qwen3.5-9b.jsonnet', 'dev_fail25.json']
        manifest = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'dev_policy': args.dev_policy,
                    'cases': cases, 'status': 'STARTING', 'served_models': models,
                    'sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
        path = output / 'manifest.json'
        if path.exists():
            raise ValueError('An RSI gate manifest already exists; preserve it and use a separate run root')
        atomic_json(path, manifest)
        source = output / 'source'
        for name in names:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())

        def run_case(case):
            if other_evaluations():
                raise RuntimeError('Another evaluation started; pausing new launches')
            state = f'RUNNING_STAGE_{case["stage"]}_{case["variant"]}'
            manifest['status'] = state
            atomic_json(path, manifest)
            publish(root, cases, splits, state)
            command = build_command(root, case, splits)
            print(time.strftime('%H:%M:%S'), state, flush=True)
            with (root / 'logs' / (case['condition'] + '.log')).open('a') as log:
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                                'APPWORLD_ROOT': str(root), 'APPWORLD_CACHE': str(root / '.cache/appworld'),
                                                'APPWORLD_MODEL_URL': 'http://127.0.0.1:8000/v1'})
                while process.poll() is None:
                    time.sleep(30)
                    publish(root, cases, splits, state)
            result = summarize_case(root, case, splits)
            if process.returncode or not result['complete'] or result['runner_errors']:
                raise RuntimeError(f'Incomplete or failed run: {case["condition"]}')
            publish(root, cases, splits, state + '_COMPLETE')
            return result

        try:
            a = run_case(cases[0])
            b = run_case(cases[1])
            variants = select_dev_variants(a['passed'], b['passed'], args.dev_policy)
            manifest['selected_dev_variants'] = variants
            print('Stage 1 selection:', variants, 'scores', a['passed'], b['passed'], flush=True)
            cases.extend(make_case('dev', variant) for variant in variants)
            for case in cases[2:]:
                run_case(case)
            manifest['status'] = 'COMPLETE'
            atomic_json(path, manifest)
            final = publish(root, cases, splits, 'COMPLETE')
            (output / 'COMPLETE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            print('RSI DEV GATE COMPLETE', json.dumps(final['gate']), flush=True)
        except Exception as error:
            manifest['status'] = 'FAILED'
            atomic_json(path, manifest)
            publish(root, cases, splits, 'FAILED', repr(error))
            raise


if __name__ == '__main__':
    main()
