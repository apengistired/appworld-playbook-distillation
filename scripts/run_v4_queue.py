"""Wait for the existing baseline matrix, then run the frozen V4 evaluation plan."""
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

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path('/root/autodl-tmp/appworld-cpu.VFF83j/project')
BASELINE_NAME = 'overnight_20261121'


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def baseline_ready(phase: str, complete: bool, processes: list[str]) -> bool:
    """Require a successful handoff and no remaining baseline workers or controller."""
    last = phase.strip().splitlines()[-1] if phase.strip() else ''
    return bool(re.search(r'\bDONE\b', last) and complete and not processes)


def active_baseline_processes(baseline: Path) -> list[str]:
    result = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, check=True)
    active = []
    for line in result.stdout.splitlines()[1:]:
        if ('run_overnight_matrix.sh' in line or
                ('run_benchmark.py' in line and BASELINE_NAME in line) or
                ('run_lesson.py' in line and str(baseline) in line and BASELINE_NAME in line)):
            active.append(line.strip())
    return active


def failure_ids(baseline: Path) -> list[str]:
    folder = baseline / 'experiments/gate1/v033_v2_dev_lora_full_r1'
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob('*.json'))]
    ids = [row['task_id'] for row in rows if row['success'] is False]
    if len(rows) != 57 or len(ids) != 25 or len(set(ids)) != 25:
        raise ValueError(f'Expected 57 dev records and exactly 25 distinct failures: {len(rows)}, {len(ids)}')
    if any(not re.fullmatch(r'[0-9a-f]+_\d', task) for task in ids):
        raise ValueError('Invalid failure task ID')
    return ids


def experiment_plan(failures: list[str]) -> list[dict]:
    plan = []
    for stage, split, tasks, count in (
        (1, 'dev_fail25', ','.join(failures), 25), (2, 'dev', 'dev', 57),
        (3, 'test_normal', 'test_normal', 168),
    ):
        names = ('lora_autodoc_full', 'lora_v4') if stage == 1 else (
            'lora_autodoc_full', 'lora_v4', 'base_v4',
        )
        for variant in names:
            base = variant == 'base_v4'
            selective = variant != 'lora_autodoc_full'
            condition = f'v4_{split}_{variant}'
            command = [
                sys.executable, str(ROOT / 'scripts/run_lesson.py'), '--cond', condition,
                '--tasks', tasks, '--runs', 'r1', '--np', '8',
                '--model', 'qwen3.5-9b' if base else 'ace20',
                '--injection-mode', 'jit_scoped' if base else 'none', '--guardrails', 'full',
                '--retry-mode', 'selective' if selective else 'none', '--max-tokens', '3000',
            ]
            if base:
                command += ['--lesson', str(ROOT / 'artifacts/playbook_top20.txt')]
            if selective:
                command += ['--retry-trigger', 'risk_only', '--retry-temperature', '0.4',
                            '--critic-model', 'qwen3.5-9b', '--critic-thinking']
            plan.append({'stage': stage, 'split': split, 'condition': condition,
                         'expected': count, 'command': command})
    return plan


def summarize_case(case: dict, root: Path, failures: list[str]) -> dict:
    expected = set(failures) if case['split'] == 'dev_fail25' else set(
        (root / 'data/datasets' / (case['split'] + '.txt')).read_text().splitlines()
    )
    rows = [json.loads(path.read_text()) for path in
            (root / 'experiments/gate1' / (case['condition'] + '_r1')).glob('*.json')]
    ids = [row['task_id'] for row in rows]
    if len(expected) != case['expected'] or len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError(f'Unexpected or duplicate task records for {case["condition"]}')
    complete = set(ids) == expected
    guardrails = {}
    for row in rows:
        for key, count in row.get('guardrails', {}).items():
            guardrails[key] = guardrails.get(key, 0) + count
    passed = sum(row['success'] for row in rows)
    return {key: case[key] for key in ('stage', 'split', 'condition', 'expected')} | {
        'completed': len(rows), 'passed': passed, 'complete': complete,
        'tgc': 100 * passed / len(expected) if complete else None,
        'runner_errors': sum(bool(row.get('error')) for row in rows),
        'audited': sum(bool(row.get('retry', {}).get('audited')) for row in rows),
        'retried': sum(bool(row.get('retry', {}).get('triggered')) for row in rows),
        'selected_retry': sum(row.get('retry', {}).get('selected') == 2 for row in rows),
        'guardrails': guardrails,
    }


def publish(baseline: Path, status: str, plan: list[dict], failures: list[str], error=None) -> dict:
    baseline_report = baseline / 'artifacts/benchmarks' / BASELINE_NAME / 'summary.json'
    old = json.loads(baseline_report.read_text()) if baseline_report.exists() else []
    rows = [summarize_case(case, ROOT, failures) for case in plan]
    summary = {'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': status,
               'error': error, 'baseline_test_normal': old, 'v4': rows}
    write_json(ROOT / 'v4_summary.json', summary)
    text = ['# V4 evaluation', '', f'Status: {status}', '', f'Updated: {summary["updated_at"]}', '',
            'All V4 settings are frozen before test_normal; only runtime evidence informs retry selection.', '',
            '## Existing test_normal baselines (three runs)', '',
            '| Condition | r1 | r2 | r3 | Mean TGC | Sample SD |', '| --- | --- | --- | --- | --- | --- |']
    fmt = lambda value: 'pending' if value is None else f'{value:.2f}%'
    for row in old:
        trials = [f'{run["passed"]}/{run["completed"]} ({run["completed"]}/{run["expected"]} done)'
                  for run in row['runs']]
        text.append('| ' + ' | '.join([row['condition'], *trials, fmt(row['mean_tgc']), fmt(row['std_tgc'])]) + ' |')
    text += ['', '## V4 results (r1)', '',
             '| Stage | Condition | Progress | Passed | TGC | Audited | Retried | Selected retry | Auto-doc | Runner errors |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        text.append('| ' + ' | '.join(map(str, [row['stage'], row['condition'],
            f'{row["completed"]}/{row["expected"]}', row['passed'], fmt(row['tgc']), row['audited'],
            row['retried'], row['selected_retry'], row['guardrails'].get('auto_doc_injections', 0),
            row['runner_errors']])) + ' |')
    text += ['', 'Guardrail totals and complete per-run baseline counts are saved in v4_summary.json.',
             'Per-task records: experiments/gate1/. Full trajectories and sandbox states: experiments/outputs/.',
             'V4 guardrail counters describe the selected attempt; runtime includes audit/retry overhead.', '']
    if error:
        text += ['Error: ' + str(error), '']
    temporary = ROOT / 'v4_summary.md.tmp'
    temporary.write_text('\n'.join(text))
    temporary.replace(ROOT / 'v4_summary.md')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE_ROOT)
    args = parser.parse_args()
    baseline = args.baseline_root.resolve()
    if ROOT == baseline or (ROOT / 'experiments').is_symlink() or (ROOT / 'artifacts').is_symlink():
        raise ValueError('V4 must use its own output directories')
    with (ROOT / '.v4-queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = baseline / 'artifacts/benchmarks' / BASELINE_NAME
        print('WAITING: existing baseline matrix must finish before any V4 model request.', flush=True)
        plan, failures = [], []
        try:
            while True:
                phase = (report / 'launch.phase').read_text() if (report / 'launch.phase').exists() else ''
                active = active_baseline_processes(baseline)
                if baseline_ready(phase, (report / 'COMPLETE').exists(), active):
                    break
                publish(baseline, 'WAITING_FOR_BASELINES', [], [])
                print(time.strftime('%Y-%m-%d %H:%M:%S'), 'WAITING', phase.strip(),
                      f'active baseline processes={len(active)}', flush=True)
                time.sleep(60)
            failures = failure_ids(baseline)
            write_json(ROOT / 'dev_fail25.json', failures)
            plan = experiment_plan(failures)
            with urlopen('http://127.0.0.1:8000/v1/models', timeout=10) as response:
                models = json.load(response)
            if not {m['id'] for m in models['data']} >= {'ace20', 'qwen3.5-9b'}:
                raise ValueError('The unchanged vLLM service is missing required model names')
            manifest = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'plan': plan,
                        'baseline_root': str(baseline), 'served_models': models,
                        'sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                            'scripts/run_lesson.py', 'scripts/harness_modules.py', 'scripts/run_v4_queue.py',
                            'configs/qwen3.5-9b.jsonnet', 'artifacts/playbook_top20.txt')},
                        'dev_fail25': failures}
            saved = ROOT / 'v4_manifest.json'
            if saved.exists():
                prior = json.loads(saved.read_text())
                if any(prior[key] != manifest[key] for key in ('plan', 'sha256', 'dev_fail25')):
                    raise ValueError('Frozen V4 plan/code changed; use a fresh evaluation directory')
            else:
                write_json(saved, manifest)
            (ROOT / 'logs').mkdir(exist_ok=True)
            for case in plan:
                state = f'STAGE_{case["stage"]}: {case["condition"]}'
                publish(baseline, state, plan, failures)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), state, flush=True)
                with (ROOT / 'logs' / (case['condition'] + '.log')).open('a') as log:
                    process = subprocess.Popen(case['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ, 'APPWORLD_MODEL_URL': 'http://127.0.0.1:8000/v1'})
                    while process.poll() is None:
                        time.sleep(60)
                        publish(baseline, state, plan, failures)
                if process.returncode:
                    raise RuntimeError(f'{case["condition"]} exited {process.returncode}')
                if not summarize_case(case, ROOT, failures)['complete']:
                    raise RuntimeError(f'{case["condition"]} exited without every task record')
                publish(baseline, state + ' COMPLETE', plan, failures)
            publish(baseline, 'COMPLETE', plan, failures)
            (ROOT / 'V4_COMPLETE').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            print('V4 COMPLETE; server and vLLM remain running.', flush=True)
        except Exception as error:
            publish(baseline, 'FAILED', plan, failures, repr(error))
            print('V4 FAILED:', repr(error), flush=True)
            raise


if __name__ == '__main__':
    main()
