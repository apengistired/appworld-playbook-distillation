"""Compare a completed run and diagnose retry selection strictly after policy completion."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path


def records(folder: Path) -> dict[str, dict]:
    rows = [json.loads(path.read_text()) for path in folder.glob('*.json')]
    result = {row['task_id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate task records: {folder}')
    return result


def evaluate_isolated(root: Path, task: str, experiment: str) -> dict:
    # AppWorld caches output databases by task ID rather than experiment name.
    # A fresh interpreter per attempt is essential even for post-hoc diagnostics.
    program = '''import sys,json
from appworld import update_root
update_root(sys.argv[1])
from appworld.evaluator import evaluate_task
result=evaluate_task(sys.argv[2],sys.argv[3],save_report=False).to_dict()
print(json.dumps({'success':bool(result['success']), 'failed_requirements':[x['requirement'] for x in result.get('failures',[])]}))
'''
    result = subprocess.run(
        [str(root / '.venv/bin/python'), '-B', '-c', program, str(root), task, experiment],
        cwd=root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'APPWORLD_ROOT': str(root),
                       'APPWORLD_CACHE': str(root / '.cache/appworld')},
        capture_output=True, text=True, timeout=120, check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--label', required=True)
    parser.add_argument('--split', choices=('dev', 'test_normal'), default='dev')
    parser.add_argument('--reference', action='append', default=[], help='Name=/absolute/path/to/records')
    parser.add_argument('--posthoc', action='store_true', help='Independently score both attempts after all tasks finish')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('Use a simple run label')
    root = args.root.resolve()
    folder = root / 'experiments/gate1' / args.label
    expected = set((root / 'data/datasets' / (args.split + '.txt')).read_text().splitlines())
    current = records(folder)
    if not current.keys() <= expected:
        raise ValueError('Current run contains unexpected task IDs')
    complete = current.keys() == expected
    if args.posthoc and not complete:
        raise ValueError('Post-hoc scoring is allowed only after all policy selections are complete')
    references = {}
    for spec in args.reference:
        name, path = spec.split('=', 1)
        references[name] = records(Path(path))
        if references[name].keys() != expected:
            raise ValueError(f'Reference {name} must contain the same complete split')
    comparisons = {}
    for name, previous in references.items():
        common = sorted(current.keys() & previous.keys())
        comparisons[name] = {
            'reference_passed': sum(row['success'] for row in previous.values()),
            'compared': len(common),
            'reference_passed_on_compared': sum(previous[t]['success'] for t in common),
            'current_passed_on_compared': sum(current[t]['success'] for t in common),
            'correct_to_wrong': [t for t in common if previous[t]['success'] and not current[t]['success']],
            'wrong_to_correct': [t for t in common if not previous[t]['success'] and current[t]['success']],
        }
    trace_rows = []
    for path in (folder / 'critic').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            trace_rows.append(json.loads(line))
    retry_pairs = []
    if args.posthoc:
        for task, row in sorted(current.items()):
            if not row.get('retry', {}).get('triggered'):
                continue
            pair = {'task_id': task, 'selected': row['retry']['selected']}
            for number, experiment in ((1, 'spine/gate1/' + args.label),
                                       (2, 'spine/gate1/' + args.label + '_retry')):
                pair[f'pass{number}'] = evaluate_isolated(root, task, experiment)
            if pair['pass' + str(pair['selected'])]['success'] != row['success']:
                raise ValueError(f'Isolated evaluation disagrees with selected record: {task}')
            retry_pairs.append(pair)
            print('Post-hoc', task, 'first=', pair['pass1']['success'], 'retry=', pair['pass2']['success'],
                  'selected=', pair['selected'], flush=True)
    passed = sum(row['success'] for row in current.values())
    summary = {
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'label': args.label,
        'complete': complete, 'completed': len(current), 'expected': len(expected), 'passed': passed,
        'tgc': 100 * passed / len(expected) if complete else None,
        'audited': sum(row.get('retry', {}).get('audited', False) for row in current.values()),
        'retried': sum(row.get('retry', {}).get('triggered', False) for row in current.values()),
        'selected_retry': sum(row.get('retry', {}).get('selected') == 2 for row in current.values()),
        'runner_errors': sum(bool(row.get('error')) for row in current.values()),
        'comparisons': comparisons,
        'critic_statuses': dict(Counter(row['status'] for row in trace_rows)),
        'critic_max_input_tokens': max((row.get('input_tokens', 0) for row in trace_rows), default=0),
        'posthoc_only': args.posthoc,
        'posthoc_retry_pairs': retry_pairs,
    }
    if args.posthoc:
        summary['pass1_passed'] = passed + sum(
            int(pair['pass1']['success']) - int(current[pair['task_id']]['success']) for pair in retry_pairs
        )
        summary['retry_rescues'] = [p['task_id'] for p in retry_pairs
                                   if not p['pass1']['success'] and p['pass2']['success'] and p['selected'] == 2]
        summary['retry_regressions'] = [p['task_id'] for p in retry_pairs
                                       if p['pass1']['success'] and not p['pass2']['success'] and p['selected'] == 2]
        summary['missed_rescues'] = [p['task_id'] for p in retry_pairs
                                    if not p['pass1']['success'] and p['pass2']['success'] and p['selected'] == 1]
        summary['prevented_regressions'] = [p['task_id'] for p in retry_pairs
                                           if p['pass1']['success'] and not p['pass2']['success'] and p['selected'] == 1]
    output = root / 'artifacts/diagnostics' / args.label
    output.mkdir(parents=True, exist_ok=True)
    tmp = output / 'comparison.tmp'
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    tmp.replace(output / 'comparison.json')
    with (output / 'per_task.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task_id', *references, 'current_success', 'retried', 'selected'])
        for task, row in sorted(current.items()):
            writer.writerow([task, *[references[name][task]['success'] for name in references], row['success'],
                             row.get('retry', {}).get('triggered', False), row.get('retry', {}).get('selected', 1)])
    print(json.dumps({k: v for k, v in summary.items() if k != 'posthoc_retry_pairs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
