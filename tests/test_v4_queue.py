import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_v4_queue import baseline_ready, experiment_plan, failure_ids, summarize_case


class V4QueueTests(unittest.TestCase):
    def test_handoff_requires_done_complete_and_no_baseline_processes(self):
        self.assertTrue(baseline_ready('2026-11-21 DONE four conditions', True, []))
        for phase, complete, active in (
            ('RUNNING', True, []), ('DONE', False, []), ('DONE', True, ['run_benchmark.py']),
            ('FAILED', True, []), ('DONE\nRUNNING', True, []), ('', True, []),
        ):
            self.assertFalse(baseline_ready(phase, complete, active))

    def test_plan_has_requested_eight_cases_and_frozen_test_settings(self):
        plan = experiment_plan([f'{i:07x}_1' for i in range(25)])
        self.assertEqual([row['stage'] for row in plan], [1, 1, 2, 2, 2, 3, 3, 3])
        self.assertEqual(sum(row['expected'] for row in plan), 725)
        for row in plan:
            command = row['command']
            self.assertEqual(command[command.index('--np') + 1], '8')
            self.assertEqual(command[command.index('--runs') + 1], 'r1')
            is_base = row['condition'].endswith('base_v4')
            self.assertEqual(command[command.index('--injection-mode') + 1], 'jit_scoped' if is_base else 'none')
            if not row['condition'].endswith('autodoc_full'):
                self.assertEqual(command[command.index('--retry-trigger') + 1], 'risk_only')
                self.assertEqual(command[command.index('--retry-temperature') + 1], '0.4')
                self.assertEqual(command[command.index('--critic-model') + 1], 'qwen3.5-9b')
                self.assertIn('--critic-thinking', command)

    def test_failure_ids_and_incomplete_summary_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'experiments/gate1/v033_v2_dev_lora_full_r1'
            folder.mkdir(parents=True)
            for i in range(57):
                task = f'{i:07x}_1'
                (folder / (task + '.json')).write_text(json.dumps({'task_id': task, 'success': i >= 25}))
            failures = failure_ids(root)
            self.assertEqual(len(failures), 25)
            case = experiment_plan(failures)[0]
            output = root / 'experiments/gate1' / (case['condition'] + '_r1')
            output.mkdir()
            (output / 'one.json').write_text(json.dumps({
                'task_id': failures[0], 'success': True, 'error': None,
                'guardrails': {'auto_doc_injections': 2},
                'retry': {'audited': True, 'triggered': True, 'selected': 2},
            }))
            row = summarize_case(case, root, failures)
            self.assertIsNone(row['tgc'])
            self.assertEqual(row['passed'], 1)
            self.assertEqual(row['selected_retry'], 1)
            self.assertEqual(row['guardrails']['auto_doc_injections'], 2)
            (output / 'duplicate.json').write_text((output / 'one.json').read_text())
            with self.assertRaises(ValueError):
                summarize_case(case, root, failures)
            (folder / '0000000_1.json').unlink()
            with self.assertRaises(ValueError):
                failure_ids(root)


if __name__ == '__main__':
    unittest.main()
