import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rsi_gate import build_command, compare, load_splits, make_case, select_dev_variants, summarize_case, DEV_REFERENCES
from rsi_guards import RSI_RULES


class RSIGateTests(unittest.TestCase):
    def test_tie_prefers_core_and_large_dev_only_gap_validates_both(self):
        self.assertEqual(select_dev_variants(8, 8), ['core'])
        self.assertEqual(select_dev_variants(8, 9), ['autodoc'])
        self.assertEqual(select_dev_variants(7, 10), ['autodoc', 'core'])
        self.assertEqual(select_dev_variants(10, 7, 'winner'), ['core'])

    def test_commands_use_required_actor_settings_and_reject_heldout(self):
        splits = {'dev_fail25': [f'{i:07x}_1' for i in range(25)]}
        for split in ('dev_fail25', 'dev'):
            for variant in ('core', 'autodoc'):
                command = build_command(Path('/tmp/eval'), make_case(split, variant), splits)
                for key, value in (('--runs', 'r1'), ('--np', '8'), ('--model', 'ace20'),
                                   ('--retry-mode', 'none'), ('--injection-mode', 'none'),
                                   ('--guardrails', 'full'), ('--rsi-guards', 'v5'), ('--max-tokens', '3000')):
                    self.assertEqual(command[command.index(key) + 1], value)
                self.assertEqual(command[command.index('--auto-doc') + 1], 'none' if variant == 'core' else 'full')
                self.assertEqual(command[command.index('--tasks') + 1], 'dev' if split == 'dev' else ','.join(splits['dev_fail25']))
        with self.assertRaises(ValueError):
            make_case('test_normal', 'core')
        with self.assertRaises(ValueError):
            build_command(Path('/tmp/eval'), {'split': 'test_normal'}, splits)

    def test_failure_split_is_unique_and_restricted_to_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/datasets').mkdir(parents=True)
            dev = [f'{i:07x}_1' for i in range(57)]
            (root / 'data/datasets/dev.txt').write_text('\n'.join(dev))
            (root / 'dev_fail25.json').write_text(json.dumps(dev[:25]))
            self.assertEqual(load_splits(root)['dev_fail25'], dev[:25])
            (root / 'dev_fail25.json').write_text(json.dumps(dev[:24] + [dev[0]]))
            with self.assertRaises(ValueError):
                load_splits(root)

    def test_summary_reports_regressions_and_rule_associations_without_claiming_causality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['example_a', 'example_b', 'example_c']
            previous = {task: {'task_id': task, 'success': task == 'example_a'} for task in ids}
            current = {task: {'task_id': task, 'success': task != 'example_a', 'error': None,
                              'rsi_triggers': {RSI_RULES[0]: 1}} for task in ids}
            case = make_case('dev', 'core')
            folders = {case['condition'] + '_r1': current, **{name: previous for name in DEV_REFERENCES}}
            for name, rows in folders.items():
                folder = root / 'experiments/gate1' / name
                folder.mkdir(parents=True)
                for task, row in rows.items():
                    (folder / (task + '.json')).write_text(json.dumps(row))
            summary = summarize_case(root, case, {'dev': ids})
            self.assertEqual(summary['passed'], 2)
            self.assertEqual(summary['rsi_totals'][RSI_RULES[0]], 3)
            delta = summary['comparisons']['v4fix_dev_lora_r1']
            self.assertEqual(delta['regressed'], ['example_a'])
            self.assertEqual(delta['rescued'], ['example_b', 'example_c'])
            self.assertEqual(delta['net_gain'], 1)
            self.assertEqual(summary['clusters'][RSI_RULES[0]]['by_reference']['v4fix_dev_lora_r1']['regressed'], ['example_a'])
            self.assertEqual(compare(current, previous), delta)


if __name__ == '__main__':
    unittest.main()
