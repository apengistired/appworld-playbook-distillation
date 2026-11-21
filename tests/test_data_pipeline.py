import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_lesson
from build_sft import collect_examples
from run_lesson import PLAYBOOK_HEADER


class DataPipelineTests(unittest.TestCase):
    def test_rollout_limit_applies_to_each_run(self):
        argv = [
            "run_lesson.py", "--cond", "smoke", "--tasks", "dev", "--runs", "r1,r2", "--limit", "1",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(run_lesson, "setup_env"),
            patch.object(run_lesson, "resolve_tasks", return_value=["first", "second"]),
            patch.object(run_lesson.mp, "get_context") as get_context,
            redirect_stdout(StringIO()),
        ):
            pool_factory = get_context.return_value.Pool
            pool = pool_factory.return_value.__enter__.return_value
            pool.imap_unordered.return_value = [{"success": True}, {"success": False}]
            run_lesson.main()

        self.assertEqual(pool_factory.call_args.args[0], 2)
        pool.imap_unordered.assert_called_once_with(run_lesson._job, [
            ("smoke", "r1", "first", ""),
            ("smoke", "r2", "first", ""),
        ])

    def test_rank_playbook_by_net_helpfulness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "playbook.json"
            target = root / "top.txt"
            source.write_text(json.dumps({
                "next_id": 4,
                "bullets": [
                    {"id": str(i), "section": "strategies_and_hard_rules",
                     "content": text, "helpful": helpful, "harmful": harmful}
                    for i, (text, helpful, harmful) in enumerate([
                        ("Highest net", 6, 2), ("Wins tie", 7, 4), ("Loses tie", 4, 1),
                    ])
                ],
            }))
            subprocess.run(
                [sys.executable, str(SCRIPTS / "playbook_topn.py"), str(source), "2", str(target)],
                check=True, capture_output=True,
            )
            result = target.read_text()
            self.assertLess(result.index("Highest net"), result.index("Wins tie"))
            self.assertNotIn("Loses tie", result)

    def test_keep_successes_on_train_and_remove_playbook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = "teacher_r1"
            records = root / "experiments" / "gate1" / label
            records.mkdir(parents=True)
            for task, success in (("train_success", True), ("train_failure", False), ("dev_success", True)):
                (records / f"{task}.json").write_text(json.dumps({"task_id": task, "success": success}))
            log = root / "experiments" / "outputs" / "spine" / "gate1" / label
            log = log / "tasks" / "train_success" / "logs" / "lm_calls.jsonl"
            log.parent.mkdir(parents=True)
            messages = [
                {"role": "assistant", "content": "Example response"},
                {"role": "user", "content": f"Task{PLAYBOOK_HEADER}Example rule\n"},
            ]
            first = {"input": {"messages": messages}}
            last = {
                "input": {"messages": messages + [{"role": "assistant", "content": "Agent action"}]},
                "output": {"choices": [{"message": {"content": "Agent completion"}}]},
            }
            log.write_text(json.dumps(first) + "\n" + json.dumps(last) + "\n")
            rows = collect_examples(
                root, "teacher", ["r1"], {"train_success", "train_failure"}, "Example rule", PLAYBOOK_HEADER,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_id"], "train_success")
            self.assertEqual(rows[0]["n_demo_assistant"], 1)
            self.assertEqual(rows[0]["messages"][1]["content"], "Task")
            self.assertEqual(rows[0]["messages"][-1]["content"], "Agent completion")
            # Selecting the clean retry must also select its trajectory during SFT export.
            record_path = records / "train_success.json"
            record = json.loads(record_path.read_text())
            record["selected_exp"] = f"spine/gate1/{label}_retry"
            record_path.write_text(json.dumps(record))
            retry_log = root / "experiments/outputs" / record["selected_exp"] / "tasks/train_success/logs/lm_calls.jsonl"
            retry_log.parent.mkdir(parents=True)
            last["output"]["choices"][0]["message"]["content"] = "Selected retry completion"
            retry_log.write_text(json.dumps(first) + "\n" + json.dumps(last) + "\n")
            rows = collect_examples(root, "teacher", ["r1"], {"train_success"}, "Example rule", PLAYBOOK_HEADER)
            self.assertEqual(rows[0]["messages"][-1]["content"], "Selected retry completion")


if __name__ == "__main__":
    unittest.main()
