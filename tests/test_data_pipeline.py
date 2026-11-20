import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_sft import collect_examples
from run_lesson import PLAYBOOK_HEADER


class DataPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
