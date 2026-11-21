"""Build assistant-only SFT examples from successful AppWorld training rollouts."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from run_lesson import HEADERS, ROOT, setup_env


def load_trajectory(path: Path, playbook: str, header: str) -> dict:
    """Recover the final chat history and remove the teacher's static playbook."""
    with path.open() as log:
        calls = (line for line in log if line.strip())
        first = last = next(calls, None)
        for last in calls:
            pass
    if last is None:
        raise ValueError(f"Empty model-call log: {path}")

    n_demo = sum(m["role"] == "assistant" for m in json.loads(first)["input"]["messages"])
    call = json.loads(last)
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in call["input"]["messages"]
    ]
    final = call["output"]["choices"][0]["message"]["content"] or ""
    messages.append({"role": "assistant", "content": final})

    if playbook:
        note = header + playbook.strip() + "\n"
        users = [m for m in messages if m["role"] == "user" and note in m["content"]]
        if len(users) != 1 or users[0]["content"].count(note) != 1:
            raise ValueError(f"Expected exactly one injected playbook in {path}")
        users[0]["content"] = users[0]["content"].replace(note, "", 1)
        if any(header.strip() in (m["content"] or "") for m in messages):
            raise ValueError(f"Found a remaining playbook header in {path}")

    return {"messages": messages, "n_demo_assistant": n_demo}


def collect_examples(root: Path, condition: str, runs: list[str], train_ids: set[str],
                     playbook: str, header: str) -> list[dict]:
    """Collect successful training trajectories from the requested rollout labels."""
    rows = []
    for run in runs:
        label = f"{condition}_{run}"
        records = root / "experiments" / "gate1" / label
        if not records.is_dir():
            raise ValueError(f"No result directory for {label}")
        for file in sorted(records.glob("*.json")):
            record = json.loads(file.read_text())
            task_id = record["task_id"]
            if not record["success"] or task_id not in train_ids:
                continue
            experiment = record.get("selected_exp", f"spine/gate1/{label}")
            if experiment not in (f"spine/gate1/{label}", f"spine/gate1/{label}_retry"):
                raise ValueError(f"Unexpected selected experiment in {file}")
            outputs = root / "experiments" / "outputs" / experiment
            log = outputs / "tasks" / task_id / "logs" / "lm_calls.jsonl"
            rows.append({"src": label, "task_id": task_id, **load_trajectory(log, playbook, header)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cond", required=True)
    parser.add_argument("--runs", required=True, help="Comma-separated rollout labels")
    parser.add_argument("--lesson", type=Path)
    parser.add_argument("--header", choices=HEADERS, default="playbook")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, help="Sample this many successes for a matched control")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = list(dict.fromkeys(args.runs.split(",")))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in [args.cond, *runs]):
        parser.error("--cond and --runs must contain only letters, numbers, underscores or hyphens")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    setup_env()
    from appworld import load_task_ids

    lesson = args.lesson.read_text().strip() if args.lesson else ""
    if args.lesson and not lesson:
        parser.error("the supplied lesson is empty")
    rows = collect_examples(ROOT, args.cond, runs, set(load_task_ids("train")), lesson, HEADERS[args.header])
    if not rows:
        parser.error("no successful training trajectories found")
    if args.sample_size is not None:
        if len(rows) < args.sample_size:
            parser.error(f"requested {args.sample_size} samples, found {len(rows)}")
        rows = random.Random(args.seed).sample(rows, args.sample_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"examples": len(rows), "tasks": len({row["task_id"] for row in rows})}))


if __name__ == "__main__":
    main()
