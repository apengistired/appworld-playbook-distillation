"""Run and score AppWorld ReAct rollouts with an optional playbook or LoRA adapter."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "qwen3.5-9b.jsonnet"
DEFAULT_MODEL = "qwen3.5-9b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
LESSON_HEADER = "\n\nLESSONS LEARNED FROM PAST EXPERIENCE (apply them where relevant):\n"
PLAYBOOK_HEADER = "\n\nPLAYBOOK (strategies, rules and insights curated from past experience; apply the relevant ones):\n"
HEADERS = {"lesson": LESSON_HEADER, "playbook": PLAYBOOK_HEADER}


def setup_env() -> None:
    for k in [k for k, v in os.environ.items() if any(0xDC80 <= ord(c) <= 0xDCFF for c in v)]:
        os.environ.pop(k)
    for k in ("LLM_API_KEY", "OPENAI_API_KEY"):
        os.environ.setdefault(k, "EMPTY")
    os.environ.setdefault("MODEL_SERVER_URL", "unused")
    os.chdir(ROOT)
    from appworld import update_root
    update_root(str(ROOT))


def load_agent_config() -> dict:
    from appworld.common.io import jsonnet_load
    from appworld.common.path_store import path_store
    cfg = jsonnet_load(
        str(CONFIG),
        APPWORLD_EXPERIMENT_PROMPTS_PATH=str(path_store.experiment_prompts),
        APPWORLD_EXPERIMENT_CONFIGS_PATH=str(path_store.experiment_configs),
        APPWORLD_EXPERIMENT_CODE_PATH=str(path_store.experiment_code),
    )
    agent = cfg["config"]["agent"]
    agent["model_config"]["base_url"] = os.environ.get("APPWORLD_MODEL_URL", DEFAULT_BASE_URL)
    return agent


def make_agent(agent_cfg: dict, lesson: str, header: str = LESSON_HEADER, extra: str = ""):
    from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

    class LessonReActCodeAgent(SimplifiedReActCodeAgent):
        def initialize(self, world):
            super().initialize(world)
            if lesson or extra:
                assert self.messages[-1]["role"] == "user"
                c = self.messages[-1]["content"].rstrip()
                if lesson:
                    c += header + lesson.strip() + "\n"
                self.messages[-1]["content"] = c + extra

    kwargs = {k: v for k, v in json.loads(json.dumps(agent_cfg)).items() if k != "type"}
    return LessonReActCodeAgent(**kwargs)


def resolve_tasks(spec: str) -> list[str]:
    from appworld import load_task_ids
    out: list[str] = []
    for s in spec.split(","):
        if s == "train1":
            out += [t for t in load_task_ids("train") if t.endswith("_1")]
        elif re.fullmatch(r"[0-9a-f]+_\d", s):  # 单个任务 id（冒烟测试用）
            out.append(s)
        elif s == "eval":
            out += [t for t in load_task_ids("train") if not t.endswith("_1")] + load_task_ids("dev")
        else:
            out += load_task_ids(s)
    return list(dict.fromkeys(out))


_CFG: dict = {}


def _init(model: str, max_tokens: int, header: str = LESSON_HEADER) -> None:
    setup_env()
    cfg = load_agent_config()
    cfg["model_config"]["name"] = model
    cfg["model_config"]["max_tokens"] = max_tokens
    _CFG["agent"] = cfg
    _CFG["header"] = header


def _job(args: tuple) -> dict:
    cond, run, task_id, lesson = args
    out_file = ROOT / "experiments" / "gate1" / f"{cond}_{run}" / f"{task_id}.json"
    if out_file.exists():
        return json.loads(out_file.read_text())
    from appworld import AppWorld
    from appworld.evaluator import evaluate_task
    exp = f"spine/gate1/{cond}_{run}"
    t0 = time.time()
    err = None
    try:
        make_agent(_CFG["agent"], lesson, _CFG["header"]).solve_tasks(task_ids=[task_id], experiment_name=exp)
    except Exception as e:  # noqa: BLE001  单个任务出错不影响其他任务，按失败记
        err = repr(e)[:500]
    try:
        steps = AppWorld.parse_environment_io_log(experiment_name=exp, task_id=task_id)
    except Exception:  # noqa: BLE001
        steps = []
    ev = evaluate_task(task_id, exp).to_dict()
    fails = ev.get("failures", [])
    ans = [f for f in fails if f["requirement"].startswith("assert answers match")]
    rec = {
        "cond": cond, "run": run, "task_id": task_id, "success": bool(ev["success"]),
        "difficulty": ev.get("difficulty"), "num_tests": ev.get("num_tests"),
        "failed_requirements": [f["requirement"] for f in fails],
        # 标准答案为 null 却交了答案（只记布尔，不存 trace，避免把标准答案带进后续流程）
        "answered_null_task": bool(ans and re.search(r"== 'null'\s*$", ans[0]["trace"].strip())),
        "num_steps": len(steps), "error": err, "seconds": round(time.time() - t0, 1),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    tmp_file.replace(out_file)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--lesson", default="", help="教训文本文件；不填 = 不加教训")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--runs", default="r1,r2,r3")
    ap.add_argument("--np", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--header", choices=list(HEADERS), default="lesson", help="教训段的标题：lesson（关卡 1）/ playbook（ACE）")
    args = ap.parse_args()
    setup_env()
    if args.np < 1:
        ap.error("--np must be positive")
    runs = args.runs.split(",")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in [args.cond, *runs]):
        ap.error("--cond and --runs must contain only letters, numbers, underscores or hyphens")
    lesson = Path(args.lesson).read_text() if args.lesson else ""
    tasks = resolve_tasks(args.tasks)
    jobs = [(args.cond, r, t, lesson) for r in dict.fromkeys(runs) for t in tasks]
    print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: {len(jobs)} jobs, model={args.model}", flush=True)
    ok = n = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.np, initializer=_init, initargs=(args.model, args.max_tokens, HEADERS[args.header]), maxtasksperchild=20) as pool:
        for rec in pool.imap_unordered(_job, jobs):
            n += 1
            ok += rec["success"]
            if n % 20 == 0 or n == len(jobs):
                print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: {n}/{len(jobs)} done, success {ok}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: ALL DONE {ok}/{n}", flush=True)


if __name__ == "__main__":
    main()
