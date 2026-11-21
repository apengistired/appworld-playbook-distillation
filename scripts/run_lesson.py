"""Run and score AppWorld ReAct rollouts with an optional playbook or LoRA adapter."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.request import Request, urlopen

from harness_modules import (
    GUARDRAIL_MODES,
    RSI_MODES,
    RSI_RULES,
    ExecutionGuardrails,
    PlaybookEntry,
    ScopedPlaybookRouter,
    parse_playbook,
    build_audit_prompt,
    parse_audit_verdict,
    build_pairwise_judge_prompt,
    parse_pairwise_verdict,
    summarize_attempt,
    audit_risk_reasons,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "qwen3.5-9b.jsonnet"
DEFAULT_MODEL = "qwen3.5-9b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
LESSON_HEADER = "\n\nLESSONS LEARNED FROM PAST EXPERIENCE (apply them where relevant):\n"
PLAYBOOK_HEADER = "\n\nPLAYBOOK (strategies, rules and insights curated from past experience; apply the relevant ones):\n"
HEADERS = {"lesson": LESSON_HEADER, "playbook": PLAYBOOK_HEADER}
INJECTION_MODES = ("static", "jit_scoped", "none")
RETRY_NOTE = (
    "\n\nREVIEW OF A PREVIOUS ATTEMPT (the environment has been reset to its initial state):\n"
    "{critique}\nCheck this diagnosis against the task and runtime observations before acting.\n"
)


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


def make_agent(
    agent_cfg: dict,
    lesson: str,
    header: str = LESSON_HEADER,
    extra: str = "",
    *,
    injection_mode: str = "static",
    jit_topk: int = 3,
    guardrails: str = "none",
    auto_doc: str = "full",
    rsi_guards: str = "none",
    playbook_entries: Sequence[PlaybookEntry] | None = None,
):
    """Create a ReAct agent with optional request-time routing and execution checks."""
    from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
    if auto_doc not in ("full", "none") or rsi_guards not in RSI_MODES:
        raise ValueError("Invalid auto_doc or rsi_guards mode")

    entries = ()
    if injection_mode == "jit_scoped":
        entries = tuple(playbook_entries if playbook_entries is not None else parse_playbook(lesson))
    static_lesson = lesson if injection_mode == "static" else ""

    class LessonReActCodeAgent(SimplifiedReActCodeAgent):
        guardrail_state: ExecutionGuardrails | None = None
        _step_hint = ""

        def initialize(self, world):
            super().initialize(world)
            if static_lesson or extra:
                assert self.messages[-1]["role"] == "user"
                content = self.messages[-1]["content"].rstrip()
                if static_lesson:
                    content += header + static_lesson.strip() + "\n"
                self.messages[-1]["content"] = content + extra
            if injection_mode == "jit_scoped":
                self._router = ScopedPlaybookRouter(entries, jit_topk)
                self._task_text = world.task.instruction
                self._last_code = ""
            if guardrails != "none" or rsi_guards != "none":
                self.guardrail_state = ExecutionGuardrails(
                    guardrails, task_text=world.task.instruction,
                    auto_doc=auto_doc == "full", rsi_guards=rsi_guards,
                )
                self._guard_world(world, self.guardrail_state)

        @property
        def trimmed_messages(self):
            messages = super().trimmed_messages
            if not self._step_hint:
                return messages
            last = messages[-1]
            assert last["role"] == "user"
            return [
                *messages[:-1],
                {**last, "content": last["content"] + "\n\n" + self._step_hint},
            ]

        def next_execution_inputs_usage_and_status(self, last_execution_outputs):
            if injection_mode != "jit_scoped":
                return super().next_execution_inputs_usage_and_status(last_execution_outputs)
            last_obs = last_execution_outputs[0].content if last_execution_outputs else ""
            rules = self._router.select_rules(self._task_text, self._last_code, last_obs)
            self._step_hint = self._router.format_hint(rules)
            try:
                inputs, usage, status = super().next_execution_inputs_usage_and_status(last_execution_outputs)
            finally:
                self._step_hint = ""
            self._last_code = inputs[0].content if inputs else ""
            return inputs, usage, status

        @staticmethod
        def _guard_world(world, state: ExecutionGuardrails) -> None:
            execute = world.batch_execute

            def guarded_batch_execute(codes: list[str]) -> list[str]:
                if not codes:
                    return execute(codes)
                # The pinned ReAct agent emits one ExecutionIO per decision.
                (code,) = codes
                intercepted = state.before_execute(code)
                if intercepted is not None:
                    return [intercepted]
                (observation,) = execute(codes)
                nudge = state.after_execute(code, observation)
                # ApiCollection uses the existing task requester; this never calls world.execute
                # or advances the agent/environment step counters.
                docs = state.auto_doc_on_error(code, observation, world.apis.api_docs) if (
                    state.auto_doc and getattr(world, "apis", None) is not None
                ) else None
                additions = [item for item in (nudge, docs) if item]
                return ["\n\n".join([observation.rstrip(), *additions])] if additions else [observation]

            world.batch_execute = guarded_batch_execute

    kwargs = {k: v for k, v in json.loads(json.dumps(agent_cfg)).items() if k != "type"}
    return LessonReActCodeAgent(**kwargs)


def resolve_tasks(spec: str) -> list[str]:
    from appworld import load_task_ids
    out: list[str] = []
    for s in spec.split(","):
        if s == "train1":
            out += [t for t in load_task_ids("train") if t.endswith("_1")]
        elif re.fullmatch(r"[0-9a-f]+_\d", s):  # Explicit task ID for targeted runs.
            out.append(s)
        elif s == "eval":
            out += [t for t in load_task_ids("train") if not t.endswith("_1")] + load_task_ids("dev")
        else:
            out += load_task_ids(s)
    return list(dict.fromkeys(out))


_CFG: dict = {}


def _init(model: str, max_tokens: int, header: str = LESSON_HEADER, options: dict | None = None) -> None:
    setup_env()
    cfg = load_agent_config()
    cfg["model_config"]["name"] = model
    cfg["model_config"]["max_tokens"] = max_tokens
    if options:
        # An AppWorld completion marker alone cannot restore the task-local guardrail counters.
        cfg["skip_if_finished"] = False
    if (options or {}).get("retry_mode") == "selective":
        cfg.setdefault("appworld_config", {})["load_ground_truth"] = False
    _CFG["agent"] = cfg
    _CFG["header"] = header
    _CFG["options"] = options or {}


def _check_run_settings(label: str, settings: dict | None) -> None:
    """Keep opt-in harness results and AppWorld completion markers in their own run labels."""
    directory = ROOT / "experiments" / "gate1" / label
    output = ROOT / "experiments" / "outputs" / "spine" / "gate1" / label
    manifest = directory / ".harness-config"
    if manifest.exists():
        if json.loads(manifest.read_text()) == settings:
            return
    elif settings is None:
        return
    elif not output.exists() and not any(directory.glob("*.json")):
        directory.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as file:
            json.dump(settings, file, indent=2)
        return
    raise ValueError(f"Run {label!r} already has different settings. Use a fresh --cond or --runs label.")


def _run_attempt(task_id: str, exp: str, lesson: str, extra: str = "", *, retry: bool = False) -> dict:
    """Finish and close one sandbox before the next attempt or evaluation."""
    from appworld import AppWorld
    options = _CFG["options"]
    selective = options.get("retry_mode") == "selective"
    agent_options = {k: v for k, v in options.items() if k in (
        "injection_mode", "jit_topk", "guardrails", "playbook_entries", "auto_doc", "rsi_guards",
    )}
    cfg = copy.deepcopy(_CFG["agent"])
    if retry:
        cfg["model_config"]["temperature"] = options.get("retry_temperature", 0.4)
        cfg["model_config"]["seed"] = cfg["model_config"].get("seed", 100) + 1
    err = None
    agent = None
    if selective:
        AppWorld.close_all()
    try:
        agent = make_agent(cfg, lesson, _CFG["header"], extra, **agent_options)
        agent.solve_tasks(task_ids=[task_id], experiment_name=exp)
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:500]
    finally:
        if selective:
            AppWorld.close_all()
    try:
        steps = AppWorld.parse_environment_io_log(experiment_name=exp, task_id=task_id)
    except Exception:  # noqa: BLE001
        steps = []
    state = getattr(agent, "guardrail_state", None)
    result = {
        "exp": exp, "steps": steps, "error": err,
        "guardrails": (state or ExecutionGuardrails()).stats(),
        "num_agent_steps": getattr(agent, "step_number", 0),
    }
    if state is not None and options.get("rsi_guards", "none") != "none":
        result["rsi_triggers"] = state.rsi_stats()
        result["rsi_hits"] = state.rsi_hits
    return result


def _critic_prompt_size(base_url: str, model: str, messages: list[dict], template: dict) -> tuple[int, int]:
    """Use the same vLLM chat template to count input tokens without generation."""
    request = Request(
        base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        data=json.dumps({"model": model, "messages": messages, "chat_template_kwargs": template}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")},
    )
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    count, maximum = data.get("count"), data.get("max_model_len")
    if type(count) is not int or type(maximum) is not int or count < 0 or maximum < 1:
        raise ValueError("Invalid tokenizer context metadata")
    return count, maximum


def _critic(prompt: str) -> str:
    """Make one bounded request to the configured local critic endpoint."""
    from openai import OpenAI
    cfg = _CFG["agent"]["model_config"]
    client = OpenAI(
        base_url=cfg["base_url"], api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        timeout=180, max_retries=0,
    )
    max_tokens = 8192
    model = _CFG["options"].get("critic_model") or DEFAULT_MODEL
    messages = [{"role": "user", "content": prompt}]
    extra = {**cfg.get("extra_body", {}), "chat_template_kwargs": {
        **cfg.get("extra_body", {}).get("chat_template_kwargs", {}),
        "enable_thinking": _CFG["options"].get("critic_thinking", True),
    }}
    trace = {"phase": _CFG.get("critic_phase", "audit"), "model": model,
             "prompt": prompt, "response": "", "max_tokens": max_tokens, "status": "request_failed"}
    started = time.time()
    try:
        count, maximum = _critic_prompt_size(cfg["base_url"], model, messages, extra["chat_template_kwargs"])
        trace.update(input_tokens=count, max_model_len=maximum)
        if count + max_tokens + 32 > maximum:
            trace["status"] = "context_exceeded"
            print(f"Critic context exceeded: {count} input tokens; preserving evidence and Pass 1", flush=True)
            return ""
        reply = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0, seed=cfg.get("seed", 100), max_tokens=max_tokens,
            extra_body=extra,
        )
        choice = reply.choices[0]
        content = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)
        trace.update(response=content, finish_reason=finish, status="complete")
        if finish == "length":
            trace["status"] = "output_truncated"
            print(f"Critic output truncated at max_tokens={max_tokens}", flush=True)
            return ""
        return content
    except Exception as error:  # A critic outage preserves the first attempt.
        trace["error_type"] = type(error).__name__
        print(f"Critic request failed: {type(error).__name__}", flush=True)
        return ""
    finally:
        client.close()
        trace["seconds"] = round(time.time() - started, 3)
        path = _CFG.get("critic_trace")
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")


def _judge_attempts(instruction: str, first: dict, second: dict) -> tuple[int, dict]:
    """Switch to retry only on cited evidence and order-independent agreement."""
    _CFG["critic_phase"] = "judge_forward"
    forward = parse_pairwise_verdict(_critic(build_pairwise_judge_prompt(instruction, first, second)), first, second)
    selection = {"method": "evidence_order_consensus", "forward": forward}
    if not forward["verified"] or forward["choice"] != 2:
        return 1, selection
    _CFG["critic_phase"] = "judge_reverse"
    reverse = parse_pairwise_verdict(_critic(build_pairwise_judge_prompt(instruction, second, first)), second, first)
    selection["reverse"] = reverse
    return (2 if reverse["verified"] and reverse["choice"] == 1 else 1), selection


def _select_attempt(task_id: str, exp: str, lesson: str, first: dict) -> tuple[dict, dict]:
    """Audit and choose from execution evidence before the evaluator is imported or called."""
    from appworld.task import Task

    task = Task.load(task_id=task_id, load_ground_truth=False)
    try:
        instruction = task.instruction
    finally:
        task.close()
    summary = summarize_attempt(first["steps"], first["guardrails"], first["error"])
    risks = audit_risk_reasons(instruction, summary)
    if _CFG["options"].get("retry_trigger", "risk_only") == "risk_only" and not risks:
        return first, {"audited": False, "triggered": False, "selected": 1, "critique": ""}
    if first["error"] or summary["final_status"] == "fail":
        reason = first["error"] or (
            "The previous attempt explicitly submitted status='fail'. "
            "Check the search source, pagination, and API fields before trying a different plan."
        )
        verdict = {"verdict": "RETRY", "critique": reason}
    else:
        _CFG["critic_phase"] = "audit"
        verdict = parse_audit_verdict(_critic(build_audit_prompt(instruction, summary)))
    retry = {
        "audited": True, "triggered": verdict["verdict"] == "RETRY",
        "selected": 1, "critique": verdict["critique"],
    }
    if not retry["triggered"]:
        return first, retry
    facts = "\n\nObserved execution errors (historical; check whether repaired):\n" + "\n".join(
        summary["discovered_errors"]
    ) if summary["discovered_errors"] else ""
    second = _run_attempt(
        task_id, exp + "_retry", lesson, RETRY_NOTE.format(critique=retry["critique"] + facts), retry=True,
    )
    other = summarize_attempt(second["steps"], second["guardrails"], second["error"])
    if (summary["final_status"] == "fail" or summary["unhandled_exception"]) and (
        other["final_status"] == "success" and not other["unhandled_exception"]
    ):
        choice = 2
        selection = {"method": "terminal_recovery"}
    else:
        choice, selection = _judge_attempts(instruction, summary, other)
    retry["selected"] = choice
    retry["selection"] = selection
    return second if choice == 2 else first, retry


def _job(args: tuple) -> dict:
    cond, run, task_id, lesson = args
    out_file = ROOT / "experiments" / "gate1" / f"{cond}_{run}" / f"{task_id}.json"
    if out_file.exists():
        return json.loads(out_file.read_text())
    exp = f"spine/gate1/{cond}_{run}"
    t0 = time.time()
    if _CFG["options"].get("retry_mode") == "selective":
        _CFG["critic_trace"] = str(out_file.parent / "critic" / (task_id + ".jsonl"))
    attempt = _run_attempt(task_id, exp, lesson)
    retry = None
    if _CFG["options"].get("retry_mode") == "selective":
        attempt, retry = _select_attempt(task_id, exp, lesson, attempt)
    # Evaluation is deliberately deferred until every audit and selection call has finished.
    from appworld.evaluator import evaluate_task

    selected_exp = attempt["exp"]
    ev = evaluate_task(task_id, selected_exp).to_dict()
    fails = ev.get("failures", [])
    ans = [f for f in fails if f["requirement"].startswith("assert answers match")]
    rec = {
        "cond": cond, "run": run, "task_id": task_id, "success": bool(ev["success"]),
        "difficulty": ev.get("difficulty"), "num_tests": ev.get("num_tests"),
        "failed_requirements": [f["requirement"] for f in fails],
        # Record the null-answer mismatch without persisting the reference answer or trace.
        "answered_null_task": bool(ans and re.search(r"== 'null'\s*$", ans[0]["trace"].strip())),
        "num_steps": len(attempt["steps"]), "error": attempt["error"], "seconds": round(time.time() - t0, 1),
    }
    if retry is not None:
        rec["retry"] = retry
        rec["selected_exp"] = selected_exp
    if _CFG["options"].get("guardrails", "none") != "none":
        rec["guardrails"] = attempt["guardrails"]
        rec["num_agent_steps"] = attempt["num_agent_steps"]
    if _CFG["options"].get("rsi_guards", "none") != "none":
        rec["rsi_triggers"] = attempt.get("rsi_triggers", dict.fromkeys(RSI_RULES, 0))
        rec["rsi_hits"] = attempt.get("rsi_hits", [])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    tmp_file.replace(out_file)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--lesson", default="", help="Playbook or lesson file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, help="Run only the first N selected tasks in each run")
    ap.add_argument("--runs", default="r1,r2,r3")
    ap.add_argument("--np", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--header", choices=list(HEADERS), default="lesson", help="Header for static injection")
    ap.add_argument("--injection-mode", choices=INJECTION_MODES, default="static")
    ap.add_argument("--jit-topk", type=int, default=3, help="Maximum playbook rules per JIT decision")
    ap.add_argument("--guardrails", choices=GUARDRAIL_MODES, default="none")
    ap.add_argument("--auto-doc", choices=("full", "none"), default="full")
    ap.add_argument("--rsi-guards", choices=RSI_MODES, default="none")
    ap.add_argument("--retry-mode", choices=("none", "selective"), default="none")
    ap.add_argument("--critic-model", default="", help="Critic served model name, defaults to qwen3.5-9b")
    ap.add_argument("--critic-thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retry-temperature", type=float, default=0.4)
    ap.add_argument("--retry-trigger", choices=("all_audited", "risk_only"), default="risk_only")
    args = ap.parse_args()
    if args.np < 1:
        ap.error("--np must be positive")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.jit_topk < 1:
        ap.error("--jit-topk must be positive")
    if not 0 <= args.retry_temperature <= 2:
        ap.error("--retry-temperature must be between 0 and 2")
    if args.injection_mode == "jit_scoped" and not args.lesson:
        ap.error("jit_scoped requires --lesson")
    runs = list(dict.fromkeys(args.runs.split(",")))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in [args.cond, *runs]):
        ap.error("--cond and --runs must contain only letters, numbers, underscores or hyphens")
    setup_env()
    lesson = Path(args.lesson).read_text() if args.lesson and args.injection_mode != "none" else ""
    tasks = resolve_tasks(args.tasks)
    if not tasks:
        ap.error("--tasks selected no tasks")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    settings = options = None
    if (args.injection_mode != "static" or args.guardrails != "none" or args.retry_mode == "selective"
            or args.auto_doc != "full" or args.rsi_guards != "none"):
        options = {"injection_mode": args.injection_mode, "guardrails": args.guardrails,
                   "auto_doc": args.auto_doc, "rsi_guards": args.rsi_guards}
        if args.injection_mode == "jit_scoped":
            try:
                entries = parse_playbook(lesson)
                if not entries:
                    raise ValueError("jit_scoped requires a --lesson file with playbook entries")
                ScopedPlaybookRouter(entries, args.jit_topk)
            except ValueError as error:
                ap.error(str(error))
            options.update(jit_topk=args.jit_topk, playbook_entries=entries)
        settings = {key: value for key, value in options.items() if key != "playbook_entries"}
        settings.update(
            model=args.model, max_tokens=args.max_tokens, header=args.header,
            lesson_sha256=hashlib.sha256(lesson.encode()).hexdigest(),
        )
        if (args.injection_mode == "jit_scoped" or args.guardrails != "none" or args.retry_mode == "selective"
                or args.rsi_guards != "none" or args.auto_doc != "full"):
            settings["harness_revision"] = 5
        if args.retry_mode == "selective":
            retry_options = dict(
                retry_mode="selective", critic_model=args.critic_model or DEFAULT_MODEL,
                critic_thinking=args.critic_thinking, retry_temperature=args.retry_temperature,
                retry_trigger=args.retry_trigger,
            )
            options.update(retry_options)
            settings.update(retry_options)
            settings.update(summary_policy="full_runtime_evidence", critic_max_tokens=8192,
                            selection_policy="evidence_order_consensus")
    try:
        for run in runs:
            _check_run_settings(f"{args.cond}_{run}", settings)
    except (OSError, ValueError) as error:
        ap.error(str(error))
    jobs = [(args.cond, r, t, lesson) for r in runs for t in tasks]
    print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: {len(jobs)} jobs, model={args.model}", flush=True)
    ok = n = 0
    steps = agent_steps = 0
    retried = selected_retry = 0
    guardrail_totals = ExecutionGuardrails().stats()
    rsi_totals = dict.fromkeys(RSI_RULES, 0)
    ctx = mp.get_context("spawn")
    workers = min(args.np, len(jobs))
    with ctx.Pool(
        workers, initializer=_init, initargs=(args.model, args.max_tokens, HEADERS[args.header], options),
        maxtasksperchild=20,
    ) as pool:
        for rec in pool.imap_unordered(_job, jobs):
            n += 1
            ok += rec["success"]
            steps += rec.get("num_steps", 0)
            agent_steps += rec.get("num_agent_steps", rec.get("num_steps", 0))
            retried += bool(rec.get("retry", {}).get("triggered"))
            selected_retry += rec.get("retry", {}).get("selected") == 2
            for key, count in rec.get("guardrails", {}).items():
                guardrail_totals[key] += count
            for key, count in rec.get("rsi_triggers", {}).items():
                rsi_totals[key] += count
            if n % 20 == 0 or n == len(jobs):
                print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: {n}/{len(jobs)} done, success {ok}", flush=True)
    print(f"Average environment steps: {steps / n:.2f}", flush=True)
    if args.guardrails != "none":
        print(f"Average agent steps: {agent_steps / n:.2f}", flush=True)
        print(f"Guardrail totals: {json.dumps(guardrail_totals)}", flush=True)
    if args.retry_mode == "selective":
        print(f"Retry totals: retried={retried}, selected_retry={selected_retry}", flush=True)
    if args.rsi_guards != "none":
        print(f"RSI totals: {json.dumps(rsi_totals)}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {args.cond}: ALL DONE {ok}/{n}", flush=True)


if __name__ == "__main__":
    main()
