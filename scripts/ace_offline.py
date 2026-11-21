"""Build a playbook with ACE-style offline adaptation on the AppWorld train split.

The Generator, Reflector, and Curator share a non-thinking model. The
Generator uses AppWorld's simplified ReAct prompt with the playbook appended.
The Reflector uses trajectories, reference code, and test reports to tag rules
and guide retries. Each batch shares a frozen playbook while generating and
retrying. The Curator then adds rules sequentially using the updated playbook.

Batch checkpoints and the final playbook are saved under experiments/ace/<run>.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from run_lesson import (  # noqa: E402
    DEFAULT_BASE_URL, DEFAULT_MODEL, PLAYBOOK_HEADER,
    load_agent_config, make_agent, resolve_tasks, setup_env,
)

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["strategies_and_hard_rules", "apis_to_use_for_specific_information", "useful_code_snippets_and_templates",
            "common_mistakes_to_avoid", "verification_checklist", "others"]
SLUG = {"strategies_and_hard_rules": "shr", "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code", "common_mistakes_to_avoid": "err",
        "verification_checklist": "ver", "others": "misc"}

REFLECTOR_PROMPT = """You are an expert AppWorld coding agent and educator. Your job is to diagnose the current trajectory: identify what went wrong (or could be better), grounded in execution feedback, API usage, unit test report, and ground truth when applicable.

Instructions:
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Identify root causes: wrong source of truth, bad filters (timeframe/direction/identity), formatting issues, or missing authentication and how to correct them.
- Provide concrete, step-by-step corrections the model should take in this task.
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer)
- Explicitly curate from the environment feedback the output format/schema of APIs used when unclear or mismatched with expectations (e.g., apis.blah.show_contents() returns a list of content_ids (strings), not content objects)

Inputs:
Ground truth code (reference, known-correct):
GROUND_TRUTH_CODE_START
{ground_truth_code}
GROUND_TRUTH_CODE_END

Test report (unit tests result for the task after the generated code was run):
TEST_REPORT_START
{test_report}
TEST_REPORT_END

ACE playbook (playbook that's used by model for code generation):
PLAYBOOK_START
{playbook}
PLAYBOOK_END

Outputs: Your output should be a json object, which contains the following fields
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- error_identification: what specifically went wrong in the reasoning?
- root_cause_analysis: why did this error occur? What concept was misunderstood?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, formula, or principle should be remembered to avoid this error?
- bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator

Answer in this exact JSON format:
{{"reasoning": "...", "error_identification": "...", "root_cause_analysis": "...", "correct_approach": "...", "key_insight": "...", "bullet_tags": [{{"id": "shr-00001", "tag": "helpful"}}]}}

Task: {task}
Outcome: {outcome}

FULL AGENT-ENVIRONMENT TRAJECTORY:
{trajectory}
"""

CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

Context:
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

Instructions:
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- For coding tasks, explicitly curate from the reflections the output format/schema of APIs used when unclear or mismatched with expectations (e.g., apis.blah.show_contents() returns a list of content_ids (strings), not content objects)

Task Context (the actual task instruction):
{task}

Current Playbook:
{playbook}

Current Generated Attempt (latest attempt, with reasoning and planning):
{attempt}

Current Reflections (principles and strategies that helped to achieve current task):
{reflection}

Your Task: Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to (one of: {sections})
  - content: the new content of the bullet

Available Operations:
1. ADD: Create new bullet points with fresh IDs
  - section: the section to add the new bullet to
  - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[shr-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):
{{"reasoning": "...", "operations": [{{"type": "ADD", "section": "strategies_and_hard_rules", "content": "..."}}]}}
"""

RETRY_NOTE = ("\n\nREFLECTION ON YOUR PREVIOUS ATTEMPT AT THIS EXACT TASK (it was NOT solved correctly; "
              "the environment has been fully reset):\n{reflection}\n")


# ---------------------------------------------------------------- playbook
class Playbook:
    def __init__(self) -> None:
        self.bullets: list[dict] = []
        self.next_id = 1

    def render(self) -> str:
        if not self.bullets:
            return "(empty)"
        out = []
        for sec in SECTIONS:
            bs = [b for b in self.bullets if b["section"] == sec]
            if bs:
                out.append(f"## {sec.upper()}")
                out += [f"[{b['id']}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}" for b in bs]
                out.append("")
        return "\n".join(out).strip()

    def add(self, section: str, content: str, origin: str) -> None:
        section = section.strip().lower().replace(" ", "_").replace("&", "and")
        if section not in SECTIONS:
            section = "others"
        content = content.strip()
        if not content:
            return
        self.bullets.append({"id": f"{SLUG[section]}-{self.next_id:05d}", "section": section, "content": content,
                             "helpful": 0, "harmful": 0, "origin": origin})
        self.next_id += 1

    def tag(self, tags: list) -> None:
        by_id = {b["id"]: b for b in self.bullets}
        for t in tags or []:
            if not isinstance(t, dict):
                continue
            b = by_id.get(str(t.get("id") or t.get("bullet_id") or t.get("bullet") or ""))
            tag = str(t.get("tag", "")).lower()
            if b and tag in ("helpful", "harmful"):
                b[tag] += 1

    def save(self, path: Path) -> None:
        path.with_suffix(".json").write_text(json.dumps({"next_id": self.next_id, "bullets": self.bullets}, ensure_ascii=False, indent=1))
        path.with_suffix(".txt").write_text(self.render() + "\n")

    @classmethod
    def load(cls, path: Path) -> "Playbook":
        d = json.loads(path.read_text())
        p = cls()
        p.bullets, p.next_id = d["bullets"], d["next_id"]
        return p


def chat(prompt: str, max_tokens: int = 3000) -> str:
    from openai import OpenAI
    c = OpenAI(
        base_url=os.environ.get("APPWORLD_MODEL_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        timeout=600,
    )
    for i in range(5):
        try:
            r = c.chat.completions.create(model=DEFAULT_MODEL, temperature=0.0, seed=100, max_tokens=max_tokens,
                                          messages=[{"role": "user", "content": prompt}],
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            print(f"[chat] {type(e).__name__}: {str(e)[:200]}; retry {i + 1}", flush=True)
            time.sleep(10)
    return ""


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def render_traj(steps: list[dict], obs_chars: int = 2000, total: int = 40000, code_only: bool = False) -> str:
    parts = []
    for i, s in enumerate(steps):
        if code_only:
            parts.append(f"## Step {i + 1}\n```python\n{s['input']}\n```")
        else:
            o = s["output"] if len(s["output"]) <= obs_chars else s["output"][:obs_chars] + "...[truncated]"
            parts.append(f"## Step {i + 1}\n```python\n{s['input']}\n```\nOutput:\n```\n{o}\n```")
    t = "\n\n".join(parts)
    if len(t) > total:
        t = t[: total // 4] + "\n\n...[middle steps truncated]...\n\n" + t[-3 * total // 4:]
    return t


def gt_code(task_id: str) -> str:
    # The compiled solution uses the same API surface available to the agent.
    s = (ROOT / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py").read_text()
    i = s.find("def solution(")
    return s[i:] if i >= 0 else s


_CFG: dict = {}


def _init(max_tokens: int) -> None:
    setup_env()
    cfg = load_agent_config()
    cfg["model_config"]["max_tokens"] = max_tokens
    _CFG["agent"] = cfg


def _task(args: tuple) -> dict:
    """Run one training task, reflecting after each attempt."""
    run, ep, task_id, playbook_text, rounds = args
    from appworld import AppWorld
    from appworld.evaluator import evaluate_task
    from appworld.task import Task
    instruction = Task.load(task_id=task_id).instruction
    rec = {"task_id": task_id, "epoch": ep, "attempts": [], "tags": []}
    reflection = ""
    for k in range(rounds + 1):
        exp = f"spine/ace/{run}/e{ep}_a{k}"
        extra = RETRY_NOTE.format(reflection=reflection) if reflection else ""
        err = None
        try:
            make_agent(_CFG["agent"], playbook_text, PLAYBOOK_HEADER, extra).solve_tasks(task_ids=[task_id], experiment_name=exp)
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
        try:
            steps = AppWorld.parse_environment_io_log(experiment_name=exp, task_id=task_id)
        except Exception:  # noqa: BLE001
            steps = []
        ev = evaluate_task(task_id, exp).to_dict()
        ok = bool(ev["success"])
        report_f = ROOT / "experiments" / "outputs" / exp / "tasks" / task_id / "evaluation" / "report.md"
        report = report_f.read_text()[:8000] if report_f.exists() else json.dumps(ev.get("failures", []))[:8000]
        refl = parse_json(chat(REFLECTOR_PROMPT.format(
            ground_truth_code=gt_code(task_id), test_report=report, playbook=playbook_text or "(empty)",
            task=instruction, outcome="PASSED all unit tests" if ok else "FAILED unit tests",
            trajectory=render_traj(steps))))
        rec["tags"] += refl.get("bullet_tags", []) if isinstance(refl.get("bullet_tags"), list) else []
        rec["attempts"].append({"attempt": k, "success": ok, "num_steps": len(steps), "error": err,
                                "reflection": {kk: refl.get(kk) for kk in ("error_identification", "root_cause_analysis", "correct_approach", "key_insight")}})
        rec["last_reflection"] = json.dumps({kk: v for kk, v in refl.items() if kk != "bullet_tags"}, ensure_ascii=False)
        rec["last_attempt"] = render_traj(steps, total=12000, code_only=True)
        rec["instruction"] = instruction
        if ok:
            break
        reflection = rec["last_reflection"]
    rec["first_success"] = rec["attempts"][0]["success"]
    rec["final_success"] = rec["attempts"][-1]["success"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default="train")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5, help="Maximum reflection-guided retries after failures")
    ap.add_argument("--np", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="Initialize from an existing playbook JSON file")
    args = ap.parse_args()
    setup_env()
    from appworld import load_task_ids

    tasks = resolve_tasks(args.tasks)
    if not tasks or set(tasks) - set(load_task_ids("train")):
        ap.error("ACE adaptation requires a non-empty selection from the train split")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        ap.error("--run must contain only letters, numbers, underscores or hyphens")
    if args.epochs < 1 or args.batch < 1 or args.np < 1 or args.rounds < 0:
        ap.error("epochs, batch and workers must be positive, and rounds cannot be negative")
    out = ROOT / "experiments" / "ace" / args.run
    out.mkdir(parents=True, exist_ok=True)
    pb = Playbook.load(Path(args.resume)) if args.resume else Playbook()
    log = (out / "log.jsonl").open("a")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.np, initializer=_init, initargs=(args.max_tokens,), maxtasksperchild=10) as pool:
        for ep in range(args.epochs):
            order = tasks[:]
            random.Random(args.seed + ep).shuffle(order)
            for b in range(0, len(order), args.batch):
                batch = order[b: b + args.batch]
                snap = pb.render() if pb.bullets else ""
                t0 = time.time()
                recs = pool.map(_task, [(args.run, ep, t, snap, args.rounds) for t in batch], chunksize=1)
                for r in recs:
                    pb.tag(r["tags"])
                n_add = 0
                for r in recs:  # Each Curator call sees the playbook updated by earlier calls.
                    cur = parse_json(chat(CURATOR_PROMPT.format(
                        task=r["instruction"], playbook=pb.render(), attempt=r["last_attempt"],
                        reflection=r["last_reflection"], sections=", ".join(SECTIONS))))
                    for op in cur.get("operations", []) or []:
                        if isinstance(op, dict) and str(op.get("type", "")).upper() == "ADD":
                            pb.add(str(op.get("section", "others")), str(op.get("content", "")), f"e{ep}:{r['task_id']}")
                            n_add += 1
                tag = f"e{ep}_b{b // args.batch}"
                pb.save(out / f"playbook_{tag}")
                stat = {"tag": tag, "n": len(recs), "first_success": sum(r["first_success"] for r in recs),
                        "final_success": sum(r["final_success"] for r in recs),
                        "episodes": sum(len(r["attempts"]) for r in recs), "added": n_add,
                        "bullets": len(pb.bullets), "playbook_chars": len(pb.render()),
                        "minutes": round((time.time() - t0) / 60, 1), "time": time.strftime("%H:%M:%S")}
                log.write(json.dumps({"stat": stat, "records": [{k: v for k, v in r.items() if k not in ("last_attempt",)} for r in recs]}, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps(stat), flush=True)
    pb.save(out / "playbook_final")
    print("ACE DONE", flush=True)


if __name__ == "__main__":
    main()
