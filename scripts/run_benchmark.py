"""Run three-repeat AppWorld baselines and archive the complete experiment outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "base": ("qwen3.5-9b", "none"),
    "lora20": ("ace20", "none"),
    "control105": ("ctrl105", "none"),
    "base_static20": ("qwen3.5-9b", "static"),
}
RUNS = ("r1", "r2", "r3")
SPLITS = {"dev": 57, "test_normal": 168}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def summarize(prefix: str, splits: list[str]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for split in splits:
            expected = SPLITS[split]
            ids = set((ROOT / "data/datasets" / f"{split}.txt").read_text().splitlines())
            assert len(ids) == expected
            trials = []
            for run in RUNS:
                folder = ROOT / "experiments/gate1" / f"{prefix}_{condition}_{split}_{run}"
                records = [json.loads(p.read_text()) for p in folder.glob("*.json")]
                by_task = {r["task_id"]: r for r in records}
                if len(by_task) != len(records) or not by_task.keys() <= ids:
                    raise ValueError(f"Duplicate or unexpected tasks in {folder}")
                trials.append({
                    "run": run, "completed": len(records), "expected": expected,
                    "passed": sum(r["success"] for r in records),
                    "runner_errors": sum(bool(r["error"]) for r in records),
                    "tgc": 100 * sum(r["success"] for r in records) / expected if by_task.keys() == ids else None,
                })
            scores = [r["tgc"] for r in trials if r["tgc"] is not None]
            rows.append({
                "condition": condition, "split": split, "runs": trials,
                "mean_tgc": statistics.mean(scores) if len(scores) == 3 else None,
                "std_tgc": statistics.stdev(scores) if len(scores) == 3 else None,
            })
    return rows


def archive_split(output: Path, prefix: str, condition: str, split: str) -> None:
    name = f"{condition}_{split}.tar.gz"
    archive = output / name
    if archive.exists() and archive.with_suffix(".gz.sha256").exists():
        return
    temporary = output / (name + ".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as target:
        for run in RUNS:
            label = f"{prefix}_{condition}_{split}_{run}"
            for folder in (Path("experiments/gate1") / label, Path("experiments/outputs/spine/gate1") / label):
                target.add(ROOT / folder, arcname=str(folder))
    with temporary.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    temporary.replace(archive)
    archive.with_suffix(".gz.sha256").write_text(f"{digest}  {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--workers-per-condition", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["test_normal"])
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix) or args.workers_per_condition < 1:
        parser.error("Use a simple prefix and positive worker count")
    output = ROOT / "artifacts/benchmarks" / args.prefix
    output.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        print(json.dumps(summarize(args.prefix, args.splits), indent=2))
        return

    manifest_path = output / "manifest.json"
    settings = {"prefix": args.prefix, "conditions": CONDITIONS, "splits": args.splits, "runs": RUNS,
                "workers_per_condition": args.workers_per_condition, "temperature": 0, "seed": 100,
                "max_tokens": 3000, "max_steps": 50, "guardrails": "none", "retry_mode": "none"}
    settings = json.loads(json.dumps(settings))
    files = ["scripts/run_benchmark.py", "scripts/run_lesson.py", "scripts/harness_modules.py",
             "configs/qwen3.5-9b.jsonnet", "artifacts/playbook_top20.txt",
             "artifacts/lora_ace20_vllm/adapter_config.json", "artifacts/lora_ace20_vllm/adapter_model.safetensors",
             "artifacts/lora_ctrl105_vllm/adapter_config.json", "artifacts/lora_ctrl105_vllm/adapter_model.safetensors"]
    settings["sha256"] = {}
    for name in files:
        with (ROOT / name).open("rb") as file:
            settings["sha256"][name] = hashlib.file_digest(file, "sha256").hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            raise ValueError("Saved benchmark manifest has different code or settings")
    else:
        base = os.environ.get("APPWORLD_MODEL_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        with urlopen(base + "/models", timeout=10) as reply:
            models = json.load(reply)
        assert {m["id"] for m in models["data"]} >= {"qwen3.5-9b", "ace20", "ctrl105"}
        with urlopen(base.removesuffix("/v1") + "/version", timeout=10) as reply:
            version = json.load(reply)
        assert version["version"] == "0.33.0", version
        snapshot = output / "source"
        for name in files:
            if name.endswith(".safetensors"):
                continue
            dst = snapshot / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dst)
        write_json(manifest_path, {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "settings": settings, "server_version": version, "served_models": models})

    lock = threading.Lock()
    active = {name: "pending" for name in CONDITIONS}
    def publish() -> None:
        with lock:
            summary = summarize(args.prefix, args.splits)
            write_json(output / "summary.json", summary)
            write_json(output / "status.json", {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "groups": active})
            with (output / "summary.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["condition", "split", "r1_passed", "r2_passed", "r3_passed", "mean_tgc", "sample_std_tgc"])
                for row in summary:
                    writer.writerow([row["condition"], row["split"], *[r["passed"] for r in row["runs"]],
                                     row["mean_tgc"], row["std_tgc"]])

    def run_condition(condition: str) -> None:
        model, mode = CONDITIONS[condition]
        try:
            for split in args.splits:
                active[condition] = f"running {split}"
                publish()
                command = [
                    sys.executable, str(ROOT / "scripts/run_lesson.py"),
                    "--cond", f"{args.prefix}_{condition}_{split}", "--tasks", split,
                    "--runs", ",".join(RUNS), "--np", str(args.workers_per_condition),
                    "--model", model, "--injection-mode", mode, "--guardrails", "none",
                    "--retry-mode", "none", "--max-tokens", "3000",
                ]
                if mode == "static":
                    command += ["--header", "playbook", "--lesson", str(ROOT / "artifacts/playbook_top20.txt")]
                for attempt in range(2):
                    with (output / f"{condition}_{split}.log").open("a") as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=12 * 3600)
                    if result.returncode == 0:
                        break
                if result.returncode:
                    raise RuntimeError(f"{condition}/{split} exited {result.returncode}")
                row = next(r for r in summarize(args.prefix, args.splits) if r["condition"] == condition and r["split"] == split)
                if row["mean_tgc"] is None or any(r["runner_errors"] for r in row["runs"]):
                    raise RuntimeError(f"{condition}/{split} has missing tasks or runner errors")
                active[condition] = f"archiving {split}"
                publish()
                archive_split(output, args.prefix, condition, split)
            active[condition] = "done"
        except Exception as error:
            active[condition] = f"failed: {type(error).__name__}: {error}"
            raise
        finally:
            publish()

    stop = threading.Event()
    def monitor() -> None:
        while not stop.wait(60):
            publish()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_condition, name) for name in CONDITIONS]
            for future in futures:
                future.result()
        publish()
        (output / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    finally:
        stop.set()
        monitor_thread.join()


if __name__ == "__main__":
    main()
