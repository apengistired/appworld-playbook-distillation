"""Keep a local copy of a remote benchmark and verify its finished output archives."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.local_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ssh = shlex.join(["ssh", "-p", str(args.port), "-o", "BatchMode=yes",
                      "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
                      "-o", f"UserKnownHostsFile={args.known_hosts.resolve()}"])
    command = [
        "rsync", "-rt", "--partial", "--timeout=120", "--exclude=*.partial", "--exclude=*.tmp",
        "--exclude=.backup.lock", "--exclude=backup.log", "--exclude=LOCAL_BACKUP_COMPLETE",
        "-e", ssh, f"{args.host}:{args.remote_dir.rstrip('/')}/", str(destination) + "/",
    ]
    with (destination / ".backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = subprocess.run(command, timeout=3600)
                if result.returncode:
                    raise RuntimeError(f"Backup transfer exited {result.returncode}")
                manifest = destination / "manifest.json"
                if (destination / "COMPLETE").exists() and manifest.exists():
                    settings = json.loads(manifest.read_text())["settings"]
                    expected = len(settings["conditions"]) * len(settings["splits"])
                    checksums = sorted(destination.glob("*.tar.gz.sha256"))
                    if len(checksums) != expected:
                        raise RuntimeError("Waiting for every archive checksum file")
                    for checksum in checksums:
                        digest, name = checksum.read_text().strip().split(maxsplit=1)
                        if Path(name).name != name:
                            raise ValueError("Unexpected archive filename")
                        with (destination / name).open("rb") as file:
                            actual = hashlib.file_digest(file, "sha256").hexdigest()
                        if digest != actual:
                            (destination / name).unlink()
                            raise RuntimeError(f"Checksum mismatch, will recopy {name}")
                    (destination / "LOCAL_BACKUP_COMPLETE").write_text(
                        time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
                    )
                    print("All results and archives downloaded and verified", flush=True)
                    return
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Backup refreshed, benchmark still running", flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), type(error).__name__, str(error), flush=True)
            time.sleep(120)


if __name__ == "__main__":
    main()
