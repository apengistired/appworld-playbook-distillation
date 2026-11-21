#!/usr/bin/env bash
set -euo pipefail
v4_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "$v4_script_dir/scripts" ]]; then
  cd "$v4_script_dir"
else
  cd "$v4_script_dir/.."
fi
export PYTHONDONTWRITEBYTECODE=1
export APPWORLD_ROOT="$PWD"
export APPWORLD_CACHE="$PWD/.cache/appworld"
export APPWORLD_MODEL_URL=http://127.0.0.1:8000/v1
exec .venv/bin/python -u scripts/run_v4_queue.py "$@"
