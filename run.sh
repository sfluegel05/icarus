#!/usr/bin/env bash
# Launch the ICaRuS web app (uvicorn, single process: sessions live in memory).
#
# Usage:   ./run.sh            (from Windows: wsl -e bash run.sh)
# then open http://localhost:8000
#
# Environment (all optional; .env in this directory is read by the app itself):
#   ICARUS_PYTHON  interpreter to use (default: ./.venv, then ./.wslvenv, then python3 on PATH)
#   ICARUS_HOST    bind address (default 0.0.0.0; use 127.0.0.1 behind a reverse proxy)
#   ICARUS_PORT    HTTP port (default 8000)
# Extra arguments are passed to uvicorn, e.g. ./run.sh --log-level warning
# Everything else (data dir, limits, LLM) is configured in icarus/config.py / .env.
set -euo pipefail

ICARUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ICARUS_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in "$ICARUS_DIR/.venv/bin/python" "$ICARUS_DIR/.wslvenv/bin/python"; do
    [ -x "$cand" ] && { PY="$cand"; break; }
  done
fi
PY="${PY:-$(command -v python3 || command -v python || true)}"
[ -n "$PY" ] || { echo "run.sh: no python found; create .venv or set ICARUS_PYTHON" >&2; exit 1; }
echo "run.sh: using $PY" >&2

export PYTHONPATH="$ICARUS_DIR:${PYTHONPATH:-}"
cd "$ICARUS_DIR"

exec "$PY" -m uvicorn icarus.app:app \
  --host "${ICARUS_HOST:-0.0.0.0}" --port "${ICARUS_PORT:-8000}" "$@"
