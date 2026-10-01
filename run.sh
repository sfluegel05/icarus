#!/usr/bin/env bash
# Launch the ICaRuS demo. By default it runs under this project's own virtualenv
# (.wslvenv, with requirements.txt installed: chebILP, chebi_utils, popper, clingo,
# rdkit, xclingo, PIL, ...) and reads the ChEBI data from this project's data/.
#
# Usage (from Windows, in this repo):   wsl -e bash run.sh
# then open http://localhost:8000
#
# Configure for another system with environment variables (all optional):
#   ICARUS_PYTHON        interpreter to use (default: ./.wslvenv/bin/python)
#   ICARUS_CHEBI_VERSION ChEBI release the data is for (default 251)
#   ICARUS_DATA_DIR      data dir: ChEBI data files, fingerprint cache, ILP work files
#   ICARUS_POOL_SIZE     number of molecules fingerprinted for suggestions
#   ICARUS_PORT          HTTP port (default 8000)
set -euo pipefail

ICARUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#PY="${ICARUS_PYTHON:-$ICARUS_DIR/.wslvenv/bin/python}"
PORT="${ICARUS_PORT:-8000}"

export PYTHONPATH="$ICARUS_DIR:${PYTHONPATH:-}"
cd "$ICARUS_DIR"

exec python -m uvicorn icarus.app:app --host 0.0.0.0 --port "$PORT" "$@"
