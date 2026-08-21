#!/usr/bin/env bash
# Launch the ICaRuS demo. By default it runs under the chebILP virtualenv (which
# already has popper, clingo, rdkit, xclingo, PIL, chebi_utils and chebILP
# installed) and imports the pre-built ChEBI data from that checkout.
#
# Usage (from Windows, in this repo):   wsl -e bash run.sh
# then open http://localhost:8000
#
# Configure for another system with environment variables (all optional):
#   ICARUS_CHEBILP_DIR   chebILP checkout: source packages + data/ (default below)
#   ICARUS_PYTHON        interpreter to use (default: $ICARUS_CHEBILP_DIR/.wslvenv/bin/python)
#   ICARUS_CHEBI_VERSION ChEBI release the data is for (default 251)
#   ICARUS_DATA_DIR      writable dir for the fingerprint cache + ILP work files
#   ICARUS_POOL_SIZE     number of molecules fingerprinted for suggestions
#   ICARUS_PORT          HTTP port (default 8000)
set -euo pipefail

CHEBILP_DIR="${ICARUS_CHEBILP_DIR:-/mnt/c/Users/sifluegel/PycharmProjects/chebILP}"
ICARUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ICARUS_PYTHON:-$CHEBILP_DIR/.wslvenv/bin/python}"
PORT="${ICARUS_PORT:-8000}"

export ICARUS_CHEBILP_DIR="$CHEBILP_DIR"
export PYTHONPATH="$ICARUS_DIR:${PYTHONPATH:-}"
cd "$ICARUS_DIR"

exec "$PY" -m uvicorn icarus.app:app --host 0.0.0.0 --port "$PORT" "$@"
