"""Central configuration and paths for the ICaRuS demo.

Paths default to the sibling ``chebILP`` checkout, which already carries the
pre-built ChEBI v251 molecule DataFrame and hierarchy graph. The app is meant to
run under the ``chebILP/.wslvenv`` interpreter (Linux), so the defaults are
POSIX ``/mnt/c/...`` paths; override with environment variables if your layout
differs.
"""

import os

# Root of the chebILP checkout (provides molecules.pkl, chebi_graph.pkl and the
# chebILP / chebi_utils packages on the import path).
CHEBILP_DIR = os.environ.get(
    "ICARUS_CHEBILP_DIR",
    "/mnt/c/Users/sifluegel/PycharmProjects/chebILP",
)

CHEBI_VERSION = int(os.environ.get("ICARUS_CHEBI_VERSION", "251"))

_DATA = os.path.join(CHEBILP_DIR, "data", f"chebi_v{CHEBI_VERSION}")
MOLECULES_PKL = os.path.join(_DATA, "ChEBI25_3_STAR", "molecules.pkl")
GRAPH_PKL = os.path.join(_DATA, "chebi_graph.pkl")

# Working directory for this app (fingerprint cache, generated ILP problems).
ICARUS_DATA = os.environ.get(
    "ICARUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)
os.makedirs(ICARUS_DATA, exist_ok=True)

FP_CACHE = os.path.join(ICARUS_DATA, "fingerprints.pkl")
WORK_DIR = os.path.join(ICARUS_DATA, "work")
os.makedirs(WORK_DIR, exist_ok=True)

# Web assets.
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

# Size of the random ChEBI molecule pool used for similarity suggestions.
SIMILARITY_POOL_SIZE = int(os.environ.get("ICARUS_POOL_SIZE", "12000"))

# The generic target predicate the learned rule defines. Kept class-agnostic so
# arbitrary molecule concepts work, not only ChEBI classes.
TARGET_LABEL = "concept"

# Defaults for example gathering from a ChEBI class (kept small so non-noisy
# Popper can find a perfectly-separating rule quickly).
DEFAULT_MAX_POS = 12
DEFAULT_MAX_NEG = 8

# Popper search bias defaults.
DEFAULT_MAX_VARS = 4
DEFAULT_MAX_BODY = 5
DEFAULT_MAX_CLAUSES = 2
DEFAULT_TIMEOUT = 20


# External data files this app reads (everything else it generates itself). Kept
# here so a deployment on another machine has a single place to point at.
REQUIRED_FILES = {
    "molecules DataFrame (ICARUS_CHEBILP_DIR/data/...)": MOLECULES_PKL,
    "ChEBI hierarchy graph (ICARUS_CHEBILP_DIR/data/...)": GRAPH_PKL,
}


def check_data_files() -> list[str]:
    """Return a human-readable list of missing required files (empty if all present)."""
    missing = []
    for label, path in REQUIRED_FILES.items():
        if not os.path.exists(path):
            missing.append(f"  - {label}: {path}")
    return missing
