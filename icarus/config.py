"""Central configuration and paths for the ICaRuS demo.

Paths default to the sibling ``chebILP`` checkout, which already carries the
pre-built ChEBI v251 molecule DataFrame and hierarchy graph. The app is meant to
run under the ``chebILP/.wslvenv`` interpreter (Linux), so the defaults are
POSIX ``/mnt/c/...`` paths; override with environment variables if your layout
differs.
"""

import os

# Root of the chebILP directory (provides data files and the
# chebILP / chebi_utils packages on the import path).
CHEBILP_DIR = os.environ.get(
    "ICARUS_CHEBILP_DIR",
    ".",
)

# chebILP keeps its API config (OPENAI_API_BASE/OPENAI_API_KEY, ANTHROPIC_API_KEY)
# in CHEBILP_DIR/.env and reads it via load_dotenv(), which searches from the
# process cwd upward. We run from the icarus dir, so that search never reaches
# chebILP's sibling .env — load it explicitly here (without overriding anything
# already set in the real environment) so the openai/ LLM backend finds its base
# URL and key.
def _load_chebilp_env() -> None:
    env_path = os.path.join(CHEBILP_DIR, ".env")
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Fallback: minimal KEY=VALUE parser so the openai/ backend still finds its
    # config even if python-dotenv isn't importable in this interpreter.
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip("'\"")


_load_chebilp_env()

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

# Rule-generation methods the /api/learn endpoint accepts. All three reuse
# chebILP infrastructure and produce a rule whose head is TARGET_LABEL, so the
# classify / explain / NL steps downstream are method-agnostic.
LEARN_METHODS = ("popper", "aleph", "llm")
DEFAULT_LEARN_METHOD = "popper"

# Aleph search bias. Aleph (chebILP.aleph_runner) runs under swipl over the same
# atom-level background as Popper; clauselength is derived from this max_body.
DEFAULT_ALEPH_MAX_BODY = 6

# LLM path (chebILP.predicate_generation.generate_auxiliary_rules). chebILP's
# llm_client routes on the model id: a `provider/name` id (e.g.
# `openai/agent_d7-gH6BBrzS_2ndkcKjLB`, the qwen3.5 model) goes to the
# OpenAI-compatible endpoint set by OPENAI_API_BASE/OPENAI_API_KEY; a bare id
# (e.g. `claude-haiku-4-5`) runs through the locally logged-in `claude` CLI.
LLM_MODEL = os.environ.get("ICARUS_LLM_MODEL", "")
# Number of auxiliary predicates to request from the model per rule.
LLM_N_PREDICATES = int(os.environ.get("ICARUS_LLM_N_PREDICATES", "4"))
# Reuse candidates retrieved from the session-local rule library per learn.
LLM_TOP_K = int(os.environ.get("ICARUS_LLM_TOP_K", "16"))
# Number of pos/neg example SMILES shown to the model in the prompt.
LLM_PROMPT_SAMPLES = int(os.environ.get("ICARUS_LLM_PROMPT_SAMPLES", "8"))
# Session-local shared library the LLM pipeline writes/reuses auxiliary rules in
# (the real chebILP pipeline retrieves reuse candidates from it across learns).
LLM_LIBRARY_DIR = os.path.join(ICARUS_DATA, "rule_library")
# Popper / Aleph background augmentation: when the concept has a definition, the
# N library predicates most relevant to it (chebILP's HybridPredicateRetriever over
# name + definition) are grounded on the session molecules and offered as extra
# body predicates. 0 disables the augmentation.
ILP_LIBRARY_TOP_N = int(os.environ.get("ICARUS_ILP_LIBRARY_TOP_N", "8"))
# Offer mol_weight / ring_size facts to the model. Kept OFF so every predicate the
# model writes grounds against icarus's standard atom-level background (the same
# bk the classify step builds); otherwise a rule using those facts would silently
# never fire at inference time.
LLM_COMPUTED_FACTS = False


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
