"""Central configuration and paths for the ICaRuS demo.

All data lives in this project's ``data/`` directory (override with
``ICARUS_DATA_DIR``), including the pre-built ChEBI v251 molecule DataFrame and
hierarchy graph. Every setting below can be overridden by an environment variable
or a ``.env`` file in the project root (see ``.env.example`` and the README).
"""

import os

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# chebILP reads its API config (OPENAI_API_BASE/OPENAI_API_KEY, ANTHROPIC_API_KEY)
# via load_dotenv(), which searches from the process cwd upward. Load this
# project's .env explicitly (without overriding anything already set in the real
# environment) so the openai/ LLM backend finds its base URL and key regardless
# of the cwd.
def _load_env() -> None:
    env_path = os.path.join(_PROJECT_DIR, ".env")
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


_load_env()

# Data directory for this app: ChEBI data files (read-only), fingerprint cache,
# generated ILP problems and the rule library.
ICARUS_DATA = os.environ.get("ICARUS_DATA_DIR", os.path.join(_PROJECT_DIR, "data"))
os.makedirs(ICARUS_DATA, exist_ok=True)

CHEBI_VERSION = int(os.environ.get("ICARUS_CHEBI_VERSION", "251"))

_CHEBI_DATA = os.path.join(ICARUS_DATA, f"chebi_v{CHEBI_VERSION}")
MOLECULES_PKL = os.path.join(_CHEBI_DATA, "ChEBI25_3_STAR", "molecules.pkl")
GRAPH_PKL = os.path.join(_CHEBI_DATA, "chebi_graph.pkl")

FP_CACHE = os.path.join(ICARUS_DATA, "fingerprints.pkl")
WORK_DIR = os.path.join(ICARUS_DATA, "work")
os.makedirs(WORK_DIR, exist_ok=True)

# Web assets.
WEB_DIR = os.path.join(_PROJECT_DIR, "web")

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
DEFAULT_TIMEOUT = 60
# Upper bound on the user-settable ILP timeout (seconds).
MAX_TIMEOUT = int(os.environ.get("ICARUS_MAX_TIMEOUT", "300"))
# Learning runs allowed at once across all users (the rest get "server busy").
MAX_CONCURRENT_LEARNS = int(os.environ.get("ICARUS_MAX_CONCURRENT_LEARNS", "4"))
# Wall-clock budget (seconds) for one LLM learn: all model calls of the pipeline
# together. Enforced server-side (not client-settable) in icarus/llm_guard.py.
LLM_TIMEOUT = int(os.environ.get("ICARUS_LLM_TIMEOUT", "300"))

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
# Human-readable model name shown in the UI (for transparency about which model
# writes the rules) — useful when LLM_MODEL is an opaque endpoint id. Defaults to
# the model id itself.
LLM_MODEL_NAME = os.environ.get("ICARUS_LLM_MODEL_NAME", "") or LLM_MODEL
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

# How the `claude` CLI authenticates (bare model ids only). "subscription" uses the
# CLI's own login; "api_key" bills ANTHROPIC_API_KEY from the environment / .env
# (chebILP strips that key by default so it can't shadow the subscription).
LLM_AUTH = os.environ.get("ICARUS_LLM_AUTH", "subscription").lower()

# Abuse limits for a public deployment (see icarus/llm_guard.py). A "prompt" is one
# LLM learn that actually reaches the model (cached library hits are free).
# Global sliding window shared by every user: at most N prompts per window.
LLM_RATE_LIMIT = int(os.environ.get("ICARUS_LLM_RATE_LIMIT", "10"))
LLM_RATE_WINDOW = int(os.environ.get("ICARUS_LLM_RATE_WINDOW", "300"))
# Prompts per browser session before the user is asked to self-host or get in
# touch (0 = unlimited).
LLM_SESSION_LIMIT = int(os.environ.get("ICARUS_LLM_SESSION_LIMIT", "10"))
# Hard cap on one prompt's size (system + user prompt, in characters). A normal
# prompt is ~10-15k characters; this only stops pathological inputs.
LLM_MAX_PROMPT_CHARS = int(os.environ.get("ICARUS_LLM_MAX_PROMPT_CHARS", "40000"))
# Cap on the user-editable concept definition, the main free-text prompt input.
MAX_DEFINITION_CHARS = int(os.environ.get("ICARUS_MAX_DEFINITION_CHARS", "4000"))
# Invite codes (comma-separated) that lift both limits above for a browser: open the
# demo once as `/?invite=<code>` and the code is remembered in that browser. Only
# the API key's workspace spend limit still applies to invited users.
INVITE_CODES = [c.strip() for c in os.environ.get("ICARUS_INVITE_CODES", "").split(",")
                if c.strip()]
# Shown to users who hit the per-session limit (e.g. an email address or URL).
CONTACT = os.environ.get("ICARUS_CONTACT", "https://github.com/sfluegel05/icarus/issues/new")
# Append-only JSONL log of every model call (prompt, response, cost, session, client).
LLM_PROMPT_LOG = os.environ.get(
    "ICARUS_LLM_PROMPT_LOG", os.path.join(ICARUS_DATA, "llm_log", "prompts.jsonl"))


# External data files this app reads (everything else it generates itself). Kept
# here so a deployment on another machine has a single place to point at.
REQUIRED_FILES = {
    "molecules DataFrame": MOLECULES_PKL,
    "ChEBI hierarchy graph": GRAPH_PKL,
}


def check_data_files() -> list[str]:
    """Return a human-readable list of missing required files (empty if all present)."""
    missing = []
    for label, path in REQUIRED_FILES.items():
        if not os.path.exists(path):
            missing.append(f"  - {label}: {path}")
    return missing
