# ICaRuS — Interactive ChemicAl RUle System

ICaRuS is a web app for describing a chemical concept as a **Prolog rule**. You
give it positive and negative example molecules, and one of three interchangeable
backends learns the rule: the ILP systems **Popper** (noisy/MDL) and **Aleph**, or
an **LLM**. Examples come from a ChEBI class or from pasted SMILES/InChI, and the
app suggests more molecules by fingerprint similarity. It flags the molecules the
rule misclassifies, explains why a molecule matches, and lets you edit the rule
(visually or as Prolog), see which classifications change, and iterate.

It extends the per-class ChEBI approach of [chebILP](https://pypi.org/project/chebILP/)
to arbitrary molecules. The in-app tutorial (`/tutorial`) walks through a
full session.

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Deploying on a web server](#deploying-on-a-web-server)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Limitations](#limitations)

## Quick start

ICaRuS runs on Linux (on Windows, inside WSL). You need:

- **Python 3.14+**, required by current Popper. If your distribution ships an older
  version, [uv](https://docs.astral.sh/uv/) can fetch 3.14 for you (see below).
- **SWI-Prolog** (`swipl` on `PATH`), used by Popper and Aleph.
- A **C compiler** and **git**, because Popper is installed from git and builds a
  Cython extension.

On Debian/Ubuntu:

```bash
sudo apt install swi-prolog-nox build-essential git
```

### 1. Install

```bash
git clone https://github.com/sfluegel05/icarus.git
cd icarus
uv venv --python 3.14 .venv          # or: python3.14 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
```

`requirements.txt` covers everything, including chebILP, chebi_utils and Popper
(pinned to a tested upstream commit). Keep **rdkit** at the pinned version: the
molecule data file contains pickled RDKit objects, and another version may fail to
load them.

### 2. Get the ChEBI data

The app reads two prebuilt files, which are not in git (about 230 MB):

| File | Path under `data/` |
|------|--------------------|
| Molecule table (~52k molecules: SMILES, InChI, name, RDKit mol) | `chebi_v251/ChEBI25_3_STAR/molecules.pkl` |
| ChEBI hierarchy graph (networkx DiGraph) | `chebi_v251/chebi_graph.pkl` |

chebILP builds both. It writes to `data/` under the current directory, so run it
from the repo root:

```bash
.venv/bin/python -m chebILP prepare_dataset --chebi_version 251
```

**Optional:** the LLM **rule library** (`data/rule_library/`) holds predicates and
rules generated earlier. ICaRuS only reads it. With a library, LLM answers for
known ChEBI classes are served from it without a model call, and Popper and Aleph
get extra background predicates. If you have `rule_library.zip`, unzip it so that
`data/rule_library/programs/` exists. Without a library everything still works.

### 3. Run

```bash
./run.sh                     # from Windows: wsl -e bash run.sh
```

Open <http://localhost:8000>. At startup the server reports whether it found the
data files and names any that are missing. The first launch builds a fingerprint
cache (about 30 s); later launches reuse it.

`run.sh` uses `./.venv`, then `./.wslvenv`, then `python3` on `PATH`. Set
`ICARUS_PYTHON` to choose a different interpreter.

To check the ILP pipeline without the web server or an LLM:

```bash
PYTHONPATH=. .venv/bin/python smoke_test.py   # learns "contains an oxygen atom"; ends with OK
```

Popper and Aleph need no API key. The LLM method needs extra setup, described in
[LLM backend](#llm-backend).

## Configuration

All settings are environment variables, and all are optional. The app also reads a
`.env` file in the repo root; copy [`.env.example`](.env.example) to start. Real
environment variables take precedence over `.env`. The defaults live in
[`icarus/config.py`](icarus/config.py).

| Variable | Default | Purpose |
|----------|---------|---------|
| `ICARUS_PYTHON` | `.venv` / `.wslvenv` / `python3` | Interpreter `run.sh` starts¹ |
| `ICARUS_HOST` | `0.0.0.0` | Bind address¹ (use `127.0.0.1` behind a reverse proxy) |
| `ICARUS_PORT` | `8000` | HTTP port¹ |
| `ICARUS_DATA_DIR` | `data/` | ChEBI data, fingerprint cache, temporary ILP files, rule library, prompt log |
| `ICARUS_CHEBI_VERSION` | `251` | ChEBI release of the data files |
| `ICARUS_POOL_SIZE` | `12000` | Number of molecules fingerprinted for suggestions |
| `ICARUS_MAX_TIMEOUT` | `300` | Upper bound (s) on the ILP timeout a user can set |
| `ICARUS_MAX_CONCURRENT_LEARNS` | `4` | Learning runs allowed at once across all users; further requests get "server busy" |
| `ICARUS_ILP_LIBRARY_TOP_N` | `8` | Library predicates added to the Popper/Aleph background (`0` = off) |

¹ `run.sh` reads these before the app starts, so set them in the real environment,
not in `.env`.

The app writes only into `ICARUS_DATA_DIR`, which it creates if needed. Each
learning run gets its own temporary directory under `work/`, deleted afterwards.

### LLM backend

The LLM method calls a model through the `claude` CLI that ships with the
`claude-agent-sdk` Python package (installed via chebILP). **Set `ICARUS_LLM_MODEL`
to enable it.**

| Variable | Default | Purpose |
|----------|---------|---------|
| `ICARUS_LLM_MODEL` | — | Model id, e.g. `claude-haiku-4-5`. A `provider/name` id goes to the OpenAI-compatible endpoint in `OPENAI_API_BASE` / `OPENAI_API_KEY` instead |
| `ICARUS_LLM_MODEL_NAME` | the model id | Human-readable name shown in the UI and rule history |
| `ICARUS_LLM_AUTH` | `subscription` | `subscription` uses the CLI's own login (`claude` → `/login`). `api_key` bills `ANTHROPIC_API_KEY` instead, so the server needs no login |
| `ICARUS_LLM_TIMEOUT` | `300` | Wall-clock limit (s) for one LLM learn |
| `ICARUS_LLM_N_PREDICATES` / `ICARUS_LLM_TOP_K` / `ICARUS_LLM_PROMPT_SAMPLES` | `4` / `16` / `8` | Predicates requested per rule / library candidates offered for reuse / example SMILES per class in the prompt |

Put keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) in `.env`, which git ignores.

## Deploying on a web server

ICaRuS is a single Starlette app served by uvicorn. Keep it to **one process**
(do not add `--workers`): sessions and learning progress are held in memory.

**Server sizing.** Plan for about **4 GB RAM**: the loaded ChEBI data takes about
2 GB. Each running Popper or Aleph search uses one CPU core for up to its timeout,
and `ICARUS_MAX_CONCURRENT_LEARNS` caps how many run at once. Startup takes 10–30 s
while the data loads.

A typical setup puts nginx in front of the app with TLS, and runs the app as a
systemd service bound to localhost:

1. **Install** as in the [Quick start](#quick-start), e.g. into `/srv/icarus`, as a
   dedicated user that owns `/srv/icarus/data`.
2. **Configure** `/srv/icarus/.env` (see [Public demo limits](#public-demo-limits)).
3. **Service:** adjust `User` and the paths in [`deploy/icarus.service`](deploy/icarus.service), then
   ```bash
   sudo cp deploy/icarus.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now icarus
   journalctl -u icarus -f          # wait for "reference data preloaded"
   ```
4. **Reverse proxy:** adapt [`deploy/nginx.conf`](deploy/nginx.conf) (server name, and
   either the site root or a sub-path), enable it, then add TLS, e.g. with
   `certbot --nginx -d icarus.example.org`.

**Sub-paths.** All URLs in the app are relative, so it can be served under a prefix
such as `https://example.org/icarus/`. The proxy must strip the prefix and redirect
`/icarus` to `/icarus/`. `deploy/nginx.conf` includes such a block.

**Client IPs.** uvicorn trusts `X-Forwarded-For` from `127.0.0.1` by default, so
the prompt log records real client addresses when nginx runs on the same host.

### Public demo limits

The LLM method costs money per prompt, and ILP runs cost CPU, so a public instance
should set limits. ILP load is bounded by `ICARUS_MAX_TIMEOUT` and
`ICARUS_MAX_CONCURRENT_LEARNS`. The server also clamps request sizes such as the
number of suggestions, examples and image dimensions. For the LLM:

1. Use an API key: set `ANTHROPIC_API_KEY` and `ICARUS_LLM_AUTH=api_key` in `.env`.
2. Set a **spend limit** on the key's workspace in the Anthropic Console. This is
   the only hard cap on cost.
3. Adjust the limits below, which `icarus/llm_guard.py` enforces.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ICARUS_LLM_RATE_LIMIT` / `ICARUS_LLM_RATE_WINDOW` | `10` / `300` | At most N LLM prompts per window (s), across all users |
| `ICARUS_LLM_SESSION_LIMIT` | `10` | LLM prompts per browser session (`0` = unlimited) |
| `ICARUS_CONTACT` | GitHub issues URL | Contact shown when a user reaches the session limit |
| `ICARUS_INVITE_CODES` | — | Comma-separated codes. A browser that once opens `/?invite=<code>` skips both limits above |
| `ICARUS_LLM_MAX_PROMPT_CHARS` | `40000` | Larger prompts are refused before sending (normal: 10–15k) |
| `ICARUS_MAX_DEFINITION_CHARS` | `4000` | Max length of the editable concept definition |
| `ICARUS_LLM_PROMPT_LOG` | `data/llm_log/prompts.jsonl` | Append-only log of every model call: prompt, response, cost, session, client IP |

A "prompt" is one LLM learn that reaches the model. Answers served from the rule
library are free. The model is set on the server only; clients cannot choose it.

The per-session count is a nudge rather than a barrier. It lives in the in-memory
session, which is dropped after 12 h idle or on restart, and a new tab starts a new
session. The global window is the limit that protects the budget. To revoke an
invite code, remove it from `ICARUS_INVITE_CODES` and restart.

## How it works

1. **Examples.** Optionally pick a ChEBI class to fill the example boxes. Positives
   are the class's **smallest descendant molecules** (fewest heavy atoms), so the
   minimal class core is exposed. Negatives are the molecules **closest to the class
   in the hierarchy** but not below it, found by a BFS outward over the `is_a`
   graph and topped up with random molecules only if the neighbourhood is small.
   Only concrete molecules are used anywhere: generic structures with R-groups or
   wildcards (`*`) are excluded. You can add more molecules as SMILES or InChI,
   move them between boxes, or remove them. Click a structure for its SMILES,
   InChI, name, ID and definition.
2. **Suggestions.** ICaRuS proposes ChEBI molecules related to your positives,
   scored by Morgan-fingerprint (ECFP4) Tanimoto similarity. It picks a **diverse**
   set by Maximal Marginal Relevance (`score = λ·rel − (1−λ)·max similarity to
   already-picked`), so you see varied scaffolds instead of near-duplicates. They
   come one at a time as cards. Once a rule exists, each card shows the rule's
   prediction, so you can confirm or contradict it while labelling.
3. **Learn.** Pick a method. The run happens in a background thread and its output
   streams into the page. All three methods use chebILP's atom-level translation
   of molecules into logic and produce a rule with the head `concept/1`, so
   everything after learning works the same for each method:
   - **Popper** learns from `exs.pl` / `bk.pl` / `bias.pl` in noisy mode (MDL cost)
     and returns the best rule found, even if none separates the examples perfectly.
   - **Aleph** learns from the same background and predicates (as `.b` / `.f` / `.n`
     files) under `swipl`, non-noisy, so it looks for a rule that separates the
     examples cleanly.
   - **LLM** runs chebILP's rule-generation pipeline on the session molecules. The
     model gets the concept's name, its **editable textual definition** and its
     sibling classes. It writes auxiliary ASP predicates and a hypothesis. The
     pipeline validates them on the examples, scores them (with one feedback round
     if the hypothesis is weak), and assembles one self-contained rule. The rule
     library is read-only: each run works on a temporary copy. A ChEBI class that
     already has a stored hypothesis is served from the library without a model
     call. The **Prompt the LLM** button then requests a fresh rule.
   - **Library predicates for Popper / Aleph.** When the concept has a definition,
     the `ICARUS_ILP_LIBRARY_TOP_N` library predicates most relevant to it (BM25
     over name and definition) are evaluated on the session molecules and offered
     as extra body predicates.

   The page shows the rule with a **natural-language translation**, a confusion
   matrix, and the molecules **not classified as labelled**. Each molecule the rule
   classifies as positive has a **"?"** button that explains why, in text and as a
   picture with the relevant atoms highlighted (chebILP's `explain_molecule` via
   xclingo).
4. **Edit and iterate.** Edit the rule with the block editor or as Prolog, apply it,
   and see which classifications changed. Every learned or edited rule is kept in a
   history and can be re-selected or re-scored against the current examples.

Example of a learned rule: `concept(V0) :- has_atom(V0,V1), o(V1).` ("the molecule
has an oxygen atom").

## Architecture

| Module | Role |
|--------|------|
| `icarus/config.py` | All paths, limits and settings (environment / `.env`) |
| `icarus/data_store.py` | Loads the ChEBI molecule table and hierarchy once; class search and example gathering |
| `icarus/session.py` | In-memory sessions, one per browser tab; SMILES/InChI parsing |
| `icarus/similarity.py` | Fingerprint pool (cached in `data/fingerprints.pkl`) and MMR suggestions |
| `icarus/ilp.py` | Builds the ILP problem, runs Popper, classifies with clingo, routes learning to a backend (`learn_dispatch`), NL translation and explanations |
| `icarus/aleph.py` | Aleph backend (chebILP `aleph_runner`), renames the learned head to `concept` |
| `icarus/llm_rulegen.py` | LLM backend: chebILP's `RuleGenerator`, run on the session molecules |
| `icarus/llm_guard.py` | Rate limits, prompt-size cap, time limit and prompt log around the model call |
| `icarus/predicates.py` | Building-block catalog for the visual rule editor |
| `icarus/render.py` | Server-side SVG structure drawings (RDKit) |
| `icarus/app.py` | Starlette JSON API and static files |
| `web/` | Single-page frontend (vanilla JS), tutorial and about pages |

## Limitations

- Sessions live in memory, one per browser tab (keyed by a random id in
  `sessionStorage`, so a reload keeps it and a duplicated tab starts fresh). They
  are not persisted, are dropped after 12 h idle (at most 200 are kept), and there
  is no login.
- The ILP search is kept small (low-arity atom predicates, bounded `max_vars` and
  `max_body`) so that learning returns quickly.
- The LLM backend uses only the atom-level background predicates (no molecular
  weight or ring-size facts), so every rule it writes can be evaluated with the
  same background as the classification step.
- Planned: functional-group predicates.
