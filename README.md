# ICaRuS — Interactive ChemicAl RUle System

A minimal demo prototype. A chemistry domain expert describes a chemical concept
as a **Prolog rule**, learned from positive and negative example molecules by one
of three interchangeable backends: the ILP systems **Popper** (noisy/MDL) and
**Aleph**, or an **LLM**. Examples come from a ChEBI class
and/or from SMILES / InChI input, and the tool suggests more molecules by
fingerprint similarity. The learned rule is shown, its misclassifications are
flagged, and the expert can hand-edit the rule (seeing which classifications
change) and iterate.

This extends the per-class ChEBI approach in `../chebILP` to arbitrary molecules.

## What it does (the flow)

1. **Examples** — optionally pick a ChEBI class to auto-fill the example boxes.
   Only **concrete molecules** are ever used — generic ChEBI class structures with
   R-group/wildcard atoms (`*`) are excluded everywhere (examples and
   suggestions). Positives are the **smallest descendant molecules** (fewest heavy
   atoms) so the minimal class core is used; negatives are the molecules
   **closest to the class in the hierarchy** but not below it (near-misses via a
   BFS outward over the `is_a` graph), topped up with random negatives only if the
   neighbourhood is small. Add more molecules by pasting SMILES or InChI.
   Molecules are shown as **2D
   structures** (RDKit); click one to reveal its SMILES / InChI / ChEBI name /
   ID / definition. Move any molecule between the positive / negative boxes or
   remove it.
2. **Suggestions** — ICaRuS proposes molecules related to your positives from a
   pool of ChEBI molecules, scored by Morgan-fingerprint (ECFP4) Tanimoto
   similarity. To stay informative it selects a **diverse** set by Maximal
   Marginal Relevance — `score = λ·rel − (1−λ)·max similarity to already-picked`
   — balancing closeness to the positives against distance from the other
   suggestions, so you get varied scaffolds instead of near-duplicates. They are
   shown **one at a time** (a "tinder"-style card): the structure is primary, with
   name / id / definition secondary, and — once a rule exists — the rule's
   prediction shown so you can confirm or contradict it as you add it +/−.
3. **Learn** — pick a **method** (Popper / Aleph / LLM) and the labelled molecules
   are turned into a rule, run in a background thread whose **live output** streams
   into the UI (last few lines shown, full log on demand). All three reuse chebILP's
   atom-level FOL translation and produce a rule with the same `concept/1` head, so
   everything downstream is method-agnostic:
   - **Popper** — the molecules become an ILP problem (`exs.pl` / `bk.pl` /
     `bias.pl`) and Popper learns a rule in noisy mode (MDL cost function, via
     chebILP's training subprocess), returning the best rule found even when none
     perfectly separates the examples.
   - **Aleph** — the same background + (bias-filtered) predicates become an Aleph
     `.b` / `.f` / `.n` problem, learned under `swipl` (chebILP `aleph_runner`),
     configured non-noisy so it seeks a cleanly-separating rule.
   - **LLM** — drives chebILP's actual rule pipeline (`generate_auxiliary_rules`'s
     `RuleGenerator`, sourced from the session molecules): the local `claude` CLI is
     prompted with the concept's name, **textual definition** and sibling classes to
     write auxiliary ASP predicates and a class hypothesis; the predicates are
     validated/grounded, the hypothesis is scored (train-F1) with one feedback round
     if weak, and the kept auxiliary clauses plus the `concept(A) :- …` hypothesis are
     assembled into one self-contained rule. The **concept definition** is shown in an
     editable box once you seed from a ChEBI class — edit it and the edited text is
     what the model sees (in place of the original ChEBI definition). Predicates the
     pipeline writes are kept in a session-local library and reused across learns.
     A ChEBI class that already has a stored hypothesis (`hypotheses.json`) is served
     from the library without a new model call (rescored on the current examples).
   - **Library-augmented background (Popper / Aleph)** — when the concept has a
     definition, the `ICARUS_ILP_LIBRARY_TOP_N` (default 8) library predicates most
     relevant to it (chebILP's `HybridPredicateRetriever`, BM25 over name +
     definition) are grounded on the session molecules and offered as extra body
     predicates; a learned rule using them carries their definitions along.
   The rule is shown alongside its **natural-language translation** (chebILP `rule_to_nl`),
   with a confusion matrix and the molecules **"not classified as told"** (false
   positives / negatives). Every molecule the rule marks positive gets a **"?"**
   button showing a graphical + textual explanation of *why* (chebILP
   `explain_molecule` via xclingo, with the responsible atoms highlighted).
4. **Edit & iterate** — hand-edit the Prolog rule and apply it to see exactly
   which classifications changed; revert to the learned rule; add more samples,
   re-suggest, and re-learn.

## Architecture

- **`icarus/config.py`** — paths (defaults point at the sibling `chebILP` data)
  and demo constants.
- **`icarus/data_store.py`** — loads the ChEBI v251 molecule DataFrame and
  hierarchy graph once; class search and example gathering.
- **`icarus/session.py`** — in-memory session (single user); SMILES/InChI parsing
  via `chebi_utils.read_molecule`.
- **`icarus/similarity.py`** — fingerprint pool (cached to `data/fingerprints.pkl`)
  and similarity suggestions.
- **`icarus/ilp.py`** — builds the ILP problem with
  `chebILP.ilp_problem_builder.build_background_chemlog` +
  `chebi_utils.extract_properties`, runs Popper via chebILP's training
  subprocess, classifies molecules with clingo (`evaluate_with_clingo`), and
  dispatches a learn request to one of the three backends (`learn_dispatch`).
- **`icarus/aleph.py`** — Aleph backend: writes the `.b` / `.f` / `.n` problem
  (`build_aleph_background`) and runs it via chebILP's `aleph_runner`, rewriting the
  learned head to `concept`.
- **`icarus/llm_rulegen.py`** — LLM backend: reuses chebILP's
  `generate_auxiliary_rules` prompt/contract, structured `claude`-CLI call,
  predicate grounding and hypothesis scoring, driven over the session molecules.
- **`icarus/app.py`** — Starlette JSON API + static file serving.
- **`web/`** — single-page frontend (vanilla JS).

The learned rule defines a generic target predicate `concept/1`, e.g.
`concept(V0) :- has_atom(V0,V1), o(V1).` ("the molecule has an oxygen atom").

## Running

The app reuses the `chebILP/.wslvenv` interpreter, which already has Popper,
clingo, RDKit, `chebi_utils` and `chebILP` installed. From this repo:

```bash
wsl -e bash run.sh
```

Then open <http://localhost:8000>. First launch builds the fingerprint pool
(~30 s, cached afterwards).

Quick offline check of the ILP pipeline (no server):

```bash
PYTHONPATH=. .wslvenv/bin/python smoke_test.py   # run from chebILP, or adjust PYTHONPATH
```

## Running on another system

Every filesystem path lives in **`icarus/config.py`**, driven entirely by
environment variables — there are no paths hardcoded elsewhere (the interpreter
in `run.sh` is the only other one, also an env var). To port the app you need
three things: this repo, a Python environment, and two data files.

**1. This repo.** Copy the whole `icarus/` project (the `icarus/` package,
`web/`, `run.sh`, `requirements.txt`). It writes only into `ICARUS_DATA_DIR`
(default `icarus/data/`), which it creates itself — the fingerprint cache
(`fingerprints.pkl`) and the generated ILP files (`work/exs.pl`, `bk.pl`,
`bias.pl`) are all produced at runtime, nothing to copy.

**2. A Python environment** (3.11+). Install `requirements.txt`, plus the three
local packages from their sibling checkouts and Popper's own prerequisites:

```bash
pip install -r requirements.txt
pip install -e ../python-chebi-utils     # chebi_utils
pip install -e ../chebILP                # chebILP
pip install -e ../popper-sfluegel        # popper (ILP engine; see its README for system prereqs)
```

Point `run.sh` at this interpreter with `ICARUS_PYTHON` (or keep the default
`$ICARUS_CHEBILP_DIR/.wslvenv/bin/python`).

**3. Two ChEBI data files**, read-only, placed under the directory
`ICARUS_CHEBILP_DIR` points at (default: the sibling `chebILP` checkout):

| File | Location (relative to `ICARUS_CHEBILP_DIR`) |
|------|---------------------------------------------|
| Molecule DataFrame (52k mols: SMILES, InChI, name, RDKit mol) | `data/chebi_v251/ChEBI25_3_STAR/molecules.pkl` |
| ChEBI hierarchy graph (networkx DiGraph) | `data/chebi_v251/chebi_graph.pkl` |

Copy them from an existing chebILP checkout, or regenerate from a chebILP install
with `python -m chebILP prepare_dataset --chebi_version 251`. On startup the
server prints whether both files were located and names any that are missing.

**Configuration knobs** (all optional; see `run.sh` header):

| Env var | Default | Purpose |
|---------|---------|---------|
| `ICARUS_CHEBILP_DIR` | `/mnt/c/.../chebILP` | Holds the `data/chebi_v<version>/…` files above |
| `ICARUS_PYTHON` | `$ICARUS_CHEBILP_DIR/.wslvenv/bin/python` | Interpreter `run.sh` launches |
| `ICARUS_CHEBI_VERSION` | `251` | ChEBI release the data is for |
| `ICARUS_DATA_DIR` | `icarus/data` | Writable dir: fingerprint cache + ILP work files |
| `ICARUS_POOL_SIZE` | `12000` | Molecules fingerprinted for suggestions |
| `ICARUS_PORT` | `8000` | HTTP port |
| `ICARUS_LLM_MODEL` | `claude-haiku-4-5` | Model id for the LLM backend (local `claude` CLI) |

The LLM backend calls the locally logged-in `claude` CLI (via the Claude Agent SDK)
and bills to that subscription; it needs the CLI installed and `/login`-ed, and the
Aleph backend needs `swipl` on `PATH` (both already present in the `chebILP/.wslvenv`
setup).

## Notes / limitations (it's a demo)

- Single global in-memory session; no persistence, auth, or concurrency.
- Learning runs in a background thread (the UI polls for live output); the `timeout`
  bounds the Popper and Aleph searches, while the LLM call is bounded by the CLI.
- The Popper/Aleph bias is kept small (low-arity atom predicates, `max_vars`/`max_body`
  bounded) so learning returns quickly. Popper runs in noisy (MDL) mode — the best rule
  found, which need not separate the examples perfectly; Aleph runs non-noisy.
- The LLM backend offers only the atom-level background predicates (no `mol_weight` /
  `ring_size` computed facts), so every rule it writes grounds through the same clingo
  background the classify step builds.
- Future work (per the project plan): functional-group predicates.
