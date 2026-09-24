# Reinstalling ICaRuS on a new machine

A checklist of everything that must be set up or checked when moving ICaRuS to a
fresh machine. ICaRuS is a thin app on top of the sibling `chebILP` project — most
of the caveats are about wiring up those external dependencies, data files, and
API config, none of which live in this repo.

See also the README's *Running on another system* section for the narrative
version; this file is the caveat-oriented checklist.

## 0. Commit/push pending changes first

A fresh `git clone` only gets what's committed. Before migrating, make sure the
working tree is clean (`git status`) and everything is **pushed** — uncommitted
changes (especially in `icarus/config.py`) do not travel with a clone.

## 1. Sibling local packages (not vendored, not in this repo)

Check these out next to the location `ICARUS_CHEBILP_DIR` points at and install
each editable:

```bash
pip install -e ../python-chebi-utils     # chebi_utils
pip install -e ../chebILP                # chebILP
pip install -e ../popper-sfluegel        # popper (ILP engine; has its own system prereqs)
```

## 2. Python environment (Linux / WSL)

- The app runs under a **Linux** interpreter (default `chebILP/.wslvenv/bin/python`)
  via `wsl -e bash run.sh`. There is no separate icarus venv.
- `pip install -r requirements.txt` plus the three editable installs above.
- Keep **rdkit** close to the pinned version — `molecules.pkl` holds pickled
  RDKit `Mol` objects and a mismatched rdkit can fail to unpickle them.

## 3. Two ChEBI data files (not in git)

`data/` is gitignored, and these files live in the chebILP checkout anyway. Place
under `ICARUS_CHEBILP_DIR/data/chebi_v251/`:

| File | Path (relative to `ICARUS_CHEBILP_DIR`) |
|------|------------------------------------------|
| Molecule DataFrame (~52k mols) | `data/chebi_v251/ChEBI25_3_STAR/molecules.pkl` |
| ChEBI hierarchy graph (networkx DiGraph) | `data/chebi_v251/chebi_graph.pkl` |

Copy them from an existing chebILP checkout, or regenerate:

```bash
python -m chebILP prepare_dataset --chebi_version 251
```

On startup the server prints which required files were located and names any
missing ones (`config.check_data_files`).

## 4. LLM / API config (biggest caveat)

Which credentials you need depends on `ICARUS_LLM_MODEL`:

- **Default `openai/agent_d7-…` (qwen endpoint)** → needs `OPENAI_API_BASE` and
  `OPENAI_API_KEY` set in **`chebILP/.env`**. That `.env` is secret and **not
  committed** — recreate it by hand on the new machine. `config.py` loads it
  explicitly (running from the icarus dir, dotenv's cwd-upward search never reaches
  the sibling `.env`).
- **`claude-haiku-4-5` (bare model id)** → routes through the locally installed
  **`claude` CLI**, which must be installed *and* `/login`-ed; it bills that
  subscription. Set `ICARUS_LLM_MODEL=claude-haiku-4-5` to use this path.
- `ANTHROPIC_API_KEY` is also read from `chebILP/.env` when present.

The **Popper** and **Aleph** backends need no API key.

## 5. External tools on PATH

- `swipl` (SWI-Prolog) — required by the **Aleph** backend.
- `xclingo` + `pillow`/PIL — for the explanation feature (in `requirements.txt`).
- `claude` CLI — only for the bare-model LLM path (see above).
- Popper (`../popper-sfluegel`) may have additional system prerequisites — see its
  README.

## 6. Hardcoded paths → override with env vars

Defaults bake in `/mnt/c/Users/sifluegel/PycharmProjects/chebILP` (in `config.py`
and `run.sh`). On a new machine/user set at minimum:

| Env var | Default | Purpose |
|---------|---------|---------|
| `ICARUS_CHEBILP_DIR` | `/mnt/c/.../chebILP` | Holds the data files + sibling packages |
| `ICARUS_PYTHON` | `$ICARUS_CHEBILP_DIR/.wslvenv/bin/python` | Interpreter `run.sh` launches |
| `ICARUS_CHEBI_VERSION` | `251` | ChEBI release the data is for |
| `ICARUS_DATA_DIR` | `icarus/data` | Writable dir: fingerprint cache + ILP work files |
| `ICARUS_POOL_SIZE` | `12000` | Molecules fingerprinted for suggestions |
| `ICARUS_PORT` | `8000` | HTTP port |
| `ICARUS_LLM_MODEL` | `openai/agent_d7-…` | LLM backend model id (see §4) |

## 7. First-run / runtime notes (nothing to copy, just expect)

- `ICARUS_DATA_DIR` (default `icarus/data/`) is auto-created. The fingerprint cache
  (`fingerprints.pkl`, ~30 s to build the first time) and the generated ILP files
  (`work/exs.pl`, `bk.pl`, `bias.pl`) are all rebuilt on demand — nothing to copy.
- The LLM **rule library** (`data/rule_library/`) is session-local and gitignored;
  it starts empty on the new machine, which is fine.

## Quick sanity check

```bash
wsl -e bash run.sh
```

Watch the startup line for missing-data-file warnings, open
<http://localhost:8000>, and test the **Popper** backend first (no API keys, no
extra PATH tools) before exercising the Aleph (`swipl`) or LLM (`.env` / `claude`
CLI) paths.
