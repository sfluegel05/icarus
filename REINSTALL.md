# Reinstalling ICaRuS on a new machine

## Requirements
Use Python version 3.14.
- `pip install -r requirements.txt` (chebi-utils and chebILP have only been tested with local editable version, but should line up)
- Keep **rdkit** close to the pinned version — `molecules.pkl` holds pickled
  RDKit `Mol` objects and a mismatched rdkit can fail to unpickle them.
- [SWI-Prolog](https://www.swi-prolog.org/Download.html) and NuWLS (cf. https://github.com/logic-and-learning-lab/Popper) need to be on the PATH

## data files (not in git)

`data/` is gitignored. Place the files in `ICARUS_CHEBILP_DIR/data/`:

| File | Path (relative to `ICARUS_CHEBILP_DIR`) |
|------|------------------------------------------|
| Molecule DataFrame (~52k mols) | `data/chebi_v251/ChEBI25_3_STAR/molecules.pkl` |
| ChEBI hierarchy graph (networkx DiGraph) | `data/chebi_v251/chebi_graph.pkl` |

Download them, or regenerate:

```bash
python -m chebILP prepare_dataset --chebi_version 251
```

On startup the server prints which required files were located and names any
missing ones (`config.check_data_files`).

Also, there is a rule library folder (optional). Unzip and place it in `data/rule_library`

## LLM / API config - don't do this for now

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


## 6. Hardcoded paths → override with env vars

Defaults bake in `/mnt/c/Users/sifluegel/PycharmProjects/chebILP` (in `config.py`
and `run.sh`). On a new machine/user set at minimum:

| Env var | Default | Purpose |
|---------|---------|---------|
| `ICARUS_CHEBILP_DIR` | `.` | Holds the data files + sibling packages |
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
