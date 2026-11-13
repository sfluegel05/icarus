"""Extended Functional Groups (EFG) as library predicates.

chebILP can seed its rule library with the 583 EFG functional groups (Salmina, Haider
& Tetko 2016, vendored from ScoPy): one header-only ``efg_<name>`` predicate per group
whose extension is the RDKit match (``efg_<name>(M)``: molecule M contains the group),
not a clause body. Its retriever then splits the reuse candidates half EFG, half
generated ``aux_*`` predicates (``HybridPredicateRetriever._split_top``).

chebILP reads the EFG tables relative to the working directory (``data/…``), and the
PyPI package does not ship them, so icarus vendors them in ``icarus/resources/`` and
does the matching here. :func:`install` points chebILP's ``efg`` seed source at them,
for the LLM pipeline.
"""

from __future__ import annotations

import csv
import functools
import os
import re
import threading

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
EFG_PATH = os.path.join(_RES, "efg_functional_groups.tsv")
EFG_DESCRIPTIONS_PATH = os.path.join(_RES, "efg_descriptions.tsv")

PREFIX = "efg_"
_NAME_RE = re.compile(r"\b(efg_[A-Za-z0-9_]+)\s*\(")


def is_efg(name: str) -> bool:
    return name.startswith(PREFIX)


@functools.lru_cache(maxsize=1)
def vocabulary() -> tuple[tuple[str, str], ...]:
    """``((efg_<name>, description), …)`` for every group, as chebILP's
    ``efg_fg_vocabulary`` builds it (description ``<group>: <text>``)."""
    from chebILP.preprocessing.chebi.fg_matching import _efg_pred_name

    with open(EFG_DESCRIPTIONS_PATH, encoding="utf-8", newline="") as fh:
        texts = {r["Name"]: r["Description"] for r in csv.DictReader(fh, delimiter="\t")
                 if r["Description"]}
    vocab, seen = [], set()
    with open(EFG_PATH, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for idx, row in enumerate(reader):
            group = row[0]
            text = texts.get(group)
            vocab.append((_efg_pred_name(group, idx, seen),
                          f"{group}: {text}" if text else group))
    return tuple(vocab)


def descriptions() -> dict[str, str]:
    return dict(vocabulary())


@functools.lru_cache(maxsize=1)
def _patterns() -> dict[str, tuple[list, list]]:
    from chebILP.preprocessing.chebi.fg_matching import _load_efg_patterns

    return {name: (reject, accept) for name, reject, accept in _load_efg_patterns(EFG_PATH)}


# (smiles, group) -> bool. Molecules are matched lazily, one group at a time, and
# only for the groups a rule or retrieval actually uses.
_MATCH_CACHE: dict[tuple[str, str], bool] = {}
_MATCH_LOCK = threading.Lock()
_MATCH_CACHE_MAX = 500_000


def _matches(mol, key: str, name: str) -> bool:
    hit = _MATCH_CACHE.get((key, name))
    if hit is None:
        reject, accept = _patterns()[name]
        hit = (not any(mol.HasSubstructMatch(p) for p in reject)
               and all(mol.HasSubstructMatch(p) for p in accept))
        with _MATCH_LOCK:
            if len(_MATCH_CACHE) >= _MATCH_CACHE_MAX:
                _MATCH_CACHE.clear()
            _MATCH_CACHE[(key, name)] = hit
    return hit


def _mol_of(m):
    """``(rdkit mol, cache key)`` for a session molecule (``.mol``/``.smiles``)."""
    return m.mol, getattr(m, "smiles", None) or str(id(m.mol))


def fact_lines(mols, names) -> list[str]:
    """``efg_<name>(<mol id>).`` facts for the given session molecules and groups."""
    known = _patterns()
    names = [n for n in dict.fromkeys(names) if n in known]
    lines = []
    for m in mols:
        mol, key = _mol_of(m)
        if mol is None:
            continue
        lines += [f"{n}({m.id})." for n in names if _matches(mol, key, n)]
    return lines


def referenced(rule: str) -> list[str]:
    """The EFG predicates a rule text uses (in order of first use)."""
    known = _patterns()
    return [n for n in dict.fromkeys(_NAME_RE.findall(rule or "")) if n in known]


def rule_facts(rule: str, mols) -> list[str]:
    """The EFG facts ``rule`` needs to ground over ``mols`` (empty if it uses none)."""
    names = referenced(rule)
    return fact_lines(mols, names) if names else []


def retrieval_entries() -> list[dict]:
    """EFG groups as ``HybridPredicateRetriever`` entries (``stem`` = predicate name)."""
    return [{"name": n, "description": d, "kind": "rule", "stem": n} for n, d in vocabulary()]


def install() -> None:
    """Point chebILP's ``efg`` seed source at the vendored tables (vocabulary + matcher),
    so ``seed_fg_library(..., "efg")`` and the LLM pipeline's grounding of reused EFG
    predicates work regardless of the working directory."""
    from chebILP.preprocessing.chebi import fg_matching as fm

    src = fm.FG_SEED_SOURCES.get("efg")
    if src is None or getattr(src.matcher, "_icarus", False):
        return

    def matcher(rows, cache_path=None):
        return fm.get_efg_fgs(rows, efg_path=EFG_PATH)

    matcher._icarus = True
    fm.FG_SEED_SOURCES["efg"] = src._replace(matcher=matcher, vocabulary=lambda: list(vocabulary()))
