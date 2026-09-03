"""Loads the ChEBI molecule dataset and hierarchy graph once, and provides
class search plus example gathering from a ChEBI class.

Everything here is read-only reference data shared across sessions.
"""

import functools
import pickle

import networkx as nx
import pandas as pd

from . import config


@functools.lru_cache(maxsize=1)
def molecules() -> pd.DataFrame:
    """ChEBI molecule DataFrame, indexed by ChEBI id (string). Columns include
    ``smiles``, ``INCHI``, ``ChEBI NAME`` and the RDKit ``mol`` object."""
    print(f"[data_store] loading molecules from {config.MOLECULES_PKL} ...", flush=True)
    df = pd.read_pickle(config.MOLECULES_PKL)
    df.index = df.index.astype(str)
    print(f"[data_store] {len(df)} molecules loaded", flush=True)
    return df


@functools.lru_cache(maxsize=1)
def graph() -> nx.DiGraph:
    """ChEBI hierarchy graph (nodes = ChEBI ids, attrs include ``name``)."""
    print(f"[data_store] loading graph from {config.GRAPH_PKL} ...", flush=True)
    with open(config.GRAPH_PKL, "rb") as f:
        return pickle.load(f)


@functools.lru_cache(maxsize=1)
def transitive_closure() -> nx.DiGraph:
    """Transitive closure of the hierarchy: ``predecessors(id)`` = all descendants."""
    print("[data_store] computing transitive closure ...", flush=True)
    return nx.transitive_closure_dag(graph())


@functools.lru_cache(maxsize=1)
def valid_molecule_ids() -> frozenset:
    """ChEBI ids of concrete molecules only — SMILES present and free of any
    wildcard / R-group (``*``). Generic class structures are never valid
    molecules, so they are excluded everywhere: as positive or negative examples
    and as similarity suggestions."""
    df = molecules()
    valid = {
        str(cid) for cid, smi in df["smiles"].items()
        if isinstance(smi, str) and smi and "*" not in smi
    }
    print(f"[data_store] {len(valid)} concrete (non-wildcard) molecules", flush=True)
    return frozenset(valid)


def class_name(chebi_id: str) -> str:
    g = graph()
    if chebi_id in g.nodes:
        return g.nodes[chebi_id].get("name") or f"CHEBI:{chebi_id}"
    return f"CHEBI:{chebi_id}"


def class_definition(chebi_id: str) -> str | None:
    """The ChEBI textual definition of a class, or ``None`` if it has none."""
    g = graph()
    if chebi_id in g.nodes:
        return g.nodes[chebi_id].get("definition")
    return None


def search_classes(query: str, limit: int = 25) -> list[dict]:
    """Search ChEBI classes by id or (case-insensitive) name substring.

    Only returns classes that actually have descendant molecules in the dataset,
    so every hit can seed positive examples.
    """
    query = query.strip().lower()
    if not query:
        return []
    g = graph()
    mol_index = set(molecules().index)
    tc = transitive_closure()

    results = []
    for node, attrs in g.nodes(data=True):
        name = attrs.get("name") or ""
        if query in node.lower() or query in name.lower():
            # Count descendant molecules (self included).
            descendants = set(tc.predecessors(node)) | {node}
            n_mol = len(descendants & mol_index)
            if n_mol == 0:
                continue
            results.append({"id": node, "name": name, "n_mol": n_mol})
            if len(results) >= limit * 4:
                # Enough candidates to rank well; stop scanning the ~200k-node graph
                # (each match also does a descendant set-intersection, so this bounds
                # the per-keystroke cost).
                break
    # Rank: exact id match first, then name-startswith, then by molecule count.
    def rank(r):
        starts = 0 if r["name"].lower().startswith(query) or r["id"] == query else 1
        return (starts, -r["n_mol"])

    results.sort(key=rank)
    return results[:limit]


@functools.lru_cache(maxsize=1)
def _undirected() -> nx.Graph:
    """Undirected view of the is_a hierarchy (cached), for BFS by hierarchy distance."""
    return graph().to_undirected()


def _heavy_atom_count(chebi_id: str) -> int:
    """Number of heavy atoms of a ChEBI molecule (fallback: large, so it sorts last)."""
    df = molecules()
    try:
        mol = df.loc[chebi_id, "mol"]
        return mol.GetNumHeavyAtoms() if mol is not None else 10 ** 6
    except Exception:
        return 10 ** 6


def _closest_negatives(chebi_id, candidates, pos_descendants, max_samples) -> list[str]:
    """Molecule ids closest to ``chebi_id`` in the hierarchy but not below it.

    BFS outward over the undirected is_a graph, collecting descendant molecules of
    each visited neighbour, until ``max_samples`` are found. Reuses the cached
    transitive closure, so it is fast, and explicitly excludes the target's own
    descendants (which would otherwise leak in through a parent node).
    """
    import collections

    tc = transitive_closure()
    und = _undirected()
    q = collections.deque([chebi_id])
    visited = {chebi_id}
    selected, seen = [], set()
    while q:
        current = q.popleft()
        for nb in und.neighbors(current):
            if nb in visited:
                continue
            visited.add(nb)
            q.append(nb)
            for sub in tc.predecessors(nb):
                s = str(sub)
                if s in candidates and s not in pos_descendants and s not in seen:
                    seen.add(s)
                    selected.append(s)
                    if len(selected) >= max_samples:
                        return selected
        if len(selected) >= max_samples:
            break
    return selected


def gather_class_examples(chebi_id: str, max_pos: int, max_neg: int) -> tuple[list[str], list[str]]:
    """Return ``(positive_ids, negative_ids)`` ChEBI molecule ids for a class.

    Only **concrete** molecules are used (wildcard/R-group structures are excluded
    globally — see :func:`valid_molecule_ids`). Positives are the **smallest**
    descendant molecules (fewest heavy atoms) so the minimal core motif is exposed;
    negatives are the molecules **closest to the class in the hierarchy** but not
    below it (near-misses), topped up with random negatives only if the
    neighbourhood yields fewer than requested.
    """
    import random

    tc = transitive_closure()
    valid = valid_molecule_ids()

    pos_descendants = set(tc.predecessors(chebi_id)) | {chebi_id}

    # Positives: smallest concrete descendant molecules first.
    pos_all = list(pos_descendants & valid)
    if not pos_all:  # class has no concrete molecules — fall back to all descendants
        pos_all = list(pos_descendants & set(molecules().index))
    pos = sorted(pos_all, key=lambda cid: (_heavy_atom_count(cid), cid))[:max_pos]

    # Negatives: closest concrete molecules in the hierarchy, not below the target.
    neg = _closest_negatives(chebi_id, valid, pos_descendants, max_neg)

    # Top up with random concrete negatives if the neighbourhood was too small.
    if len(neg) < max_neg:
        rng = random.Random(42)
        others = list(valid - pos_descendants - set(neg))
        rng.shuffle(others)
        neg += others[: max_neg - len(neg)]

    return pos, neg
