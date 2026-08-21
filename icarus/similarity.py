"""Fingerprint-similarity suggestions over a pool of ChEBI molecules.

A random subset of the ChEBI dataset is fingerprinted once (Morgan / ECFP4) and
cached to disk. Given the session's positive molecules, we rank the pool by
maximum Tanimoto similarity to any positive and return the closest, unseen
candidates.
"""

import os
import pickle
import random

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from . import config, data_store

_RADIUS = 2
_NBITS = 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=_RADIUS, fpSize=_NBITS)

# Loaded lazily: (list[chebi_id], list[ExplicitBitVect]).
_POOL_IDS: list[str] | None = None
_POOL_FPS: list | None = None


def _morgan(mol):
    try:
        return _GEN.GetFingerprint(mol)
    except Exception:
        return None


def _build_pool():
    df = data_store.molecules()
    # Concrete molecules only — never suggest wildcard/R-group structures.
    ids = list(data_store.valid_molecule_ids())
    rng = random.Random(7)
    rng.shuffle(ids)
    ids = ids[: config.SIMILARITY_POOL_SIZE]

    pool_ids, pool_fps = [], []
    for cid in ids:
        mol = df.loc[cid, "mol"]
        if mol is None:
            continue
        fp = _morgan(mol)
        if fp is None:
            continue
        pool_ids.append(cid)
        pool_fps.append(fp)
    return pool_ids, pool_fps


def _load_pool():
    global _POOL_IDS, _POOL_FPS
    if _POOL_IDS is not None:
        return
    if os.path.exists(config.FP_CACHE):
        print(f"[similarity] loading fingerprint cache {config.FP_CACHE}", flush=True)
        with open(config.FP_CACHE, "rb") as f:
            data = pickle.load(f)
        _POOL_IDS = data["ids"]
        _POOL_FPS = data["fps"]  # ExplicitBitVect objects pickle directly
        return
    print("[similarity] building fingerprint pool (one-time, ~30s) ...", flush=True)
    _POOL_IDS, _POOL_FPS = _build_pool()
    with open(config.FP_CACHE, "wb") as f:
        pickle.dump({"ids": _POOL_IDS, "fps": _POOL_FPS}, f)
    print(f"[similarity] pool ready: {len(_POOL_IDS)} molecules", flush=True)


# MMR trade-off: weight on relevance (closeness to positives) vs. diversity
# (distance to already-selected suggestions). 1.0 = pure similarity, 0.0 = pure
# spread. 0.65 keeps candidates clearly related to the positives while avoiding
# near-duplicates of each other.
_MMR_LAMBDA = 0.65


def suggest(positive_mols, exclude_chebi_ids, exclude_smiles, n=10, mmr_lambda=_MMR_LAMBDA) -> list[dict]:
    """Suggest a *diverse* set of molecules related to the positives.

    Relevance is max Tanimoto similarity to any positive. Rather than returning
    the top-``n`` (which cluster into near-duplicates), candidates are chosen by
    greedy Maximal Marginal Relevance:

        score(c) = λ · rel(c) − (1 − λ) · max_{s ∈ selected} sim(c, s)

    so each pick trades closeness to the positives against dissimilarity to the
    already-chosen suggestions. Skips anything already in the session.
    """
    _load_pool()
    pos_fps = [fp for fp in (_morgan(m.mol) for m in positive_mols) if fp is not None]
    if not pos_fps or not _POOL_FPS:
        return []

    df = data_store.molecules()
    exclude_chebi_ids = set(exclude_chebi_ids)
    exclude_smiles = set(exclude_smiles)

    # 1. Relevance of every (non-excluded) pool molecule to the positives.
    scored = []  # (rel, cid, fp)
    for cid, fp in zip(_POOL_IDS, _POOL_FPS):
        if cid in exclude_chebi_ids:
            continue
        rel = max(DataStructs.BulkTanimotoSimilarity(fp, pos_fps))
        scored.append((rel, cid, fp))
    scored.sort(key=lambda t: t[0], reverse=True)

    # 2. Restrict MMR to a relevance shortlist (bounds cost and keeps everything
    #    at least loosely related to the positives).
    shortlist = scored[: max(n * 25, 150)]
    if not shortlist:
        return []

    # 3. Greedy MMR selection.
    rels = [t[0] for t in shortlist]
    fps = [t[2] for t in shortlist]
    remaining = set(range(len(shortlist)))
    max_sim_to_selected = [0.0] * len(shortlist)  # updated incrementally
    order = []

    first = max(remaining, key=lambda i: rels[i])  # most relevant seeds the set
    order.append(first)
    remaining.discard(first)

    while remaining and len(order) < n * 3:  # over-select; smiles filter trims later
        just = order[-1]
        rem_list = list(remaining)
        sims = DataStructs.BulkTanimotoSimilarity(fps[just], [fps[i] for i in rem_list])
        for i, s in zip(rem_list, sims):
            if s > max_sim_to_selected[i]:
                max_sim_to_selected[i] = s
        best_i = max(
            remaining,
            key=lambda i: mmr_lambda * rels[i] - (1 - mmr_lambda) * max_sim_to_selected[i],
        )
        order.append(best_i)
        remaining.discard(best_i)

    # 4. Materialise, skipping session-duplicate structures, until we have n.
    out = []
    for i in order:
        _, cid, _ = shortlist[i]
        row = df.loc[cid]
        smiles = str(row["smiles"]) if row["smiles"] is not None else None
        if smiles is None:
            continue
        # Compare on canonical SMILES: the session stores canonical forms, so a
        # raw-dataset match alone would miss structures the user typed in.
        mol = row["mol"]
        canon = smiles
        if mol is not None:
            try:
                from rdkit import Chem

                canon = Chem.MolToSmiles(mol)
            except Exception:
                pass
        if smiles in exclude_smiles or canon in exclude_smiles:
            continue
        definition = row.get("DEFINITION")
        try:
            import math

            if isinstance(definition, float) and math.isnan(definition):
                definition = None
        except Exception:
            pass
        out.append(
            {
                "chebi_id": cid,
                "name": str(row.get("ChEBI NAME") or f"CHEBI:{cid}"),
                "smiles": smiles,
                "definition": str(definition) if definition else None,
                "similarity": round(float(rels[i]), 3),
            }
        )
        if len(out) >= n:
            break
    return out
