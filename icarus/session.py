"""In-memory session state for the ICaRuS demo.

A single global session is kept (demo, single user). A session holds a set of
molecules, each labelled positive / negative / unlabelled, plus the most recent
learned + (optionally) hand-edited hypothesis and the last classification map so
edits can be diffed.
"""

import itertools
import threading

from chebi_utils.read_molecule import smiles_or_inchi_to_mol

from . import data_store


def _clean(v):
    """Normalise pandas NaN / empty strings to None."""
    if v is None:
        return None
    try:
        import math

        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s or None


class Molecule:
    __slots__ = ("id", "smiles", "name", "chebi_id", "label", "mol", "inchi", "definition")

    def __init__(self, mid, smiles, name, chebi_id, label, mol, inchi=None, definition=None):
        self.id = mid
        self.smiles = smiles
        self.name = name
        self.chebi_id = chebi_id  # source ChEBI id, or None for user input
        self.label = label        # "pos" | "neg" | "unlabeled"
        self.mol = mol            # RDKit Mol (not serialized)
        self.inchi = inchi
        self.definition = definition

    def to_dict(self):
        return {
            "id": self.id,
            "smiles": self.smiles,
            "name": self.name,
            "chebi_id": self.chebi_id,
            "label": self.label,
            "inchi": self.inchi,
            "definition": self.definition,
        }


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.molecules: dict[str, Molecule] = {}
        self._counter = itertools.count(1)
        # Seen SMILES -> molecule id, to avoid duplicates.
        self._by_smiles: dict[str, str] = {}
        # Hypothesis state.
        self.learned_rule: str | None = None
        self.current_rule: str | None = None  # learned or hand-edited
        # mol_id -> bool prediction, from the last evaluation.
        self.last_predictions: dict[str, bool] = {}

    # ── molecule management ────────────────────────────────────────────────
    def _new_id(self) -> str:
        return f"m{next(self._counter)}"

    def add_molecule(self, smiles, name, chebi_id, label, mol, inchi=None, definition=None) -> Molecule | None:
        key = smiles
        if key in self._by_smiles:
            existing = self.molecules[self._by_smiles[key]]
            # Re-labelling an existing molecule is allowed.
            if label != "unlabeled":
                existing.label = label
            return existing
        mid = self._new_id()
        m = Molecule(mid, smiles, name, chebi_id, label, mol, inchi=inchi, definition=definition)
        self.molecules[mid] = m
        self._by_smiles[key] = mid
        return m

    def add_from_chebi_id(self, chebi_id: str, label: str) -> Molecule | None:
        df = data_store.molecules()
        if chebi_id not in df.index:
            return None
        row = df.loc[chebi_id]
        smiles = row["smiles"]
        name = row.get("ChEBI NAME") or data_store.class_name(chebi_id)
        mol = row["mol"]
        if mol is None or smiles is None:
            return None
        inchi = row.get("INCHI")
        definition = row.get("DEFINITION")
        # Use the RDKit-canonical SMILES as the stored form, matching what
        # add_from_text stores — otherwise the same structure added from both
        # sources would not deduplicate and would evade suggestion exclusion.
        from rdkit import Chem

        try:
            canonical = Chem.MolToSmiles(mol)
        except Exception:
            canonical = str(smiles)
        return self.add_molecule(
            canonical, str(name), chebi_id, label, mol,
            inchi=_clean(inchi), definition=_clean(definition),
        )

    def add_from_text(self, text: str, label: str) -> tuple[list[Molecule], list[str]]:
        """Parse one-or-more SMILES/InChI lines and add them. Returns (added, errors)."""
        added, errors = [], []
        for raw in text.splitlines():
            s = raw.strip()
            if not s:
                continue
            mol = smiles_or_inchi_to_mol(s)
            if mol is None:
                errors.append(s)
                continue
            from rdkit import Chem

            try:
                canonical = Chem.MolToSmiles(mol)
            except Exception:
                canonical = s
            try:
                inchi = Chem.MolToInchi(mol) or None
            except Exception:
                inchi = None
            m = self.add_molecule(canonical, canonical, None, label, mol, inchi=inchi)
            if m is not None:
                added.append(m)
        return added, errors

    def remove(self, mol_id: str) -> bool:
        m = self.molecules.pop(mol_id, None)
        if m is None:
            return False
        self._by_smiles.pop(m.smiles, None)
        self.last_predictions.pop(mol_id, None)
        return True

    def set_label(self, mol_id: str, label: str) -> bool:
        m = self.molecules.get(mol_id)
        if m is None:
            return False
        m.label = label
        return True

    def labeled(self) -> list[Molecule]:
        return [m for m in self.molecules.values() if m.label in ("pos", "neg")]

    def positives(self) -> list[Molecule]:
        return [m for m in self.molecules.values() if m.label == "pos"]

    def clear(self):
        self.molecules.clear()
        self._by_smiles.clear()
        self._counter = itertools.count(1)
        self.learned_rule = None
        self.current_rule = None
        self.last_predictions.clear()


# The single global demo session.
SESSION = Session()
