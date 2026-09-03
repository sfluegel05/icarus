"""In-memory session state for the ICaRuS demo.

A single global session is kept (demo, single user). A session holds a set of
molecules, each labelled positive / negative / unlabelled, plus the most recent
learned + (optionally) hand-edited hypothesis and the last classification map so
edits can be diffed.
"""

import itertools
import threading
import time

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


class RuleEntry:
    """One rule in the session's history.

    Created whenever a rule is learned (``origin`` = the backend: ``popper`` /
    ``aleph`` / ``llm``) or a hand-edited rule is applied (``origin`` = ``edited``).
    ``confusion`` is the {TP,FP,TN,FN} matrix computed at creation (or the last
    re-evaluation); ``signature`` snapshots the labelled example set it was scored
    against, so a later change to the examples can be flagged as making it stale.
    """

    __slots__ = ("id", "origin", "rule", "confusion", "signature", "created_at")

    def __init__(self, rid, origin, rule, confusion, signature, created_at):
        self.id = rid
        self.origin = origin
        self.rule = rule
        self.confusion = confusion            # {"TP","FP","TN","FN"} or None
        self.signature = signature            # labelled-set snapshot when scored
        self.created_at = created_at

    def to_dict(self, current_signature):
        return {
            "id": self.id,
            "origin": self.origin,
            "rule": self.rule,
            "confusion": self.confusion,
            "created_at": self.created_at,
            # Outdated if the labelled examples changed since this was scored.
            "stale": self.confusion is not None and self.signature != current_signature,
        }


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.molecules: dict[str, Molecule] = {}
        self._counter = itertools.count(1)
        # Seen SMILES -> molecule id, to avoid duplicates.
        self._by_smiles: dict[str, str] = {}
        # Hypothesis history: every learned/edited rule is kept; one is selected.
        self.rule_history: list[RuleEntry] = []
        self.selected_rule_id: str | None = None
        self._rule_counter = itertools.count(1)
        # mol_id -> bool prediction, from the last evaluation.
        self.last_predictions: dict[str, bool] = {}
        # Concept description shown to the LLM: name + the (editable) definition
        # text, seeded from a ChEBI class when one is used. ``concept_chebi_id`` ties
        # the LLM pipeline's library entries to that class for reuse.
        self.concept_name: str | None = None
        self.concept_chebi_id: str | None = None
        self.concept_definition: str | None = None

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

    # ── rule history ───────────────────────────────────────────────────────
    def labeled_signature(self) -> tuple:
        """A snapshot of the labelled example set (id + label), used to detect
        when a stored confusion matrix has gone stale."""
        return tuple(sorted((m.id, m.label) for m in self.labeled()))

    @property
    def current_rule(self) -> str | None:
        e = self.selected_entry()
        return e.rule if e else None

    def selected_entry(self) -> "RuleEntry | None":
        if self.selected_rule_id is None:
            return None
        for e in self.rule_history:
            if e.id == self.selected_rule_id:
                return e
        return None

    def add_rule(self, origin: str, rule: str, confusion: dict | None = None) -> "RuleEntry":
        """Append a new rule to the history and make it the selected one."""
        entry = RuleEntry(f"r{next(self._rule_counter)}", origin, rule, confusion,
                          self.labeled_signature(), time.time())
        self.rule_history.append(entry)
        self.selected_rule_id = entry.id
        return entry

    def select_rule(self, rule_id: str) -> bool:
        if any(e.id == rule_id for e in self.rule_history):
            self.selected_rule_id = rule_id
            return True
        return False

    def set_concept(self, name: str | None, chebi_id: str | None, definition: str | None):
        """Record the concept a seeding ChEBI class describes (name + definition).

        The definition is only overwritten while the user has not begun editing one
        for a *different* concept — re-seeding the same class refreshes it, switching
        class replaces it, but adding more molecules from the same class keeps any
        edit the user has since made."""
        if chebi_id and chebi_id == self.concept_chebi_id and self.concept_definition is not None:
            self.concept_name = name or self.concept_name
            return
        self.concept_name = name
        self.concept_chebi_id = chebi_id
        self.concept_definition = definition

    def clear(self):
        self.molecules.clear()
        self._by_smiles.clear()
        self._counter = itertools.count(1)
        self.rule_history.clear()
        self.selected_rule_id = None
        self._rule_counter = itertools.count(1)
        self.last_predictions.clear()
        self.concept_name = None
        self.concept_chebi_id = None
        self.concept_definition = None


# The single global demo session.
SESSION = Session()
