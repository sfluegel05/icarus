"""In-memory session state for the ICaRuS demo.

Each browser tab gets its own session (keyed by a client-generated id, see
:class:`SessionStore`), so tabs and concurrent users never see each other's work. A
session holds a set of molecules, each labelled positive / negative / unlabelled, plus the most recent
learned + (optionally) hand-edited hypothesis and the last classification map so
edits can be diffed.
"""

import datetime
import itertools
import math
import re
import threading
import time

from chebi_utils.read_molecule import smiles_or_inchi_to_mol
from rdkit import Chem

from . import data_store


def _clean(v):
    """Normalise pandas NaN / empty strings to None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
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

    __slots__ = ("id", "origin", "rule", "confusion", "signature", "created_at", "model")

    def __init__(self, rid, origin, rule, confusion, signature, created_at, model=None):
        self.id = rid
        self.origin = origin
        self.rule = rule
        self.confusion = confusion            # {"TP","FP","TN","FN"} or None
        self.signature = signature            # labelled-set snapshot when scored
        self.created_at = created_at
        self.model = model                    # LLM rules: which model wrote it

    def to_dict(self, current_signature):
        return {
            "id": self.id,
            "origin": self.origin,
            "rule": self.rule,
            "confusion": self.confusion,
            "created_at": self.created_at,
            "model": self.model,
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
        # Live-learning state, updated by the background learner thread and polled
        # by the UI; guarded by ``learn_lock`` (not ``lock``, so polling never waits
        # on a slow session operation).
        self.learn_lock = threading.Lock()
        self.learn = new_learn_state()
        self.last_used = time.time()
        # LLM learn runs that prompted the model, with their exchanges (for the
        # conversation download); the newest MAX_LLM_RUNS are kept.
        self.llm_runs: list[dict] = []
        # LLM prompts this session has sent (llm_guard's per-session quota). Not reset
        # by clear(), so clearing the session doesn't refill the quota.
        self.llm_prompts = 0
        # Set once any request of this session carried a valid invite code (lifts the
        # LLM limits). Sticky, so requests without the header can't unset it mid-run.
        self.invited = False

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
        try:
            canonical = Chem.MolToSmiles(mol)
        except Exception:
            canonical = str(smiles)
        return self.add_molecule(
            canonical, str(name), chebi_id, label, mol,
            inchi=_clean(inchi), definition=_clean(definition),
        )

    def add_from_structure(self, s: str, label: str, name: str | None = None) -> Molecule | None:
        """Parse one SMILES/InChI string and add it (None if it doesn't parse)."""
        mol = smiles_or_inchi_to_mol(s)
        if mol is None:
            return None
        try:
            canonical = Chem.MolToSmiles(mol)
        except Exception:
            canonical = s
        try:
            inchi = Chem.MolToInchi(mol) or None
        except Exception:
            inchi = None
        return self.add_molecule(canonical, name or canonical, None, label, mol, inchi=inchi)

    def add_from_text(self, text: str, label: str) -> tuple[list[Molecule], list[str]]:
        """Parse one-or-more SMILES/InChI lines and add them. Returns (added, errors)."""
        added, errors = [], []
        for raw in text.splitlines():
            s = raw.strip()
            if not s:
                continue
            m = self.add_from_structure(s, label)
            if m is None:
                errors.append(s)
            else:
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

    def add_rule(self, origin: str, rule: str, confusion: dict | None = None,
                 model: str | None = None) -> "RuleEntry":
        """Append a new rule to the history and make it the selected one."""
        entry = RuleEntry(f"r{next(self._rule_counter)}", origin, rule, confusion,
                          self.labeled_signature(), time.time(), model=model)
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
        self.llm_runs.clear()

    def add_llm_run(self, run: dict):
        self.llm_runs.append(run)
        del self.llm_runs[:-MAX_LLM_RUNS]

    # ── export / import ────────────────────────────────────────────────────
    def to_export(self) -> dict:
        """The session as a human-readable, JSON-serialisable document (read back by
        :meth:`load_export`). Rules are split into lines; the confusion matrices are
        informational only and are recomputed on import."""
        def mol(m):
            d = {"name": m.name} if m.name and m.name != m.smiles else {}
            if m.chebi_id:
                d["chebi_id"] = m.chebi_id
            d["smiles"] = m.smiles
            return d

        return {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "exported_at": _iso(time.time()),
            "concept": {"name": self.concept_name, "chebi_id": self.concept_chebi_id,
                        "definition": self.concept_definition},
            "examples": {key: [mol(m) for m in self.molecules.values() if m.label == label]
                         for key, label in _EXPORT_LABELS.items()},
            # Oldest first; "selected" marks the rule shown in the editor.
            "rules": [{
                "origin": e.origin,
                "model": e.model,
                "created_at": _iso(e.created_at),
                "selected": e.id == self.selected_rule_id,
                "confusion": e.confusion,
                "rule": e.rule.splitlines(),
            } for e in self.rule_history],
        }

    def load_export(self, doc, max_molecules: int, max_rules: int,
                    max_definition: int) -> list[str]:
        """Replace this session's examples, concept and rule history with an exported
        document. The document is validated first (a ``ValueError`` leaves the session
        untouched); molecules that can't be resolved are skipped and returned as
        warnings. Rules are added unscored."""
        if not isinstance(doc, dict) or doc.get("format") != EXPORT_FORMAT:
            raise ValueError("Not an ICaRuS session file.")
        if not isinstance(doc.get("version"), int) or doc["version"] > EXPORT_VERSION:
            raise ValueError("Unsupported session file version.")
        concept = doc.get("concept") or {}
        examples = doc.get("examples") or {}
        rules = doc.get("rules") or []
        if not (isinstance(concept, dict) and isinstance(examples, dict)
                and isinstance(rules, list) and all(isinstance(r, dict) for r in rules)
                and all(isinstance(examples.get(k) or [], list) for k in _EXPORT_LABELS)):
            raise ValueError("Malformed session file.")
        mols = [(label, m) for key, label in _EXPORT_LABELS.items()
                for m in examples.get(key) or []]
        if not all(isinstance(m, dict) for _, m in mols):
            raise ValueError("Malformed session file.")
        if len(mols) > max_molecules:
            raise ValueError(f"Too many molecules ({len(mols)}, limit {max_molecules}).")
        if len(rules) > max_rules:
            raise ValueError(f"Too many rules ({len(rules)}, limit {max_rules}).")
        definition = _str_or_none(concept.get("definition"))
        if definition and len(definition) > max_definition:
            raise ValueError(f"The concept definition is too long ({len(definition)} "
                             f"characters, limit {max_definition}).")
        parsed_rules = []
        for r in rules:
            text = r.get("rule")
            if isinstance(text, list):
                text = "\n".join(str(line) for line in text)
            text = _str_or_none(text)
            if not text:
                raise ValueError("A rule in the session file is empty.")
            origin = r.get("origin") if r.get("origin") in RULE_ORIGINS else "edited"
            model = (_str_or_none(r.get("model")) or "")[:100] or None
            parsed_rules.append((origin, text, model, _parse_iso(r.get("created_at")),
                                 bool(r.get("selected"))))

        self.clear()
        self.concept_name = _str_or_none(concept.get("name"))
        self.concept_chebi_id = _str_or_none(concept.get("chebi_id"))
        self.concept_definition = definition
        skipped = []
        for label, m in mols:
            chebi_id = _str_or_none(m.get("chebi_id"))
            smiles = _str_or_none(m.get("smiles"))
            name = _str_or_none(m.get("name"))
            # Prefer the ChEBI entry (brings name, InChI, definition); fall back to the
            # structure for user-entered molecules or ids missing from this ChEBI version.
            added = self.add_from_chebi_id(chebi_id, label) if chebi_id else None
            if added is None and smiles:
                added = self.add_from_structure(smiles, label, name)
            if added is None:
                skipped.append(name or smiles or chebi_id or "?")
        selected_id = None
        for origin, text, model, created_at, selected in parsed_rules:
            entry = self.add_rule(origin, text, model=model)
            entry.created_at = created_at
            if selected:
                selected_id = entry.id
        if selected_id:
            self.selected_rule_id = selected_id  # add_rule left the newest selected
        return skipped


MAX_LLM_RUNS = 20
EXPORT_FORMAT = "icarus-session"
EXPORT_VERSION = 1
_EXPORT_LABELS = {"positive": "pos", "negative": "neg", "unlabeled": "unlabeled"}
RULE_ORIGINS = ("popper", "aleph", "llm", "edited")


def _str_or_none(v) -> str | None:
    if isinstance(v, int) and not isinstance(v, bool):
        v = str(v)
    return (v.strip() or None) if isinstance(v, str) else None


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(v) -> float:
    try:
        dt = datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return time.time()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


def new_learn_state(method: str | None = None) -> dict:
    return {"running": False, "done": False, "error": None, "lines": [],
            "message": None, "score": None, "method": method, "cached": False}


# Session ids are generated client-side (a random UUID per tab); anything else is
# rejected so arbitrary strings can't be used to fill the store.
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class SessionStore:
    """All live sessions, keyed by the per-tab session id.

    Sessions idle for longer than ``ttl`` seconds are dropped (unless a learn is still
    running for them), and at most ``max_sessions`` are kept — the least recently
    used idle session is evicted first."""

    def __init__(self, ttl: float = 12 * 3600, max_sessions: int = 200):
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    @staticmethod
    def valid_id(sid) -> bool:
        return isinstance(sid, str) and bool(_SID_RE.match(sid))

    def get(self, sid: str) -> Session:
        """The session for ``sid``, created on first use."""
        now = time.time()
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                self._evict(now)
                s = self._sessions[sid] = Session()
            s.last_used = now
            return s

    def _evict(self, now: float):
        idle = [(s.last_used, sid) for sid, s in self._sessions.items()
                if not s.learn["running"]]
        for last_used, sid in idle:
            if now - last_used > self.ttl:
                del self._sessions[sid]
        excess = len(self._sessions) - self.max_sessions + 1
        if excess > 0:
            for _, sid in sorted(i for i in idle if i[1] in self._sessions)[:excess]:
                del self._sessions[sid]

    def __len__(self):
        return len(self._sessions)


SESSIONS = SessionStore()
