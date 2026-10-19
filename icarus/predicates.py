"""The predicate catalog that drives the visual building-block rule editor.

Single source of truth for the blocks the UI offers, kept in lock-step with the
molecule formalization in ``chebi_utils.extract_properties.mol_to_fol_atoms`` (the
same atom-level predicates the Popper/Aleph/clingo pipeline learns and classifies
over). The block→Prolog compile and Prolog→block parse both live in the browser
(``web/blocks.js``); this module only describes the vocabulary.

Each block template is a JSON-able dict:

    {
      "id":       stable id,
      "category": palette group heading,
      "scope":    "molecule" | "atom" | "bond",
      "popular":  shown by default (vs. behind the "more" toggle),
      "label":    readable name for the block,
      "nl":       readable predicate phrase (matches rule_to_nl phrasing),
      "input":    one of
                    {"type": "toggle",  "pred": "<name>"}
                    {"type": "choice",  "options": [{"label","pred","nl"?}], "default"?}
                    {"type": "element"}     # free-text element symbol → lowercased pred
    }

``aux_blocks`` adds dynamic blocks for the LLM-generated ``aux_*`` predicates once
they exist in the session rule library, so a generated concept becomes a reusable
building block.
"""

import functools
import json
import os
import re


# Common elements offered as ready-made blocks / recognised by name. Arbitrary
# elements are still expressible through the free-text "Other element" block; the
# formalization emits the lowercased element symbol as the predicate.
COMMON_ELEMENTS = [
    ("c", "Carbon"), ("n", "Nitrogen"), ("o", "Oxygen"), ("s", "Sulfur"),
    ("p", "Phosphorus"), ("f", "Fluorine"), ("cl", "Chlorine"), ("br", "Bromine"),
    ("i", "Iodine"), ("h", "Hydrogen"), ("b", "Boron"), ("si", "Silicon"),
    ("se", "Selenium"),
]

# Element symbols the popular row surfaces directly.
_POPULAR_ELEMENTS = {"c", "n", "o"}

# Ranges kept small: they cover essentially every value the formalization emits
# for demo-sized molecules, and keep the choice controls readable.
_CHARGE_RANGE = range(-2, 3)      # -2 .. +2
_H_RANGE = range(0, 5)            # exact H count 0 .. 4
_RING_SIZES = [3, 4, 5, 6, 7, 8]
_STEROID_POSITIONS = range(1, 18)  # steroid_1 .. steroid_17


def _charge_pred(n: int) -> str:
    """Exact-charge predicate name, matching extract_properties.get_atom_properties."""
    if n == 0:
        return "charge0"
    return f"charge_m{-n}" if n < 0 else f"charge{n}"


def _charge_nl(n: int) -> str:
    if n == 0:
        return "has no formal charge"
    return f"has formal charge {'+' if n > 0 else '-'}{abs(n)}"


def _element_blocks() -> list[dict]:
    blocks = []
    for sym, name in COMMON_ELEMENTS:
        article = "an" if name[0].lower() in "aeiou" else "a"
        blocks.append({
            "id": f"el_{sym}",
            "category": "Element",
            "scope": "atom",
            "popular": sym in _POPULAR_ELEMENTS,
            "label": name,
            "nl": f"is {article} {name.lower()} atom",
            "input": {"type": "toggle", "pred": sym},
        })
    blocks.append({
        "id": "el_other",
        "category": "Element",
        "scope": "atom",
        "popular": False,
        "label": "Other element…",
        "nl": "is an atom of a given element",
        "input": {"type": "element"},
    })
    return blocks


def catalog() -> list[dict]:
    """The full static block catalog (everything but the dynamic aux blocks)."""
    blocks: list[dict] = []

    # ── Element ──────────────────────────────────────────────────────────────
    blocks += _element_blocks()

    # ── Hydrogens ────────────────────────────────────────────────────────────
    blocks.append({
        "id": "h_one",
        "category": "Hydrogens",
        "scope": "atom",
        "popular": True,
        "label": "Has 1 hydrogen",
        "nl": "has one hydrogen atom",
        "input": {"type": "toggle", "pred": "has_1_hs"},
    })
    blocks.append({
        "id": "h_exact",
        "category": "Hydrogens",
        "scope": "atom",
        "popular": False,
        "label": "Number of hydrogens",
        "nl": "has a given number of hydrogen atoms",
        "input": {
            "type": "choice",
            "default": "has_1_hs",
            "options": [
                {"label": ("no" if n == 0 else str(n)), "pred": f"has_{n}_hs",
                 "nl": f"has {'no' if n == 0 else n} hydrogen atom{'' if n == 1 else 's'}"}
                for n in _H_RANGE
            ],
        },
    })
    blocks.append({
        "id": "h_at_least",
        "category": "Hydrogens",
        "scope": "atom",
        "popular": False,
        "label": "At least N hydrogens",
        "nl": "has at least a given number of hydrogen atoms",
        "input": {
            "type": "choice",
            "default": "has_at_least_1_hs",
            "options": [
                {"label": f"≥ {n}", "pred": f"has_at_least_{n}_hs",
                 "nl": f"has at least {n} hydrogen atom{'' if n == 1 else 's'}"}
                for n in range(1, 5)
            ],
        },
    })

    # ── Charge ───────────────────────────────────────────────────────────────
    blocks.append({
        "id": "charge_exact",
        "category": "Charge",
        "scope": "atom",
        "popular": False,
        "label": "Formal charge",
        "nl": "has a given formal charge",
        "input": {
            "type": "choice",
            "default": "charge0",
            "options": [
                {"label": ("0" if n == 0 else f"{'+' if n > 0 else '−'}{abs(n)}"),
                 "pred": _charge_pred(n), "nl": _charge_nl(n)}
                for n in _CHARGE_RANGE
            ],
        },
    })
    blocks.append({
        "id": "charge_sign",
        "category": "Charge",
        "scope": "atom",
        "popular": False,
        "label": "Charge sign",
        "nl": "has a charge of a given sign",
        "input": {
            "type": "choice",
            "default": "charge0",
            "options": [
                {"label": "neutral", "pred": "charge0", "nl": "has no formal charge"},
                {"label": "positive", "pred": "charge_p", "nl": "has a positive formal charge"},
                {"label": "negative", "pred": "charge_n", "nl": "has a negative formal charge"},
            ],
        },
    })

    # ── Rings ────────────────────────────────────────────────────────────────
    blocks.append({
        "id": "in_ring",
        "category": "Rings",
        "scope": "atom",
        "popular": True,
        "label": "In a ring",
        "nl": "is in a ring",
        "input": {"type": "toggle", "pred": "in_ring"},
    })
    blocks.append({
        "id": "in_ring_n",
        "category": "Rings",
        "scope": "atom",
        "popular": False,
        "label": "In an N-membered ring",
        "nl": "is in a ring of a given size",
        "input": {
            "type": "choice",
            "default": "in_ring6",
            "options": [
                {"label": f"{n}-ring", "pred": f"in_ring{n}",
                 "nl": f"is in a {n}-membered ring"}
                for n in _RING_SIZES
            ],
        },
    })

    # ── Stereochemistry ──────────────────────────────────────────────────────
    blocks.append({
        "id": "cip",
        "category": "Stereochemistry",
        "scope": "atom",
        "popular": False,
        "label": "CIP chirality",
        "nl": "has a given CIP configuration",
        "input": {
            "type": "choice",
            "default": "cip_code_R",
            "options": [
                {"label": "R", "pred": "cip_code_R", "nl": "has CIP code R"},
                {"label": "S", "pred": "cip_code_S", "nl": "has CIP code S"},
            ],
        },
    })

    # ── Steroid nucleus ──────────────────────────────────────────────────────
    blocks.append({
        "id": "steroid",
        "category": "Steroid nucleus",
        "scope": "atom",
        "popular": False,
        "label": "Steroid position",
        "nl": "is at a given steroid-nucleus position",
        "input": {
            "type": "choice",
            "default": "steroid_1",
            "options": [
                {"label": str(n), "pred": f"steroid_{n}",
                 "nl": f"is at steroid position {n}"}
                for n in _STEROID_POSITIONS
            ],
        },
    })

    # ── Bonds / relations (connect two atom areas) ───────────────────────────
    blocks.append({
        "id": "bond_any",
        "category": "Bonds",
        "scope": "bond",
        "popular": True,
        "label": "Bonded",
        "nl": "is bonded to",
        "input": {"type": "toggle", "pred": "has_bond_to"},
    })
    blocks.append({
        "id": "bond_type",
        "category": "Bonds",
        "scope": "bond",
        "popular": False,
        "label": "Bond type",
        "nl": "has a given bond to",
        "input": {
            "type": "choice",
            "default": "bSINGLE",
            "options": [
                {"label": "single", "pred": "bSINGLE", "nl": "has a single bond to"},
                {"label": "double", "pred": "bDOUBLE", "nl": "has a double bond to"},
                {"label": "triple", "pred": "bTRIPLE", "nl": "has a triple bond to"},
                {"label": "aromatic", "pred": "bAROMATIC", "nl": "has an aromatic bond to"},
            ],
        },
    })
    blocks.append({
        "id": "bond_stereo",
        "category": "Bonds",
        "scope": "bond",
        "popular": False,
        "label": "Stereo bond",
        "nl": "has a stereo bond to",
        "input": {
            "type": "choice",
            "default": "bSTEREOE",
            "options": [
                {"label": "E", "pred": "bSTEREOE", "nl": "has a stereo bond to (E configuration)"},
                {"label": "Z", "pred": "bSTEREOZ", "nl": "has a stereo bond to (Z configuration)"},
                {"label": "cis", "pred": "bSTEREOCIS", "nl": "has a stereo bond to (CIS configuration)"},
                {"label": "trans", "pred": "bSTEREOTRANS", "nl": "has a stereo bond to (TRANS configuration)"},
            ],
        },
    })

    # ── Molecule-level (global) properties ───────────────────────────────────
    blocks.append({
        "id": "aromaticity",
        "category": "Molecule",
        "scope": "molecule",
        "popular": True,
        "label": "Aromaticity",
        "nl": "the molecule's aromaticity",
        "input": {
            "type": "choice",
            "default": "aromatic",
            "options": [
                {"label": "aromatic", "pred": "aromatic", "nl": "the molecule is aromatic"},
                {"label": "aliphatic", "pred": "aliphatic", "nl": "the molecule is aliphatic"},
            ],
        },
    })
    blocks.append({
        "id": "net_charge",
        "category": "Molecule",
        "scope": "molecule",
        "popular": True,
        "label": "Net charge",
        "nl": "the molecule's net charge",
        "input": {
            "type": "choice",
            "default": "net_charge_neutral",
            "options": [
                {"label": "neutral", "pred": "net_charge_neutral", "nl": "the molecule has no net charge"},
                {"label": "positive", "pred": "net_charge_positive", "nl": "the molecule has a positive net charge"},
                {"label": "negative", "pred": "net_charge_negative", "nl": "the molecule has a negative net charge"},
            ],
        },
    })

    return blocks


# ── dynamic LLM auxiliary-predicate blocks ──────────────────────────────────
_HEAD_ARITY_RE = re.compile(r"^\s*([a-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:-", re.MULTILINE)


def _aux_head_arity(program_path: str) -> int | None:
    """Arity of an aux predicate from the head of its ``.pl`` program, or None."""
    try:
        with open(program_path, encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return None
    m = _HEAD_ARITY_RE.search(source)
    if not m:
        return None
    args = m.group(2).strip()
    return 0 if not args else len([a for a in args.split(",") if a.strip()])


_ID_SUFFIX_RE = re.compile(r"_(\d{4,})$")
_LOG_NAME_RE = re.compile(r"^chebi_(\d+)\.md$")
_LOG_TIME_RE = re.compile(r"^- Timestamp:\s*(.+?)\s*$", re.MULTILINE)
_LOG_NEW_RE = re.compile(r"^- New rules added:\n((?:[ \t]+- .*\n?)+)", re.MULTILINE)


def _origins(library_dir: str) -> dict[str, str]:
    """``aux_name -> ChEBI id`` of the class each library predicate was first
    generated for: the earliest generation log listing it as a new rule, falling
    back to the first class in ``class_map.json`` that uses it. (A ``_<id>`` name
    suffix overrides this, see :func:`_origin`.)"""
    first_seen: dict[str, tuple[str, str]] = {}
    logs = os.path.join(library_dir, "generation_logs")
    if os.path.isdir(logs):
        for fname in os.listdir(logs):
            m = _LOG_NAME_RE.match(fname)
            if not m:
                continue
            try:
                with open(os.path.join(logs, fname), encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            new = _LOG_NEW_RE.search(text)
            if not new:
                continue
            t = _LOG_TIME_RE.search(text)
            seen = (t.group(1) if t else "~", m.group(1))  # undated logs sort last
            for name in re.findall(r"`(aux_\w+)`", new.group(1)):
                if name not in first_seen or seen < first_seen[name]:
                    first_seen[name] = seen
    origins = {n: cid for n, (_, cid) in first_seen.items()}
    try:
        with open(os.path.join(library_dir, "class_map.json"), encoding="utf-8") as f:
            for cid, names in json.load(f).items():
                for n in names:
                    origins.setdefault(n, str(cid))
    except (OSError, ValueError):
        pass
    return origins


def _origin(name: str, origins: dict[str, str]) -> str | None:
    # A ``_<id>`` suffix is the generator's own disambiguation of a name clash.
    m = _ID_SUFFIX_RE.search(name)
    return m.group(1) if m else origins.get(name)


@functools.lru_cache(maxsize=4)
def aux_index(library_dir: str) -> tuple[dict, dict]:
    """Every predicate of the (read-only) library, loaded once:
    ``({name: (description, arity, label, origin_chebi_id)}, {label: count})``.
    Several predicates share a readable label (``aux_acyclic_molecule`` and
    ``aux_acyclic_molecule_10283`` both read "acyclic molecule"); the counts mark
    the labels that need the ChEBI id to tell them apart."""
    try:
        from chebILP.explainability.rule_to_nl import load_aux_descriptions, _format_aux_name
    except Exception:
        return {}, {}
    if not (library_dir and os.path.isdir(library_dir)):
        return {}, {}
    programs_dir = os.path.join(library_dir, "programs")
    if not os.path.isdir(programs_dir):
        programs_dir = library_dir

    def arity(name):
        for ext in (".pl", ".py"):
            path = os.path.join(programs_dir, name + ext)
            if os.path.exists(path):
                return _aux_head_arity(path) if ext == ".pl" else None
        return None

    origins = _origins(library_dir)
    entries, counts = {}, {}
    for name, desc in load_aux_descriptions(library_dir).items():
        label = _format_aux_name(name) or name
        entries[name] = (desc or "", arity(name), label, _origin(name, origins))
        counts[label] = counts.get(label, 0) + 1
    return entries, counts


def _aux_block(name, desc, arity, label, chebi_id, ambiguous) -> dict:
    """Palette block for one generated predicate. ``ambiguous``: its readable label
    is shared with another predicate, so the label carries the ChEBI id of the class
    it was generated for."""
    if ambiguous and chebi_id:
        label = f"{label} (CHEBI:{chebi_id})"
    # An arity-1 generated predicate may hold of the molecule OR of a single
    # atom — the arity alone doesn't say which — so scope "any" lets the user
    # drop it onto the molecule area or an atom, wherever it belongs. Arity-2
    # is a relation between two atoms (drawn with the bond/connect handle).
    if arity == 2:
        scope, nl = "bond", f"forms “{label}” with"
    else:
        scope, nl = "any", f"has the property “{label}”"
    return {
        "id": f"aux_{name}",
        "category": "Generated (LLM)",
        "scope": scope,
        "popular": False,
        "label": label,
        "nl": nl,
        "description": desc or "",
        "chebi_id": chebi_id,
        "input": {"type": "toggle", "pred": name},
    }


def aux_blocks(library_dir: str, names=None, rule_descriptions=None, rule=None) -> list[dict]:
    """Blocks for the LLM-generated ``aux_*`` predicates in the session library,
    restricted to ``names`` when given (``None`` = the whole library), plus those the
    current ``rule`` defines itself (``rule_descriptions``: name → description from
    its headers) — predicates a fresh LLM run generated are not in the read-only
    library.

    Names + descriptions come from ``rule_to_nl.load_aux_descriptions``; the arity
    (parsed from each program head) decides the scope — arity 1 is treated as a
    molecule-level property, arity 2 as an atom relation. These are only a palette
    default: the Prolog→block parser places an aux usage by its actual argument
    role, so a rule always round-trips regardless of this hint.
    """
    try:
        from chebILP.explainability.rule_to_nl import _format_aux_name
    except Exception:
        return []

    entries, counts = aux_index(library_dir)
    counts = dict(counts)
    chosen = [(n, *e) for n, e in entries.items() if names is None or n in names]

    def rule_arity(name):
        for m in _HEAD_ARITY_RE.finditer(rule or ""):
            if m.group(1) == name:
                args = m.group(2).strip()
                return 0 if not args else len([a for a in args.split(",") if a.strip()])
        return None

    for n, d in (rule_descriptions or {}).items():
        if n in entries:
            continue
        label = _format_aux_name(n) or n
        counts[label] = counts.get(label, 0) + 1
        m = _ID_SUFFIX_RE.search(n)
        chosen.append((n, d, rule_arity(n), label, m.group(1) if m else None))

    return [_aux_block(n, d, a, label, cid, counts.get(label, 0) > 1)
            for n, d, a, label, cid in sorted(chosen)]


def search_aux_blocks(library_dir: str, query: str, limit: int = 50,
                      offset: int = 0) -> tuple[list[dict], int]:
    """Search the WHOLE generated-predicate library (not just the predicates
    relevant to the session's class). Every whitespace-separated term must occur in
    the predicate name, readable label, description or origin ChEBI id; an empty
    query lists everything. Label matches rank before description-only matches.
    Returns ``(blocks[offset:offset+limit], total)``."""
    entries, counts = aux_index(library_dir)
    terms = (query or "").lower().replace("chebi:", "").split()
    q = " ".join(terms)
    hits = []
    for name, (desc, _, label, cid) in entries.items():
        low_name, low_label = name.lower(), label.lower()
        if not all(t in f"{low_name} {low_label} {desc.lower()} {cid or ''}" for t in terms):
            continue
        rank = (0 if low_label.startswith(q) else
                1 if all(t in low_label or t in low_name for t in terms) else 2)
        hits.append((rank, low_label, name))
    hits.sort()
    blocks = []
    for _, _, n in hits[offset:offset + limit]:
        desc, arity, label, cid = entries[n]
        blocks.append(_aux_block(n, desc, arity, label, cid, counts[label] > 1))
    return blocks, len(hits)
