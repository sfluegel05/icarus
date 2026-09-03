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


def aux_blocks(library_dir: str) -> list[dict]:
    """Blocks for the LLM-generated ``aux_*`` predicates in the session library.

    Names + descriptions come from ``rule_to_nl.load_aux_descriptions``; the arity
    (parsed from each program head) decides the scope — arity 1 is treated as a
    molecule-level property, arity 2 as an atom relation. These are only a palette
    default: the Prolog→block parser places an aux usage by its actual argument
    role, so a rule always round-trips regardless of this hint.
    """
    if not library_dir or not os.path.isdir(library_dir):
        return []
    try:
        from chebILP.explainability.rule_to_nl import (
            load_aux_descriptions, _format_aux_name,
        )
    except Exception:
        return []

    descriptions = load_aux_descriptions(library_dir)
    if not descriptions:
        return []

    programs_dir = os.path.join(library_dir, "programs")
    if not os.path.isdir(programs_dir):
        programs_dir = library_dir

    blocks = []
    for name in sorted(descriptions):
        arity = None
        for ext in (".pl", ".py"):
            path = os.path.join(programs_dir, name + ext)
            if os.path.exists(path):
                arity = _aux_head_arity(path) if ext == ".pl" else None
                break
        arity = arity if arity in (1, 2) else 1
        label = _format_aux_name(name) or name
        desc = descriptions.get(name) or ""
        if arity == 2:
            scope, nl = "bond", f"forms “{label}” with"
        else:
            scope, nl = "molecule", f"the molecule has the property “{label}”"
        blocks.append({
            "id": f"aux_{name}",
            "category": "Generated (LLM)",
            "scope": scope,
            "popular": False,
            "label": label,
            "nl": nl,
            "description": desc,
            "input": {"type": "toggle", "pred": name},
        })
    return blocks
