"""Explanations and the block-editor form of a concept rule, via viloch.

- :func:`explain_rule`: what the rule asks of a molecule (a picture of the pattern(s)
  and one or two sentences).
- :func:`explain_molecule`: why the rule does or does not accept one molecule (the
  molecule with the match, or the closest match, highlighted, and a sentence).
- :func:`editor_rule` / :func:`editor_model`: the hypothesis with its auxiliary
  predicates (or just one of them) unfolded, for the block editor.

A rule here is a whole icarus program: the ``concept`` hypothesis plus any auxiliary
programs it defines itself (an LLM rule's new predicates, with their
``% PREDICATE_NAME`` / ``% DESCRIPTION`` headers). Auxiliary predicates it only *uses*
come from the (read-only) rule library, and functional groups (``efg_*``) from
:mod:`viloch.chem.efg`.
"""

from __future__ import annotations

import functools

from viloch import api
from viloch.definition import Definition
from viloch.editor import asp
from viloch.editor import unfold as _unfold

from . import config, ilp

TARGET = config.TARGET_LABEL


# ── inputs ──────────────────────────────────────────────────────────────────


def definition(rule: str) -> Definition:
    """The rule as a viloch :class:`Definition`: the ``concept`` clauses, the programs
    the rule defines itself, then the library programs it relies on but does not
    define (transitively). Descriptions missing from the rule's headers come from the
    library."""
    defn = Definition.parse(rule or "", TARGET)
    defn = defn.with_descriptions(ilp.library_descriptions())
    # no library, or predicates it does not define: keep what can be resolved
    return defn.resolve(config.LLM_LIBRARY_DIR, strict=False)


# ── the unfolded rule (block editor) ────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def editor_rule(rule: str, only: str | None = None) -> str:
    """The rule's hypothesis with its auxiliary predicates unfolded into plain clauses
    (:func:`viloch.editor.unfold.editor_rule`); with ``only``, just that predicate."""
    return _unfold.editor_rule(definition(rule), only)


def editor_model(text: str) -> dict:
    """The block-editor model of ``concept`` clauses (:func:`viloch.editor.asp.from_asp`)."""
    return asp.from_asp(text or "", TARGET)


def compile_model(model: dict) -> str:
    """The ``concept`` clauses of a block-editor model."""
    return asp.to_asp(model, TARGET)


def unfold_pred(rule: str, pred: str) -> tuple[str, str | None]:
    """``(rule, error)``: the hypothesis with only ``pred`` unfolded, and why it stayed
    folded (None if it was unfolded)."""
    return _unfold.unfold_pred(definition(rule), pred)


# ── explanations ────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=64)
def explain_rule(rule: str) -> dict:
    """What ``rule`` asks of a molecule: ``{"text", "image" (PNG data URL)}``. Cached
    per rule text: drawing the figure takes a moment."""
    return api.as_json(api.explain_rule(definition(rule), strict=False))


def explain_molecule(mol, rule: str, name: str | None = None) -> dict:
    """Why ``rule`` does or does not accept the molecule (an RDKit mol or SMILES):
    ``{"fires", "text", "image"}`` with the match, or the closest match, highlighted."""
    return api.as_json(
        api.explain_molecule(mol, definition(rule), strict=False, name=name)
    )
