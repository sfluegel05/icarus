"""Explanations of a concept rule, built with chebILP's explainability pipeline.

- :func:`explain_rule` — what the rule asks of a molecule: a picture of the pattern(s)
  and one or two sentences.
- :func:`explain_molecule` — why the rule does or does not accept one molecule: the
  molecule with the match (or the closest match) highlighted, and a sentence.
- :func:`editor_rule` — the hypothesis with its auxiliary predicates unfolded, for the
  block editor.

A rule here is a whole icarus program: the ``concept`` hypothesis plus any auxiliary
programs it defines itself (an LLM rule's new predicates, with their ``% PREDICATE_NAME``
/ ``% DESCRIPTION`` headers). Auxiliary predicates it only *uses* come from the
(read-only) rule library, and functional groups (``efg_*``) from :mod:`icarus.efg`.
"""

from __future__ import annotations

import base64
import functools
import io
import re

from . import config, efg, ilp

TARGET = config.TARGET_LABEL


# ── inputs ──────────────────────────────────────────────────────────────────


def rule_programs(rule: str) -> tuple[str, list[dict]]:
    """``(hypothesis, aux programs)`` for chebILP's explainer.

    The hypothesis is the rule's ``concept`` clauses. The aux programs
    (``{"name", "description", "source"}``) are the predicates the rule defines itself,
    then the library programs it relies on but does not define (transitively).
    """
    order, groups = [], {}
    for clause in ilp._split_clauses(rule or ""):
        head = ilp._clause_head(clause)
        if head is None:
            continue
        if head not in groups:
            groups[head] = []
            order.append(head)
        groups[head].append(clause)
    hypothesis = "\n".join(groups.get(TARGET, []))
    descriptions = {**ilp.library_descriptions(), **ilp.rule_aux_descriptions(rule)}
    programs = [{"name": n, "description": descriptions.get(n, ""),
                 "source": "\n".join(groups[n])} for n in order if n != TARGET]
    defined = set(groups)
    for dep in _library_dependencies(rule):
        if dep.name not in defined:
            defined.add(dep.name)
            programs.append({"name": dep.name, "source": dep.source,
                             "description": dep.description or descriptions.get(dep.name, "")})
    return hypothesis, programs


def _library_dependencies(rule: str) -> list:
    """Library programs (``RuleProgram``) of the ``aux_*`` predicates ``rule`` uses."""
    if "aux_" not in (rule or ""):
        return []
    from chebILP.predicate_generation.auxiliary_rules import (
        RuleProgram, resolve_rule_dependencies,
    )

    try:
        return resolve_rule_dependencies([RuleProgram(name="__rule__", source=rule)],
                                         config.LLM_LIBRARY_DIR)
    except Exception:
        return []


def _png_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── the unfolded rule (block editor) ────────────────────────────────────────


def _restore_anonymous(items, bound: set[str]) -> list:
    """``items`` with each variable that occurs only in negated literals turned back
    into ``_``. Unfolding renames every ``_`` of an inlined definition apart, which
    inside ``not …`` would make the clause unsafe."""
    from chebILP.explainability.rule_ir import Agg, AggElem, Lit, ordered_vars

    bound = bound | set(ordered_vars([b for b in items if not (isinstance(b, Lit) and b.neg)]))

    def unbound_var(term: str) -> bool:
        return (term[:1].isupper() or term[:1] == "_") and term not in bound

    out = []
    for b in items:
        if isinstance(b, Lit) and b.neg:
            b = Lit(b.pred, tuple("_" if unbound_var(a) else a for a in b.args), True)
        elif isinstance(b, Agg):
            b = Agg(b.func, tuple(AggElem(e.terms, tuple(_restore_anonymous(
                list(e.cond), bound | set(ordered_vars([" ".join(e.terms)]))))) for e in b.elems),
                b.lguard, b.rguard)
        out.append(b)
    return out


@functools.lru_cache(maxsize=256)
def editor_rule(rule: str) -> str:
    """The rule's hypothesis with its auxiliary predicates unfolded into plain clauses,
    for the block editor: chebILP's ``rule_ir.unfold``, so a multi-clause definition
    becomes several (OR) clauses. A negated auxiliary predicate stays as written (the
    editor has no block for a negated group), as does anything chebILP's unfolding policy
    keeps folded (recursion, too many alternatives); those still need their definitions,
    which the rule keeps. Equivalent to the hypothesis under the rule's definitions."""
    from chebILP.explainability.rule_ir import (
        UnfoldPolicy, definitions, ordered_vars, parse_program, unfold,
    )

    hypothesis, programs = rule_programs(rule)
    if not programs or not hypothesis:
        return hypothesis
    try:
        clauses = parse_program(hypothesis)
        defs = definitions(programs)
    except Exception:
        return hypothesis
    support = "\n".join(p["source"] for p in programs)
    policy = UnfoldPolicy(inline_negation=False)
    lines = []
    for clause in clauses:
        # Unfolded clause by clause: one that does not come out as valid, groundable
        # ASP (an aggregate inlined into another aggregate's condition, a "_" from a
        # definition's aggregate) is kept as written.
        try:
            unfolded = []
            for c in unfold([clause], defs, policy).clauses:
                body = _drop_lonely_guards(
                    c.head, _restore_anonymous(list(c.body), set(ordered_vars([c.head]))))
                body = _negations_as_boxes(c.head, body, defs, policy)
                unfolded.append(f"{c.head} :- {', '.join(map(str, body))}." if body
                                else f"{c.head}.")
            _check_grounds("\n".join(unfolded), support)
            lines += unfolded
        except Exception:
            lines.append(f"{clause}")
    return "\n".join(lines)


def _drop_lonely_guards(head, body: list) -> list:
    """``body`` without ``has_atom(M, X)`` literals whose ``X`` is used nowhere else
    (an inlined definition's "the molecule has an atom" guard, an empty atom block in
    the editor), as long as another positive literal still binds ``M``."""
    import re

    from chebILP.explainability.rule_ir import Lit

    mol = head.args[0] if head.args else None
    text = " ".join(map(str, [head] + body))

    def lonely(b) -> bool:
        return (isinstance(b, Lit) and not b.neg and b.pred == "has_atom" and len(b.args) == 2
                and b.args[0] == mol and b.args[1][:1].isupper()
                and len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(b.args[1])}(?![A-Za-z0-9_])",
                                   text)) == 1)

    kept = [b for b in body if not lonely(b)]
    binds_mol = any(isinstance(b, Lit) and not b.neg and mol in b.args for b in kept)
    if not binds_mol:  # keep exactly one guard so the clause stays safe
        guard = next((b for b in body if lonely(b)), None)
        kept = ([guard] if guard is not None else []) + kept
    return kept


def _negations_as_boxes(head, body: list, defs: dict, policy) -> list:
    """``body`` with each negated auxiliary literal whose definition the block editor can
    draw replaced by "not" boxes: ``#count{ X,… : <definition body> } = 0`` ("no atoms X,…
    with …"), one per clause of the definition (``not (a ; b)`` is ``not a, not b``).
    A definition with no atoms of its own and a single molecule property becomes
    ``not p(M)``. Kept folded: definitions with aggregates or nested negated groups, ones
    that constrain the clause's own atoms (``c(X)`` on an outer ``X``), recursive ones."""
    from chebILP.explainability.rule_ir import Agg, Lit, ordered_vars

    mol = head.args[0] if head.args else None
    bound = set(ordered_vars([head] + [b for b in body
                                       if not (isinstance(b, Lit) and b.neg)
                                       and not isinstance(b, Agg)]))
    taken = tuple(ordered_vars([head] + body))
    out = []
    for b in body:
        boxes = None
        if isinstance(b, Lit) and b.neg and b.pred in defs and mol:
            boxes = _neg_boxes(b, mol, bound, taken, defs, policy)
        out.extend(boxes if boxes is not None else [b])
    return out


def _neg_boxes(lit, mol: str, bound: set, taken: tuple, defs: dict, policy) -> list | None:
    import itertools

    from chebILP.explainability.rule_ir import Clause, Cmp, Lit, ordered_vars, unfold

    anon = (f"_Anon{i}" for i in itertools.count(1))
    args = tuple(next(anon) if a == "_" else a for a in lit.args)
    call = Lit(lit.pred, args)
    # The helper head carries every variable of the clause, so unfolding renames the
    # definition's own variables apart from all of them.
    res = unfold([Clause(Lit("__neg", taken + args), [call])], defs, policy)
    if lit.pred in res.kept or "*" in res.kept:
        return None
    is_var = lambda t: t[:1].isupper() or t[:1] == "_"  # noqa: E731
    boxes = []
    for c in res.clauses:
        alt = list(dict.fromkeys(c.body))  # drop exact duplicates, keep order
        if not alt or any(isinstance(x, Lit) and x.pred == lit.pred for x in alt):
            return None
        own = [v for v in ordered_vars(alt) if v not in bound and v != mol]
        text = " ".join(map(str, alt))
        # "the molecule has an atom" guards over an atom used nowhere else
        alt = [x for x in alt if not (
            isinstance(x, Lit) and not x.neg and x.pred == "has_atom" and len(x.args) == 2
            and x.args[0] == mol and x.args[1] in own
            and len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(x.args[1])}(?![A-Za-z0-9_])",
                               text)) == 1)]
        own = [v for v in ordered_vars(alt) if v not in bound and v != mol]
        if not alt:
            return None
        for x in alt:
            if isinstance(x, Cmp):
                ok = (x.op in ("!=", "<") and is_var(x.left) and is_var(x.right)
                      and mol not in (x.left, x.right) and (x.left in own or x.right in own))
            elif isinstance(x, Lit) and len(x.args) == 1:
                a = x.args[0]
                ok = a in own or (a == mol and not x.neg)
            elif isinstance(x, Lit) and len(x.args) == 2 and x.pred == "has_atom":
                ok = not x.neg and x.args[0] == mol and x.args[1] in own
            elif isinstance(x, Lit) and len(x.args) == 2:
                ok = (all(is_var(a) and a != mol for a in x.args)
                      and any(a in own for a in x.args))
            else:
                ok = False
            if not ok:
                return None
        if not own and len(alt) == 1:
            boxes.append(f"not {alt[0]}")
        else:
            terms = ",".join(own) if own else mol
            boxes.append(f"#count{{ {terms} : {', '.join(map(str, alt))} }} = 0")
    return boxes or None


def _check_grounds(clauses: str, support: str) -> None:
    """Raise unless ``clauses`` (with the definitions in ``support``) parse and ground."""
    import clingo

    ctl = clingo.Control(["--warn=none"], logger=lambda _code, _msg: None)
    ctl.add("base", [], support + "\n" + clauses)
    ctl.ground([("base", [])])


# ── explanations ────────────────────────────────────────────────────────────


def _explainer_inputs(rule: str) -> tuple[str, list[dict]]:
    """``rule_programs(rule)`` with each "not" box of the block editor
    (``#count{ … } = 0``) turned back into a negated helper predicate, which chebILP's
    explainer presents as a pattern the molecule must not contain (rather than as
    "exactly 0 …")."""
    from chebILP.explainability.rule_ir import Agg, Clause, ordered_vars, parse_program

    hypothesis, programs = rule_programs(rule)
    if "#count" not in hypothesis:
        return hypothesis, programs
    try:
        clauses = parse_program(hypothesis)
    except Exception:
        return hypothesis, programs
    helpers, out = [], []
    for c in clauses:
        outer = set(ordered_vars([c.head] + [b for b in c.body if not isinstance(b, Agg)]))
        body = []
        for b in c.body:
            helper = _no_match_helper(b, outer, len(helpers) + 1) if isinstance(b, Agg) else None
            if helper is None:
                body.append(b)
                continue
            lit, program = helper
            helpers.append(program)
            body.append(lit)
        out.append(Clause(c.head, body))
    if not helpers:
        return hypothesis, programs
    return "\n".join(map(str, out)), programs + helpers


def _no_match_helper(agg, outer: set, k: int):
    """``(not aux_icarus_no_match_k(Shared…), program)`` for ``#count{ … : cond } = 0``,
    or None when ``agg`` is anything else or the helper would be unsafe."""
    from chebILP.explainability.rule_ir import Lit, ordered_vars

    zero = ("=", "0")
    if agg.func != "count" or len(agg.elems) != 1:
        return None
    if not ((agg.lguard == zero and agg.rguard is None) or (agg.rguard == zero and agg.lguard is None)):
        return None
    cond = list(agg.elems[0].cond)
    shared = [v for v in ordered_vars(cond) if v in outer]
    positive = set(ordered_vars([x for x in cond if isinstance(x, Lit) and not x.neg]))
    if any(v not in positive for v in shared):
        return None
    name = f"aux_icarus_no_match_{k}"
    head = f"{name}({','.join(shared)})" if shared else name
    program = {"name": name, "description": "a pattern the molecule must not contain",
               "source": f"{head} :- {', '.join(map(str, cond))}."}
    return Lit(name, tuple(shared), True), program


@functools.lru_cache(maxsize=64)
def explain_rule(rule: str) -> dict:
    """What ``rule`` asks of a molecule: ``{"text", "image" (PNG data URL)}``. Cached
    per rule text: drawing the figure takes a moment."""
    from chebILP.explainability.explain import explain_rule as _explain_rule

    efg.install()
    hypothesis, programs = _explainer_inputs(rule)
    expl = _explain_rule(hypothesis, aux_programs=programs)
    return {"text": expl.text, "image": _png_data_url(expl.image)}


def explain_molecule(mol, rule: str, name: str | None = None) -> dict:
    """Why ``rule`` does or does not accept the molecule (an RDKit mol or SMILES):
    ``{"fires", "text", "image"}`` with the match, or the closest match, highlighted."""
    from chebILP.explainability.explain import explain_molecule as _explain_molecule

    efg.install()
    hypothesis, programs = _explainer_inputs(rule)
    expl = _explain_molecule(mol, hypothesis, aux_programs=programs, name=name)
    return {"fires": expl.fires, "text": expl.text, "image": _png_data_url(expl.image)}
