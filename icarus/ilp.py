"""Translate session molecules into a Popper ILP problem, learn a hypothesis,
and classify molecules against a (learned or hand-edited) rule.

Reuses chebILP's atom-level FOL translation, its Popper training subprocess, and
its clingo-based rule evaluation, so the demo stays consistent with the main
pipeline.
"""

import inspect
import json
import os
import re
import subprocess
import sys

import pandas as pd

from chebILP.evaluation.clingo_eval import evaluate_with_clingo
from chebILP.ilp_classifier import run_ilp_training_subprocess
from chebILP.ilp_problem_builder import build_background_chemlog

from . import config

TARGET = config.TARGET_LABEL


def _rows(mols) -> pd.DataFrame:
    """DataFrame with index = molecule id and a ``mol`` column, as
    ``build_background_chemlog`` expects (it iterates ``row.Index`` / ``row.mol``)."""
    return pd.DataFrame({"mol": [m.mol for m in mols]}, index=[m.id for m in mols])


def build_background(mols) -> tuple[list[str], list[tuple[str, int]]]:
    """Atom-level background facts + (predicate, arity) list for the given molecules."""
    if not mols:
        return [], []
    rows = _rows(mols)
    lines, body_predicates = build_background_chemlog(rows, predicate_set="atoms")
    return lines, body_predicates


def _bias_body_predicates(body_predicates) -> list[tuple[str, int]]:
    """Keep learnable, low-arity predicates. Drops the multi-argument ring{N}
    predicates (arity up to 8) that blow up the search, keeping ring *membership*
    (in_ring / in_ringN) instead."""
    keep = []
    for pred, arity in body_predicates:
        if arity > 2:
            continue
        if pred == TARGET:
            continue
        keep.append((pred, arity))
    return keep


def _bias_lines(body_predicates) -> list[str]:
    """The bias.pl lines (search limits + head/body predicate declarations) shared
    by the Popper and Aleph paths — Aleph reads the ``body_pred`` lines back to
    learn which predicates (and arities) are in play."""
    lines = [
        f"max_vars({config.DEFAULT_MAX_VARS}).",
        f"max_body({config.DEFAULT_MAX_BODY}).",
        f"max_clauses({config.DEFAULT_MAX_CLAUSES}).",
        "",
        f"head_pred({TARGET},1).",
    ]
    for pred, arity in _bias_body_predicates(body_predicates):
        lines.append(f"body_pred({pred},{arity}).")
    return lines


def write_problem(work_dir, pos_mols, neg_mols):
    """Write exs.pl, bk.pl and bias.pl for the labelled molecules. Returns paths."""
    os.makedirs(work_dir, exist_ok=True)
    exs_path = os.path.join(work_dir, "exs.pl")
    bk_path = os.path.join(work_dir, "bk.pl")
    bias_path = os.path.join(work_dir, "bias.pl")

    all_mols = pos_mols + neg_mols
    bk_lines, body_predicates = build_background(all_mols)

    with open(exs_path, "w") as f:
        for m in pos_mols:
            f.write(f"pos({TARGET}({m.id})).\n")
        for m in neg_mols:
            f.write(f"neg({TARGET}({m.id})).\n")

    with open(bk_path, "w") as f:
        f.write("\n".join(bk_lines) + "\n")

    with open(bias_path, "w") as f:
        f.write("\n".join(_bias_lines(body_predicates)) + "\n")

    return exs_path, bk_path, bias_path


def learn(pos_mols, neg_mols, timeout=None) -> dict:
    """Run noisy (MDL) Popper on the labelled molecules.

    Returns ``{"rule": str|None, "score": {...}|None}``.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None, "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    exs_path, bk_path, bias_path = write_problem(config.WORK_DIR, pos_mols, neg_mols)

    settings = {"timeout": timeout, "noisy": True}  # MDL cost fn: best rule, need not separate perfectly
    result = run_ilp_training_subprocess(exs_path, bk_path, bias_path, settings)
    rule = result.get("prog_str")
    score = result.get("score")
    score_dict = None
    if score:
        tp, fn, tn, fp = score
        score_dict = {"TP": tp, "FN": fn, "TN": tn, "FP": fp}
    return {"rule": rule, "score": score_dict}


_RESULT_MARKER = "__ICARUS_RESULT__"


def _training_script(exs_path, bk_path, bias_path, settings) -> str:
    """Inline script run in a child interpreter: runs Popper and prints its own
    progress to stdout, then a marker line carrying the JSON result."""
    return f'''
import json
from popper.loop import popper
from popper.util import Settings, format_prog
settings = Settings(ex_file=r"{exs_path}", bk_file=r"{bk_path}", bias_file=r"{bias_path}", **{settings!r})
prog, score = popper(settings)
prog_str = format_prog(prog) if prog else None
print("{_RESULT_MARKER}" + json.dumps({{"prog_str": prog_str, "score": list(score) if score else None}}), flush=True)
'''


def learn_streaming(pos_mols, neg_mols, timeout, on_line) -> dict:
    """Run noisy (MDL) Popper, streaming its stdout line-by-line via ``on_line``.

    Returns ``{"rule": str|None, "score": {...}|None}`` (or an ``error``). The
    child runs unbuffered (``python -u``) so Popper's progress surfaces live.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    exs_path, bk_path, bias_path = write_problem(config.WORK_DIR, pos_mols, neg_mols)
    # Noisy: Popper minimises an MDL cost, returning the best rule found even when none
    # perfectly separates the examples (rather than failing to return one).
    settings = {"timeout": timeout, "noisy": True}
    script = _training_script(exs_path, bk_path, bias_path, settings)

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True, cwd=config.WORK_DIR,
    )
    result = {"rule": None, "score": None}
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line.startswith(_RESULT_MARKER):
            try:
                data = json.loads(line[len(_RESULT_MARKER):])
                result["rule"] = data.get("prog_str")
                sc = data.get("score")
                if sc:
                    tp, fn, tn, fp = sc
                    result["score"] = {"TP": tp, "FN": fn, "TN": tn, "FP": fp}
            except Exception:
                pass
            continue
        if line.strip():
            on_line(line)
    proc.wait()
    return result


def learn_dispatch(method, pos_mols, neg_mols, timeout, on_line, model=None, concept=None) -> dict:
    """Route a learn request to one of the three rule-generation backends.

    Every backend returns the same shape — ``{"rule": str|None, "score":
    {TP,FN,TN,FP}|None, "error"?: str}`` — with a rule head of ``TARGET``, so the
    caller (and everything downstream: classify, NL, explain) is method-agnostic.
    ``on_line`` receives progress lines for the live log. ``concept`` (name / chebi_id
    / editable definition) is used only by the LLM pipeline's prompt.
    """
    if method == "popper":
        return learn_streaming(pos_mols, neg_mols, timeout, on_line)
    if method == "aleph":
        from . import aleph

        return aleph.learn_streaming(pos_mols, neg_mols, timeout, on_line)
    if method == "llm":
        from . import llm_rulegen

        return llm_rulegen.learn_streaming(
            pos_mols, neg_mols, timeout, on_line,
            model=model or config.LLM_MODEL, concept=concept,
        )
    return {"rule": None, "score": None, "error": f"Unknown learning method: {method!r}"}


def expand_aux_definitions(rule: str) -> str:
    """Prepend the library definitions of any ``aux_*`` predicates a rule references
    but does not itself define, so the rule is self-contained for clingo.

    A block-editor rule that uses a generated (``aux_*``) predicate carries only the
    literal, not the predicate's ASP definition; without the definition clingo
    derives nothing and every molecule classifies negative. Reuses chebILP's
    transitive, cycle-safe dependency resolver over the session rule library, and
    skips predicates the rule already defines (an LLM-learned rule, or an aux box the
    UI reassembled), so nothing is duplicated. Degrades to the original rule when the
    library is absent or resolution fails.
    """
    if not rule or "aux_" not in rule:
        return rule
    lib = config.LLM_LIBRARY_DIR
    if not os.path.isdir(lib):
        return rule
    try:
        from chebILP.predicate_generation.auxiliary_rules import (
            RuleProgram, resolve_rule_dependencies,
        )
        deps = resolve_rule_dependencies([RuleProgram(name="__rule__", source=rule)], lib)
    except Exception:
        return rule
    if not deps:
        return rule
    return "\n\n".join(d.source.strip() for d in deps) + "\n\n" + rule


def classify(rule: str, mols) -> dict[str, bool]:
    """Which molecules the rule marks positive. Returns ``{mol_id: bool}``.

    Grounds the rule against freshly-built background facts for ``mols`` using
    clingo (fast; no Prolog), so it also works for hand-edited rules.
    """
    if not mols or not rule:
        return {m.id: False for m in mols}
    bk_lines, _ = build_background(mols)
    ids = [m.id for m in mols]
    try:
        positives = evaluate_with_clingo(rule.split("\n"), bk_lines, [TARGET], ids)
    except Exception as e:  # malformed hand-edited rule, etc.
        raise RuleError(str(e))
    hits = set(positives.get(TARGET, []))
    return {mid: (mid in hits) for mid in ids}


_HEAD_RE = re.compile(r"^([a-z_][A-Za-z0-9_]*)\s*\(")


def _split_clauses(rule: str) -> list[str]:
    """Split a program into individual clauses on the ``.`` terminators at bracket
    depth 0 (so ``.`` inside aggregates/ranges never splits), dropping ``%`` comment
    lines and collapsing each clause to a single line — chebILP's translator parses
    one clause per line."""
    text = "\n".join(l for l in rule.splitlines() if not l.strip().startswith("%"))
    depth, cur, clauses = 0, [], []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "." and depth == 0:
            c = " ".join("".join(cur).split()).strip()
            if c:
                clauses.append(c + ".")
            cur = []
        else:
            cur.append(ch)
    tail = " ".join("".join(cur).split()).strip()
    if tail:
        clauses.append(tail if tail.endswith(".") else tail + ".")
    return clauses


def _clause_head(clause: str) -> str | None:
    m = _HEAD_RE.match(clause.split(":-", 1)[0].strip())
    return m.group(1) if m else None


def _translate_one(rule_text: str) -> str | None:
    """NL for a *single-head* rule via chebILP's ``translate_rule``, annotating any
    ``aux_`` references with their descriptions from the session predicate library."""
    if not rule_text:
        return None
    try:
        from chebILP.explainability.rule_to_nl import translate_rule
    except Exception:
        return None
    kwargs = {}
    lib = config.LLM_LIBRARY_DIR
    if os.path.isdir(lib) and "aux_library_dir" in inspect.signature(translate_rule).parameters:
        kwargs["aux_library_dir"] = lib
    try:
        return translate_rule(rule_text, **kwargs)
    except Exception:
        return None


def rule_to_nl_parts(rule: str) -> dict | None:
    """Decompose a rule into its main hypothesis and auxiliary predicates, each with
    its own NL translation.

    chebILP's ``translate_rule`` handles one head predicate at a time (it rejects a
    program mixing ``concept`` with ``aux_*`` definitions), so an LLM-generated rule —
    the concept hypothesis plus the auxiliary predicates it builds on — must be split
    by head and translated piecewise. Returns ``{"main": {head, rule, nl}, "aux":
    [{name, rule, nl}, ...]}`` (aux empty for a plain Popper/Aleph rule), or ``None``.
    """
    if not rule:
        return None
    order, groups = [], {}
    for clause in _split_clauses(rule):
        name = _clause_head(clause)
        if name is None:
            continue
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(clause)
    if not order:
        return None

    # The main hypothesis is the target predicate; everything else is auxiliary.
    main_head = TARGET if TARGET in groups else order[-1]
    main_text = "\n".join(groups[main_head])
    parts = {
        "main": {"head": main_head, "rule": main_text, "nl": _translate_one(main_text)},
        "aux": [],
    }
    for name in order:
        if name == main_head:
            continue
        text = "\n".join(groups[name])
        parts["aux"].append({"name": name, "rule": text, "nl": _translate_one(text)})
    return parts


def rule_to_nl(rule: str) -> str | None:
    """Natural-language paraphrase of a single-head learned/edited rule (chebILP)."""
    return _translate_one(rule)


def explain_molecule(smiles: str, rule: str) -> dict:
    """Graphical + textual explanation of why the rule classifies a molecule positive.

    Returns ``{satisfies, text, image}`` where ``image`` is a base64 PNG data URL
    with the atoms cited in the explanation highlighted (chebILP + xclingo).
    """
    import base64
    import io

    from chebILP.explainability.explain import explain_molecule as _explain

    satisfies, text, img = _explain(smiles, rule, mol_id="mol1")

    # The generic target predicate is "concept"; rewrite chebILP's CHEBI-oriented
    # phrasing into concept-neutral text.
    if satisfies:
        marker = "because it satisfies the following conditions:"
        conds = text.split(marker, 1)[1].strip() if marker in text else ""
        text = "This molecule matches the learned concept because it satisfies:\n" + conds
    else:
        text = "This molecule does not match the learned concept."

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"satisfies": satisfies, "text": text, "image": "data:image/png;base64," + b64}


class RuleError(Exception):
    pass


def rule_targets_ok(rule: str) -> bool:
    """Sanity check that a hand-edited rule defines the target predicate."""
    for line in rule.split("\n"):
        if ":-" in line:
            head = line.split(":-")[0].strip()
            if re.match(rf"{TARGET}\s*\(", head):
                return True
    return False
