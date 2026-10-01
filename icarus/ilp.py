"""Translate session molecules into a Popper ILP problem, learn a hypothesis,
and classify molecules against a (learned or hand-edited) rule.

Reuses chebILP's atom-level FOL translation, its Popper training subprocess, and
its clingo-based rule evaluation, so the demo stays consistent with the main
pipeline.
"""

import contextlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from chebILP.evaluation.clingo_eval import evaluate_with_clingo
from chebILP.ilp_classifier import run_ilp_training_subprocess
from chebILP.ilp_problem_builder import build_background_chemlog
from chebILP.utils import split_prolog_literals

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


def _library_query(concept) -> str | None:
    """Retrieval query for the concept (``name + definition``, as chebILP's generator
    builds it), or ``None`` when there is no definition to retrieve against."""
    concept = concept or {}
    definition = (concept.get("definition") or "").strip()
    if not definition:
        return None
    return f"{concept.get('name') or ''} {definition}".strip()


def retrieve_library_programs(concept) -> list:
    """The ``config.ILP_LIBRARY_TOP_N`` session-library rule programs most relevant to
    the concept's definition, best first (chebILP's ``HybridPredicateRetriever``).

    Empty when the concept has no definition or the library has no programs yet.
    """
    query = _library_query(concept)
    lib = config.LLM_LIBRARY_DIR
    if not query or config.ILP_LIBRARY_TOP_N <= 0 or not os.path.isdir(os.path.join(lib, "programs")):
        return []
    from chebILP.molecule_processing.fg_matching import is_fg_seed_name
    from chebILP.predicate_generation.auxiliary_rules import aux_rule_path, parse_rule_program
    from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever

    # BM25-only, like the LLM path: no sentence-transformers model download.
    retriever = HybridPredicateRetriever.from_rule_library(base_dir=lib, use_dense=False)
    programs = []
    for entry in retriever.retrieve(query, top_k=config.ILP_LIBRARY_TOP_N):
        # Seeded FG predicates are RDKit matches, not ASP rules; icarus can't ground them.
        if is_fg_seed_name(entry["name"]):
            continue
        path = aux_rule_path(entry["stem"], lib)
        try:
            with open(path, encoding="utf-8") as f:
                prog = parse_rule_program(f.read(), source_file=path)
        except OSError:
            continue
        if prog is not None:
            programs.append(prog)
    return programs


def library_background(bk_lines, mols, concept, on_line=None) -> tuple[list[str], list[tuple[str, int]]]:
    """Extra background facts + (predicate, arity) list from the session rule library.

    Retrieves the library predicates most relevant to the concept definition, grounds
    them (plus the library programs they build on) over the atom-level ``bk_lines`` and
    emits the derived ``aux_*`` facts — mirroring chebILP's ``llm_generated_rules``
    background, where dependencies are ground alongside but only the selected
    predicates become ILP features. Predicates that hold for no session molecule are
    dropped. Degrades to ``([], [])`` on any failure.
    """
    log = on_line or (lambda _line: None)
    try:
        programs = retrieve_library_programs(concept)
        if not programs:
            return [], []
        from chebILP.predicate_generation.auxiliary_rules import (
            derive_rule_extensions, predicate_arg_sorts, resolve_rule_dependencies,
        )

        deps = resolve_rule_dependencies(programs, config.LLM_LIBRARY_DIR)
        extensions = derive_rule_extensions(programs + deps, bk_lines, [m.id for m in mols])
    except Exception as e:
        log(f"Rule library: retrieval/grounding failed ({e}); using the atom-level background only.")
        return [], []

    lines, preds = [], []
    for prog in programs:
        arity = len(predicate_arg_sorts(prog))
        emitted = []
        for arg_tuples in extensions.get(prog.name, {}).values():
            for args in arg_tuples:
                line = f"{prog.name}({','.join(args)})."
                if line not in emitted:
                    emitted.append(line)
        if not arity or not emitted:
            continue
        lines += emitted
        preds.append((prog.name, arity))
    if preds:
        log(f"Rule library: added {len(preds)} retrieved predicate(s) to the background: "
            + ", ".join(f"{n}/{a}" for n, a in preds))
    else:
        log("Rule library: no retrieved predicate holds for any session molecule.")
    return lines, preds


def build_learning_background(mols, concept=None, on_line=None) -> tuple[list[str], list[tuple[str, int]]]:
    """The background an ILP learner sees: atom-level facts, augmented with the
    library predicates retrieved for the concept definition (if there is one)."""
    lines, body_predicates = build_background(mols)
    extra_lines, extra_preds = library_background(lines, mols, concept, on_line)
    return lines + extra_lines, list(body_predicates) + extra_preds


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


@contextlib.contextmanager
def run_dir(prefix: str):
    """A fresh working directory under ``config.WORK_DIR`` for one learning run,
    removed afterwards — concurrent runs (other tabs / users) must not overwrite
    each other's problem files."""
    path = tempfile.mkdtemp(prefix=prefix, dir=config.WORK_DIR)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_problem(work_dir, pos_mols, neg_mols, concept=None, on_line=None):
    """Write exs.pl, bk.pl and bias.pl for the labelled molecules. Returns paths.

    With a concept definition, the background is augmented with retrieved library
    predicates (see :func:`build_learning_background`)."""
    os.makedirs(work_dir, exist_ok=True)
    exs_path = os.path.join(work_dir, "exs.pl")
    bk_path = os.path.join(work_dir, "bk.pl")
    bias_path = os.path.join(work_dir, "bias.pl")

    all_mols = pos_mols + neg_mols
    bk_lines, body_predicates = build_learning_background(all_mols, concept, on_line)

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
    with run_dir("popper_") as work_dir:
        exs_path, bk_path, bias_path = write_problem(work_dir, pos_mols, neg_mols)
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


def learn_streaming(pos_mols, neg_mols, timeout, on_line, concept=None) -> dict:
    """Run noisy (MDL) Popper, streaming its stdout line-by-line via ``on_line``.

    Returns ``{"rule": str|None, "score": {...}|None}`` (or an ``error``). The
    child runs unbuffered (``python -u``) so Popper's progress surfaces live.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    with run_dir("popper_") as work_dir:
        exs_path, bk_path, bias_path = write_problem(work_dir, pos_mols, neg_mols,
                                                     concept=concept, on_line=on_line)
        # Noisy: Popper minimises an MDL cost, returning the best rule found even when none
        # perfectly separates the examples (rather than failing to return one).
        settings = {"timeout": timeout, "noisy": True}
        script = _training_script(exs_path, bk_path, bias_path, settings)

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True, cwd=work_dir,
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
    / editable definition) feeds the LLM pipeline's prompt and, for Popper / Aleph,
    the retrieval of library predicates that augment the background.
    """
    try:
        return _learn_dispatch(method, pos_mols, neg_mols, timeout, on_line, model, concept)
    finally:
        _NL_CACHE.clear()  # the run may have changed library aux descriptions


def _learn_dispatch(method, pos_mols, neg_mols, timeout, on_line, model, concept) -> dict:
    if method in ("popper", "aleph"):
        if method == "popper":
            result = learn_streaming(pos_mols, neg_mols, timeout, on_line, concept=concept)
        else:
            from . import aleph

            result = aleph.learn_streaming(pos_mols, neg_mols, timeout, on_line, concept=concept)
        # A rule over retrieved library predicates carries only their literals; pull in
        # their definitions so it is self-contained for classify / NL / explain.
        if result.get("rule"):
            result["rule"] = expand_aux_definitions(result["rule"])
        return result
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


# NL per rule text. The state is re-sent after every UI action (relabel, remove, …),
# and re-translating an unchanged rule each time slowed those round-trips. Cleared
# by ``learn_dispatch`` since a learn run may add/rewrite library aux descriptions.
_NL_CACHE: dict[str, str | None] = {}


def _translate_one(rule_text: str) -> str | None:
    """NL for a *single-head* rule via chebILP's ``translate_rule``, annotating any
    ``aux_`` references with their descriptions from the session predicate library."""
    if not rule_text:
        return None
    if rule_text not in _NL_CACHE:
        _NL_CACHE[rule_text] = _translate_uncached(rule_text)
    return _NL_CACHE[rule_text]


def _translate_uncached(rule_text: str) -> str | None:
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


_ANON_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])_(?![A-Za-z0-9_])")


def _name_anonymous_vars(clause: str) -> str:
    """Replace each anonymous ``_`` in a positive body literal with a fresh named variable.

    xclingo copies positive body literals into the heads of its generated
    support rules, where an anonymous variable becomes unsafe and grounding
    fails. A variable that occurs only once has the same meaning as ``_``.
    Negated literals keep their ``_``: there a named variable would be unsafe.
    """
    if ":-" not in clause:
        return clause
    head, body = clause.split(":-", 1)
    counter = iter(range(1, 1_000_000))
    lits = [lit if lit.startswith("not ")
            else _ANON_VAR_RE.sub(lambda _m: f"AnonV{next(counter)}", lit)
            for lit in split_prolog_literals(body.strip().rstrip("."))]
    return f"{head.strip()} :- {', '.join(lits)}."


def _xclingo_ready_rule(smiles: str, rule: str, mol_id: str) -> str:
    """Rewrite ``rule`` into a form chebILP's xclingo explainer can handle.

    Anonymous variables in positive body literals get names, and every clause
    with an aggregate (``#count`` etc.) is replaced by the ground facts it
    derives for this molecule: xclingo finds no explanation at all through an
    aggregate, so such a predicate is cited as a given property instead (for a
    target clause, via a helper predicate the target is bridged to). Target
    clauses come last. Returns one clause per line, as chebILP parses the rule
    line by line.
    """
    import clingo
    from rdkit import Chem

    clauses = _split_clauses(rule)
    agg = [c for c in clauses if ":-" in c and "#" in c.split(":-", 1)[1]]
    # Aggregate clauses are ground by plain clingo below, so they keep their ``_``.
    clauses = [c if c in agg else _name_anonymous_vars(c) for c in clauses]
    # chebILP explains the head of the last clause, so the target clauses go last.
    clauses.sort(key=lambda c: _clause_head(c) == TARGET)
    mol = Chem.MolFromSmiles(smiles)
    if not agg or mol is None:  # invalid SMILES: let chebILP raise its error
        return "\n".join(clauses)

    # A target clause is not replaced by facts itself (chebILP would then explain some
    # other predicate): its aggregate body becomes a helper, the target a bridge to it.
    helper = f"aux_{TARGET}_condition"
    target_agg = [c for c in agg if _clause_head(c) == TARGET]
    agg_facts_src = [re.sub(rf"^\s*{TARGET}\s*\(", f"{helper}(", c, count=1) if c in target_agg else c
                     for c in agg]

    # Same background chebILP's explainer builds, so atom ids line up.
    mol_df = pd.DataFrame([{"mol": mol}], index=[mol_id])
    bk_lines, _ = build_background_chemlog(mol_df)
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], "\n".join(bk_lines + [c for c in clauses if c not in agg] + agg_facts_src))
    ctl.ground([("base", [])])
    agg_heads = {_clause_head(c) for c in agg_facts_src}
    facts: list[str] = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            facts = sorted(f"{s}." for s in model.symbols(atoms=True) if s.name in agg_heads)
            break
    bridge = [f"{TARGET}(M) :- {helper}(M)."] if target_agg else []
    return "\n".join(facts + [c for c in clauses if c not in agg] + bridge)


def explain_molecule(smiles: str, rule: str) -> dict:
    """Graphical + textual explanation of why the rule classifies a molecule positive.

    Returns ``{satisfies, text, image}`` where ``image`` is a base64 PNG data URL
    with the atoms cited in the explanation highlighted (chebILP + xclingo).
    """
    import base64
    import io

    from chebILP.explainability.explain import explain_molecule as _explain

    satisfies, text, img = _explain(smiles, _xclingo_ready_rule(smiles, rule, "mol1"), mol_id="mol1")

    # The generic target predicate is "concept"; rewrite chebILP's CHEBI-oriented
    # phrasing into concept-neutral text.
    if satisfies:
        marker = "because it satisfies the following conditions:"
        conds = text.split(marker, 1)[1].strip() if marker in text else ""
        text = ("This molecule matches the learned concept because it satisfies:\n" + conds
                if conds else "This molecule matches the learned concept.")
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
