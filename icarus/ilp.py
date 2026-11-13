"""Translate session molecules into a Popper ILP problem, learn a hypothesis,
and classify molecules against a (learned or hand-edited) rule.

Reuses chebILP's atom-level FOL translation, its Popper training subprocess, and
its clingo-based rule evaluation, so the demo stays consistent with the main
pipeline.
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from chebILP.evaluation.clingo_eval import evaluate_with_clingo
from chebILP.preprocessing.chebi.mol_to_fol import build_background_chemlog

from . import config

TARGET = config.TARGET_LABEL


def _rows(mols) -> pd.DataFrame:
    """DataFrame with index = molecule id and a ``mol`` column, as
    ``build_background_chemlog`` expects (it iterates ``row.Index`` / ``row.mol``)."""
    return pd.DataFrame({"mol": [m.mol for m in mols]}, index=[m.id for m in mols])


def build_background(mols) -> tuple[list[str], list[tuple[str, int]]]:
    """Background facts + (predicate, arity) list for the given molecules: the
    atom-level facts plus chebILP's computed facts (see :func:`computed_facts`)."""
    if not mols:
        return [], []
    rows = _rows(mols)
    lines, body_predicates = build_background_chemlog(rows)
    # The computed facts are numeric (mol_weight(M,146)), so they are not offered to
    # Popper/Aleph as body predicates; they're there for library rules built on them.
    return lines + computed_facts(rows), body_predicates



def computed_facts(rows) -> list[str]:
    """``mol_weight(M,W)`` / ``ring_size(M,S)`` facts for ``rows`` (chebILP's
    ``build_computed_facts``). Many library predicates were generated with these facts
    available; without them such a predicate never fires (or, negated, always does)."""
    from chebILP.preprocessing.chebi.mol_to_fol import build_computed_facts

    rows = rows[rows["mol"].notna()]
    return build_computed_facts(rows) if len(rows) else []


def _library_query(concept) -> str | None:
    """Retrieval query for the concept (``name + definition``, as chebILP's generator
    builds it), or ``None`` when there is no definition to retrieve against."""
    concept = concept or {}
    definition = (concept.get("definition") or "").strip()
    if not definition:
        return None
    return f"{concept.get('name') or ''} {definition}".strip()


def library_options(concept) -> tuple[bool, bool]:
    """``(use_aux, use_efg)``: which library predicate kinds a learn may use (user
    settings, carried on the concept dict; both on by default)."""
    concept = concept or {}
    return concept.get("use_aux", True) is not False, concept.get("use_efg", True) is not False


_LIB_ENTRIES: tuple | None = None  # (programs dir mtime, retriever entries)


def _library_entries() -> list[dict]:
    """Retriever entries for the (read-only) library's ``aux_*`` programs, cached until
    the programs directory changes — parsing ~3k files costs seconds per learn."""
    global _LIB_ENTRIES
    progs_dir = os.path.join(config.LLM_LIBRARY_DIR, "programs")
    if not os.path.isdir(progs_dir):
        return []
    mtime = os.stat(progs_dir).st_mtime
    if _LIB_ENTRIES is None or _LIB_ENTRIES[0] != mtime:
        from chebILP.preprocessing.chebi.fg_matching import is_fg_seed_name
        from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever

        entries = HybridPredicateRetriever.from_rule_library(
            base_dir=config.LLM_LIBRARY_DIR, use_dense=False).entries
        # A seeded FG file in the library would duplicate the EFG entries added below.
        _LIB_ENTRIES = (mtime, [e for e in entries if not is_fg_seed_name(e["stem"])])
    return _LIB_ENTRIES[1]


def retrieve_library_predicates(concept) -> tuple[list, list[str]]:
    """The ``config.ILP_LIBRARY_TOP_N`` library predicates most relevant to the concept's
    definition: ``(aux rule programs, efg_* group names)``.

    Candidates are the library's generated ``aux_*`` programs and the Extended
    Functional Groups (see :mod:`icarus.efg`); chebILP's ``HybridPredicateRetriever``
    gives each kind half of the spots (a kind with too few candidates, or switched off
    in the settings, leaves its spots to the other). Empty when the concept has no
    definition.
    """
    query = _library_query(concept)
    use_aux, use_efg = library_options(concept)
    if not query or config.ILP_LIBRARY_TOP_N <= 0 or not (use_aux or use_efg):
        return [], []
    from chebILP.predicate_generation.auxiliary_rules import aux_rule_path, parse_rule_program
    from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever

    from . import efg

    entries = (list(_library_entries()) if use_aux else []) + \
              (efg.retrieval_entries() if use_efg else [])
    if not entries:
        return [], []
    # BM25-only, like the LLM path: no sentence-transformers model download.
    retriever = HybridPredicateRetriever(entries, use_dense=False)
    programs, groups = [], []
    for entry in retriever.retrieve(query, top_k=config.ILP_LIBRARY_TOP_N):
        if efg.is_efg(entry["name"]):
            groups.append(entry["name"])
            continue
        path = aux_rule_path(entry["stem"], config.LLM_LIBRARY_DIR)
        try:
            with open(path, encoding="utf-8") as f:
                prog = parse_rule_program(f.read(), source_file=path)
        except OSError:
            continue
        if prog is not None:
            programs.append(prog)
    return programs, groups


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
    use_aux, use_efg = library_options(concept)
    if not (use_aux or use_efg):
        log("Library predicates are switched off; using the atom-level background only.")
        return [], []
    try:
        programs, groups = retrieve_library_predicates(concept)
        if not programs and not groups:
            return [], []
        from chebILP.predicate_generation.auxiliary_rules import (
            derive_rule_extensions, predicate_arg_sorts, resolve_rule_dependencies,
        )

        from . import efg

        extensions = {}
        if programs:
            deps = resolve_rule_dependencies(programs, config.LLM_LIBRARY_DIR)
            extensions = derive_rule_extensions(programs + deps, bk_lines, [m.id for m in mols])
        efg_lines = efg.fact_lines(mols, groups)
    except Exception as e:
        log(f"Rule library: retrieval/grounding failed ({e}); using the atom-level background only.")
        return [], []

    lines, preds = [], []
    # Functional groups first: molecule-level presence facts, efg_<name>(M).
    for name in groups:
        emitted = [l for l in efg_lines if l.startswith(name + "(")]
        if emitted:
            lines += emitted
            preds.append((name, 1))
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


def score_dict(score) -> dict | None:
    """An ILP learner's ``(tp, fn, tn, fp)`` score tuple as a confusion dict."""
    if not score:
        return None
    tp, fn, tn, fp = score
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp}


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
                    result["score"] = score_dict(data.get("score"))
                except Exception:
                    pass
                continue
            if line.strip():
                on_line(line)
        proc.wait()
    return result


def learn_dispatch(method, pos_mols, neg_mols, timeout, on_line, model=None, concept=None,
                   admit=None, use_cache=True) -> dict:
    """Route a learn request to one of the three rule-generation backends.

    Every backend returns the same shape — ``{"rule": str|None, "score":
    {TP,FN,TN,FP}|None, "error"?: str}`` — with a rule head of ``TARGET``, so the
    caller (and everything downstream: classify, NL, explain) is method-agnostic.
    ``on_line`` receives progress lines for the live log. ``concept`` (name / chebi_id
    / editable definition) feeds the LLM pipeline's prompt and, for Popper / Aleph,
    the retrieval of library predicates that augment the background. ``admit`` is the
    LLM quota check (``llm_guard.admit``); the ILP backends ignore it, as they do
    ``use_cache`` (``False`` = prompt the LLM even if the library has a cached rule).
    """
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
            model=model or config.LLM_MODEL, concept=concept, admit=admit,
            use_cache=use_cache,
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
    from . import efg

    bk_lines, _ = build_background(mols)
    # Functional groups are RDKit matches, not clauses: supply the facts the rule uses.
    bk_lines += efg.rule_facts(rule, mols)
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
    lines and collapsing each clause to a single line."""
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


_HEADER_RE = re.compile(r"^%\s*(PREDICATE_NAME|DESCRIPTION)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def rule_aux_descriptions(rule: str) -> dict[str, str]:
    """``aux_name -> description`` from the ``% PREDICATE_NAME`` / ``% DESCRIPTION``
    headers a rule carries for the auxiliary programs it defines (an LLM rule's newly
    generated predicates are not in the read-only library, only in the rule)."""
    out, name = {}, None
    for line in (rule or "").splitlines():
        m = _HEADER_RE.match(line.strip())
        if not m:
            continue
        if m.group(1).upper() == "PREDICATE_NAME":
            name = m.group(2)
        elif name:
            out[name] = m.group(2)
            # A header may name the predicate without its aux_ prefix (chebILP writes
            # the model's name as given); the clauses always carry the prefix.
            if not name.startswith(("aux_", "efg_")):
                out.setdefault("aux_" + name, m.group(2))
    return out


def _aux_header(name: str, descriptions: dict[str, str]) -> str:
    desc = descriptions.get(name)
    return f"% PREDICATE_NAME: {name}\n% DESCRIPTION: {desc}\n" if desc else ""


def library_descriptions() -> dict[str, str]:
    """``aux_name -> description`` for the (read-only) library's predicates."""
    return {e["name"]: e.get("description") or "" for e in _library_entries()}


def rule_parts(rule: str) -> dict | None:
    """Decompose a rule into its main hypothesis and the auxiliary predicates it defines.

    Returns ``{"main": {head, rule}, "aux": [{name, description, rule}, ...]}`` (aux
    empty for a plain Popper/Aleph rule), or ``None``. This is the rule's own structure,
    as the logic-program view shows it; the explanation (:mod:`icarus.explain`) shows it
    with the auxiliary predicates unfolded.
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
    descriptions = rule_aux_descriptions(rule)
    return {
        "main": {"head": main_head, "rule": "\n".join(groups[main_head])},
        # Each aux box keeps the description header, so a rule the UI reassembles from
        # the boxes still carries it.
        "aux": [{"name": name, "description": descriptions.get(name, ""),
                 "rule": _aux_header(name, descriptions) + "\n".join(groups[name])}
                for name in order if name != main_head],
    }


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
