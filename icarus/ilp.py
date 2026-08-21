"""Translate session molecules into a Popper ILP problem, learn a hypothesis,
and classify molecules against a (learned or hand-edited) rule.

Reuses chebILP's atom-level FOL translation, its Popper training subprocess, and
its clingo-based rule evaluation, so the demo stays consistent with the main
pipeline.
"""

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

    bias_lines = [
        f"max_vars({config.DEFAULT_MAX_VARS}).",
        f"max_body({config.DEFAULT_MAX_BODY}).",
        f"max_clauses({config.DEFAULT_MAX_CLAUSES}).",
        "",
        f"head_pred({TARGET},1).",
    ]
    for pred, arity in _bias_body_predicates(body_predicates):
        bias_lines.append(f"body_pred({pred},{arity}).")
    with open(bias_path, "w") as f:
        f.write("\n".join(bias_lines) + "\n")

    return exs_path, bk_path, bias_path


def learn(pos_mols, neg_mols, timeout=None) -> dict:
    """Run non-noisy Popper on the labelled molecules.

    Returns ``{"rule": str|None, "score": {...}|None}``.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None, "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    exs_path, bk_path, bias_path = write_problem(config.WORK_DIR, pos_mols, neg_mols)

    settings = {"timeout": timeout}  # non-noisy (Popper default)
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
    """Run non-noisy Popper, streaming its stdout line-by-line via ``on_line``.

    Returns ``{"rule": str|None, "score": {...}|None}`` (or an ``error``). The
    child runs unbuffered (``python -u``) so Popper's progress surfaces live.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    exs_path, bk_path, bias_path = write_problem(config.WORK_DIR, pos_mols, neg_mols)
    settings = {"timeout": timeout}  # non-noisy (Popper default)
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


def rule_to_nl(rule: str) -> str | None:
    """Natural-language paraphrase of a learned/edited rule (chebILP)."""
    if not rule:
        return None
    try:
        from chebILP.explainability.rule_to_nl import translate_rule

        return translate_rule(rule)
    except Exception:
        return None


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
