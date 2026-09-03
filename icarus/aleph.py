"""Learn a rule for the session concept with Aleph, as a Popper alternative.

Reuses chebILP's Aleph infrastructure end to end:
  * ``build_aleph_background`` assembles the structural ``.b`` (modes +
    determinations + background) from the same atom-level facts and (bias-filtered)
    body predicates the Popper path uses;
  * ``run_ilp_training_aleph`` injects the search settings and drives the vendored
    engine under ``swipl``, returning the same result shape as Popper's subprocess.

chebILP's Aleph runner is written around ChEBI classes: it names the head
``chebi_<id>`` and parses that back out of the engine's output. We therefore learn
under a numeric dummy id and rewrite the head to the generic ``concept`` target so
the rest of the demo (classify / NL / explain) stays method-agnostic.
"""

import os
import re

from chebILP.aleph_runner import run_ilp_training_aleph
from chebILP.ilp_problem_builder import build_aleph_background

from . import config, ilp

TARGET = config.TARGET_LABEL

# Numeric dummy id: chebILP's Aleph runner builds the head as ``chebi_<id>`` and
# its output parser matches ``chebi_\d+``, so the id must be digits. The learned
# head is rewritten to TARGET afterwards.
_HEAD_ID = "0"
_HEAD_NAME = f"chebi_{_HEAD_ID}"


def _write_problem(work_dir, pos_mols, neg_mols):
    """Write the Aleph ``.b`` / ``.f`` / ``.n`` stem files and a bias file.

    Returns ``(aleph_stem, bias_path)``. The background and the (bias-filtered,
    low-arity) body predicates are exactly those the Popper path learns over, so
    the two backends see the same problem."""
    os.makedirs(work_dir, exist_ok=True)
    bk_lines, body_predicates = ilp.build_background(pos_mols + neg_mols)
    learnable = ilp._bias_body_predicates(body_predicates)

    stem = os.path.join(work_dir, "aleph_problem")
    with open(stem + ".b", "w") as f:
        f.write(build_aleph_background(_HEAD_NAME, learnable, bk_lines))
    with open(stem + ".f", "w") as f:
        for m in pos_mols:
            f.write(f"{_HEAD_NAME}({m.id}).\n")
    with open(stem + ".n", "w") as f:
        for m in neg_mols:
            f.write(f"{_HEAD_NAME}({m.id}).\n")

    bias_path = os.path.join(work_dir, "aleph_bias.pl")
    with open(bias_path, "w") as f:
        f.write("\n".join(ilp._bias_lines(body_predicates)) + "\n")

    return stem, bias_path


def _rewrite_head(rule: str | None) -> str | None:
    """Rename the learned head ``chebi_0`` to the generic ``concept`` target."""
    if not rule:
        return rule
    return re.sub(rf"\b{re.escape(_HEAD_NAME)}\b", TARGET, rule)


def learn_streaming(pos_mols, neg_mols, timeout, on_line) -> dict:
    """Run Aleph on the labelled molecules, mirroring ``ilp.learn_streaming``.

    Aleph's engine is driven by a blocking subprocess that self-terminates within
    ``timeout`` and writes its full trace to a log file, so — unlike Popper — its
    progress cannot be streamed live. We surface a start marker, run to completion
    in this (already backgrounded) worker thread, then replay the engine's output
    into the live log for inspection.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    timeout = timeout or config.DEFAULT_TIMEOUT
    stem, bias_path = _write_problem(config.WORK_DIR, pos_mols, neg_mols)

    on_line(f"Running Aleph via SWI-Prolog (up to {timeout}s)…")
    # icarus's Popper path is non-noisy — it must find a perfectly-separating rule — so run
    # Aleph the same way (chebILP's defaults allow up to noise=200 negatives, meant for the
    # noisy class-scale problems). minpos=1 lets it learn from a single-positive session.
    result = run_ilp_training_aleph(
        _HEAD_ID, stem, bias_path, timeout,
        max_body=config.DEFAULT_ALEPH_MAX_BODY, log_dir=config.WORK_DIR,
        settings_overrides={"noise": 0, "minpos": 1},
    )

    # Replay the engine trace (written by run_ilp_training_aleph) into the log.
    out_path = os.path.join(config.WORK_DIR, "aleph", f"{_HEAD_ID}.out")
    try:
        with open(out_path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.strip():
                    on_line(line)
    except OSError:
        pass

    rule = _rewrite_head(result.get("prog_str"))
    score = result.get("score")
    score_dict = None
    if score:
        tp, fn, tn, fp = score
        score_dict = {"TP": tp, "FN": fn, "TN": tn, "FP": fp}
    return {"rule": rule, "score": score_dict}
