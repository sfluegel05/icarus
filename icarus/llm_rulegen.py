"""Learn a rule for the session concept with the chebILP LLM pipeline.

This drives chebILP's real ``RuleGenerator`` (``generate_auxiliary_rules``) — its
prompt (class name, **textual definition**, superclasses and sibling classes),
reuse retrieval over a shared library, per-predicate validation and feedback round,
and the scored/pruned class hypothesis — rather than reimplementing any of it. The
only thing swapped is the *data source*: :class:`_SessionRuleGenerator` overrides
``class_context`` so the example molecules and validation facts come from the
in-memory session instead of a class's on-disk ``exs.pl``.

The concept's definition is central LLM input. When the session was seeded from a
ChEBI class we start from that class's real definition/siblings; the user can edit
the definition text in the UI, and the edited text is what reaches the model here
(``info['definition']``). The pipeline stores every predicate it writes in a
session-local library (``config.LLM_LIBRARY_DIR``), so later learns can reuse them.
The library also acts as a cache: a ChEBI class that already has a stored hypothesis
is assembled from it without calling the model again.

After the pipeline runs we read the kept auxiliary rules and the class hypothesis
back out and assemble one self-contained rule — the auxiliary clauses followed by
the ``concept(A) :- …`` hypothesis — that grounds through icarus's standard
atom-level background exactly like a Popper- or Aleph-learned rule.
"""

import contextlib
import io
import os
import re

import pandas as pd
from rdkit import Chem

from chebILP.predicate_generation.auxiliary_generation import get_class_info
from chebILP.predicate_generation.auxiliary_rules import (
    load_class_hypothesis,
    load_class_rules,
    resolve_rule_dependencies,
)
from chebILP.predicate_generation.generate_auxiliary_rules import (
    RuleGenerator,
    _build_eval_facts,
)
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever

from . import config

TARGET = config.TARGET_LABEL

# chebILP keys everything (library class_map, hypotheses, logs) by a ChEBI id. A
# session seeded from a class uses that class's real id; an ad-hoc SMILES-only
# session has none, so it learns under this dummy id.
_GENERIC_ID = "0"


def _rows(mols) -> pd.DataFrame:
    """DataFrame with index = molecule id and a ``mol`` column, as
    ``_build_eval_facts`` expects (it iterates ``row.Index`` / ``row.mol``)."""
    return pd.DataFrame({"mol": [m.mol for m in mols]}, index=[m.id for m in mols])


class _StreamToLog(io.TextIOBase):
    """A stdout shim that forwards each completed line to ``on_line``."""

    def __init__(self, on_line):
        self._on_line = on_line
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._on_line(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._on_line(self._buf)
        self._buf = ""


class _SessionRuleGenerator(RuleGenerator):
    """chebILP's rule pipeline, sourced from the session's labelled molecules.

    Only :meth:`class_context` and :meth:`build_retriever` change; the prompt,
    validation, feedback round, hypothesis scoring and library storage are the
    pipeline's own.
    """

    def __init__(self, *args, pos_rows, neg_rows, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_rows = pos_rows
        self._neg_rows = neg_rows

    def build_retriever(self):
        # BM25-only: adequate for a small session-local library and, unlike the dense
        # channel, needs no sentence-transformers model download.
        return HybridPredicateRetriever.from_rule_library(
            base_dir=self.library_dir, use_dense=False)

    def class_context(self, chebi_id) -> dict:
        """Same ctx the parent builds, but over the session molecules rather than a
        class's on-disk train ``exs.pl`` (mirrors ``RuleGenerator.class_context``)."""
        pos_rows, neg_rows = self._pos_rows, self._neg_rows
        val_rows = pd.concat([pos_rows, neg_rows])
        val_facts, val_ids = _build_eval_facts(val_rows, self.computed_facts)
        smiles_by_id = {}
        for rows in (pos_rows, neg_rows):
            for idx, mol in zip(rows.index, rows["mol"]):
                if mol is not None:
                    smiles_by_id[str(idx)] = Chem.MolToSmiles(mol)
        n = self.prompt_samples
        return {
            "pos_smiles": [Chem.MolToSmiles(m) for m in pos_rows["mol"][:n] if m is not None],
            "neg_smiles": [Chem.MolToSmiles(m) for m in neg_rows["mol"][:n] if m is not None],
            "smiles_by_id": smiles_by_id,
            "val_facts": val_facts,
            "val_ids": val_ids,
            "val_rows": val_rows,
            "pos_ids": {str(i) for i in pos_rows.index},
            # Feedback-round bookkeeping (filled by repair; read back by prepare).
            "excluded_labels": set(),
            "excluded_names": set(),
            "excluded_reasons": {},
            "repairs": [],
            "extra_attempts": [],
        }


def _build_info(concept: dict) -> tuple[str, dict]:
    """Return ``(chebi_id, info)`` for the pipeline. When the session was seeded from
    a ChEBI class, start from that class's real name/definition/parents/siblings and
    overlay the user-edited definition; otherwise build a minimal generic info."""
    chebi_id = str(concept.get("chebi_id") or _GENERIC_ID)
    if concept.get("chebi_id"):
        from . import data_store

        info = get_class_info(data_store.graph(), str(concept["chebi_id"]))
    else:
        info = {"name": concept.get("name") or "the concept",
                "definition": None, "parents": [], "siblings": []}
    # The (possibly edited) definition text is central LLM input — it replaces the
    # class's original definition in the prompt.
    if "definition" in concept:
        info["definition"] = (concept.get("definition") or "").strip() or None
    if concept.get("name"):
        info["name"] = concept["name"]
    return chebi_id, info


def _clause_text(source: str) -> str:
    """Just the clause lines of a program (its ``%`` header comments stripped)."""
    return "\n".join(l for l in source.splitlines() if not l.strip().startswith("%")).strip()


def _assemble_rule(programs, hypothesis: str, chebi_id: str) -> str:
    """Kept auxiliary clauses + the class hypothesis, head renamed to ``concept``."""
    head = f"chebi_{chebi_id}"
    hypothesis = re.sub(rf"\b{re.escape(head)}\b", TARGET, hypothesis).strip()
    if not hypothesis.endswith("."):
        hypothesis += "."
    seen, blocks = set(), []
    for prog in programs:
        if prog.name in seen:
            continue
        seen.add(prog.name)
        text = _clause_text(prog.source)
        if text:
            blocks.append(text)
    blocks.append(hypothesis)
    return "\n\n".join(blocks)


def learn_streaming(pos_mols, neg_mols, timeout, on_line, model, concept=None) -> dict:
    """Generate a concept rule with the chebILP LLM pipeline.

    ``timeout`` is unused (the model call is bounded by the CLI); it is accepted so the
    three backends share one signature. ``concept`` carries ``{name, chebi_id,
    definition}`` for the prompt. Returns ``{"rule", "score", "error"?}``.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    chebi_id, info = _build_info(concept or {})
    library_dir = config.LLM_LIBRARY_DIR

    # The library doubles as a cache: a ChEBI class that already has a stored
    # hypothesis is served from it instead of a new model call. (Not for the generic
    # id, which every ad-hoc SMILES-only concept shares.)
    if chebi_id != _GENERIC_ID and os.path.isdir(library_dir):
        cached = _stored_result(chebi_id, library_dir)
        if cached is not None:
            on_line(f"Using the stored hypothesis for CHEBI:{chebi_id} from the rule library "
                    "(no LLM call).")
            cached["score"] = _session_score(cached["rule"], pos_mols, neg_mols)
            return cached

    pos_rows, neg_rows = _rows(pos_mols), _rows(neg_mols)
    os.makedirs(library_dir, exist_ok=True)
    problem_dir = os.path.join(config.WORK_DIR, "llm_problems")  # unused (class_context overridden)

    gen = _SessionRuleGenerator(
        library_dir, model, config.LLM_N_PREDICATES, config.LLM_TOP_K,
        molecules=pd.concat([pos_rows, neg_rows]), problem_dir=problem_dir,
        computed_facts=config.LLM_COMPUTED_FACTS, prompt_samples=config.LLM_PROMPT_SAMPLES,
        pos_rows=pos_rows, neg_rows=neg_rows,
    )
    gen.retriever = gen.build_retriever()

    on_line(f"Running the chebILP LLM rule pipeline with {model}"
            + (f" (concept: {info['name']})" if info.get("name") else "") + "…")
    # The pipeline reports its progress (predicates kept/repaired, hypothesis F1) via
    # print(); forward those lines into the live log.
    try:
        with contextlib.redirect_stdout(_StreamToLog(on_line)):
            gen.generate_for_class(chebi_id, info)
    except Exception as e:
        return {"rule": None, "score": None, "error": f"LLM pipeline failed: {e}"}

    # Read the pipeline's stored result back out and assemble the icarus rule.
    result = _stored_result(chebi_id, library_dir)
    if result is None:
        return {"rule": None, "score": None,
                "error": "The pipeline did not produce a usable hypothesis."}
    return result


def _stored_result(chebi_id: str, library_dir: str) -> dict | None:
    """The class's stored hypothesis (``hypotheses.json``) assembled with its auxiliary
    rules (``class_map.json`` + dependencies) into ``{"rule", "score"}``, or ``None``
    if the library holds no hypothesis for the class."""
    hyp_entry = load_class_hypothesis(chebi_id, library_dir=library_dir) or {}
    hypothesis = (hyp_entry.get("hypothesis") or "").strip()
    if not hypothesis:
        return None
    try:
        programs = load_class_rules(chebi_id, library_dir=library_dir)
    except FileNotFoundError:
        programs = []
    dependencies = resolve_rule_dependencies(programs, library_dir)

    rule = _assemble_rule(programs + dependencies, hypothesis, chebi_id)
    score = None
    if hyp_entry.get("tp") is not None:
        score = {"TP": hyp_entry["tp"], "FN": hyp_entry["fn"],
                 "TN": hyp_entry["tn"], "FP": hyp_entry["fp"]}
    return {"rule": rule, "score": score}


def _session_score(rule: str, pos_mols, neg_mols) -> dict | None:
    """Confusion counts of ``rule`` on the current session examples (a cached
    hypothesis's stored score refers to the examples it was generated on)."""
    from . import ilp

    try:
        preds = ilp.classify(rule, list(pos_mols) + list(neg_mols))
    except ilp.RuleError:
        return None
    tp = sum(preds[m.id] for m in pos_mols)
    fp = sum(preds[m.id] for m in neg_mols)
    return {"TP": tp, "FN": len(pos_mols) - tp, "TN": len(neg_mols) - fp, "FP": fp}
