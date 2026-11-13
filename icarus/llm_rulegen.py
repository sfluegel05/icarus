"""Learn a rule for the session concept with the chebILP LLM pipeline.

This drives chebILP's real ``RuleGenerator`` (``generate_auxiliary_rules``) — its
prompt (class name, **textual definition**, superclasses and sibling classes),
reuse retrieval over a shared library, per-predicate validation, and the
scored/pruned class hypothesis — rather than reimplementing any of it. The only thing
swapped is the *data source*: :class:`_SessionRuleGenerator` overrides
``class_context`` so the example molecules and validation facts come from the
in-memory session instead of a class's on-disk ``exs.pl``, and its dataset
(:class:`_SessionDataset`) is chebILP's ChEBI dataset over the session molecules.

The concept's definition is central LLM input. When the session was seeded from a
ChEBI class we start from that class's real definition/siblings; the user can edit
the definition text in the UI, and the edited text is what reaches the model here
(``info['definition']``).

The rule library (``config.LLM_LIBRARY_DIR``) is a **read-only** resource: the
pipeline retrieves reuse candidates from it, but each run works on a throwaway copy,
so nothing a session generates (from possibly edited examples or definition) is ever
written back. The library also acts as a cache: a ChEBI class that already has a
stored hypothesis is assembled from it without calling the model, unless the user
explicitly asks for a fresh model call (``use_cache=False``).

After the pipeline runs we read the kept auxiliary rules and the class hypothesis
back out and assemble one self-contained rule — the auxiliary clauses followed by
the ``concept(A) :- …`` hypothesis — that grounds through icarus's standard
atom-level background exactly like a Popper- or Aleph-learned rule.
"""

import contextlib
import io
import os
import re
import shutil

import pandas as pd
from rdkit import Chem

from chebILP.predicate_generation.auxiliary_rules import (
    aux_rule_path,
    load_class_hypothesis,
    load_class_rules,
    prune_hypothesis,
    resolve_rule_dependencies,
)
from chebILP.predicate_generation.generate_auxiliary_rules import RuleGenerator
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever
from chebILP.preprocessing.chebi.chebi_dataset import ChEBIDataset
from chebILP.preprocessing.chebi.fg_matching import (
    FG_SEED_SOURCES,
    is_fg_seed_name,
    seed_fg_library,
)

from . import config, efg, llm_guard

llm_guard.install()
efg.install()

TARGET = config.TARGET_LABEL

# chebILP keys everything (library class_map, hypotheses, logs) by a ChEBI id. A
# session seeded from a class uses that class's real id; an ad-hoc SMILES-only
# session has none, so it learns under this dummy id.
_GENERIC_ID = "0"


def _rows(mols) -> pd.DataFrame:
    """DataFrame with index = molecule id and a ``mol`` column, as chebILP's
    background builders expect (they iterate ``row.Index`` / ``row.mol``)."""
    return pd.DataFrame({"mol": [m.mol for m in mols]}, index=[m.id for m in mols])


class _SessionDataset(ChEBIDataset):
    """chebILP's ChEBI dataset over the session: the ChEBI graph for class info
    (name, definition, parents, siblings), the session molecules as the examples.
    Prompt text, background encoding and functional-group facts are the dataset's own;
    nothing is read from or written to the dataset's directories."""

    def __init__(self, molecules: pd.DataFrame, graph=None):
        super().__init__(version=str(config.CHEBI_VERSION), base_dir=config.ICARUS_DATA)
        self.molecules = molecules
        self.graph = graph

    def _fg_matches(self, source_key, rows):
        # The parent caches matches over the whole dataset on disk; a session is small.
        return FG_SEED_SOURCES[source_key].matcher(rows)


class _StreamToLog(io.TextIOBase):
    """A stdout shim that forwards each completed line to ``on_line``."""

    def __init__(self, on_line):
        self._on_line = on_line
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # The pipeline's server-side log path means nothing to a user.
            if line.strip() and "logged full exchange to" not in line:
                self._on_line(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._on_line(self._buf)
        self._buf = ""


class _SessionRuleGenerator(RuleGenerator):
    """chebILP's rule pipeline, sourced from the session's labelled molecules.

    Only :meth:`class_context` and :meth:`build_retriever` change; the prompt,
    validation, hypothesis scoring and library storage are the pipeline's own.
    """

    def __init__(self, *args, pos_rows, neg_rows, use_aux=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_rows = pos_rows
        self._neg_rows = neg_rows
        self._use_aux = use_aux

    def build_retriever(self):
        # BM25-only: adequate for a small session-local library and, unlike the dense
        # channel, needs no sentence-transformers model download. The retriever splits
        # its candidates half seeded functional groups (efg_*), half aux predicates.
        retriever = HybridPredicateRetriever.from_rule_library(
            base_dir=self.library_dir, use_dense=False)
        if not self._use_aux:
            # Library aux predicates switched off: offer only the functional groups
            # for reuse (the model still writes new predicates of its own).
            retriever = HybridPredicateRetriever(
                [e for e in retriever.entries if is_fg_seed_name(e["stem"])], use_dense=False)
        return retriever

    def class_context(self, chebi_id) -> dict:
        """Same ctx the parent builds, but over the session molecules rather than a
        class's on-disk train ``exs.pl`` (mirrors ``RuleGenerator.class_context``)."""
        pos_rows, neg_rows = self._pos_rows, self._neg_rows
        val_rows = pd.concat([pos_rows, neg_rows])
        val_rows = val_rows[val_rows["mol"].notna()]
        val_facts = list(self.dataset.build_background(val_rows)[0])
        if self.computed_facts:
            val_facts += self.dataset.build_computed_facts(val_rows)
        n = self.prompt_samples
        return {
            "pos_examples": [Chem.MolToSmiles(m) for m in pos_rows["mol"][:n] if m is not None],
            "neg_examples": [Chem.MolToSmiles(m) for m in neg_rows["mol"][:n] if m is not None],
            "val_facts": val_facts,
            "val_ids": [str(i) for i in val_rows.index],
            "val_rows": val_rows,
            "pos_ids": {str(i) for i in pos_rows.index},
            "neg_ids": {str(i) for i in neg_rows.index},
        }


def _build_info(concept: dict, dataset: ChEBIDataset) -> tuple[str, dict]:
    """Return ``(chebi_id, info)`` for the pipeline. When the session was seeded from
    a ChEBI class, start from that class's real name/definition/parents/siblings
    (``dataset.class_info``) and overlay the user-edited definition; otherwise build a
    minimal generic info."""
    chebi_id = str(concept.get("chebi_id") or _GENERIC_ID)
    if chebi_id != _GENERIC_ID and dataset.graph is not None and chebi_id in dataset.graph:
        info = dataset.class_info(chebi_id)
    else:
        info = {"id": chebi_id, "label": dataset.class_label(chebi_id)
                if chebi_id != _GENERIC_ID else "user-defined concept",
                "name": concept.get("name") or "the concept",
                "definition": None, "parents": [], "siblings": []}
    # The (possibly edited) definition text is central LLM input — it replaces the
    # class's original definition in the prompt.
    if "definition" in concept:
        info["definition"] = (concept.get("definition") or "").strip() or None
    if concept.get("name"):
        info["name"] = concept["name"]
    return chebi_id, info


def _assemble_rule(programs, hypothesis: str, chebi_id: str) -> str:
    """Kept auxiliary programs + the class hypothesis, head renamed to ``concept``.

    Each program keeps its ``% PREDICATE_NAME`` / ``% DESCRIPTION`` header: predicates
    generated in a run are not in the (read-only) library, so the rule itself is where
    the NL translation and the block palette find their descriptions."""
    head = f"chebi_{chebi_id}"
    hypothesis = re.sub(rf"\b{re.escape(head)}\b", TARGET, hypothesis).strip()
    if not hypothesis.endswith("."):
        hypothesis += "."
    seen, blocks = set(), []
    for prog in programs:
        # A reused functional group (efg_*) is a header-only seed: its facts come from
        # RDKit matching at classify time, not from a clause in the rule.
        if prog.name in seen or is_fg_seed_name(prog.name):
            continue
        seen.add(prog.name)
        text = prog.source.strip()
        if text:
            blocks.append(text)
    blocks.append(hypothesis)
    return "\n\n".join(blocks)


def learn_streaming(pos_mols, neg_mols, timeout, on_line, model, concept=None,
                    admit=None, use_cache=True) -> dict:
    """Generate a concept rule with the chebILP LLM pipeline.

    ``timeout`` is unused (the model calls are bounded by ``config.LLM_TIMEOUT``, see
    ``llm_guard.deadline``); it is accepted so the three backends share one signature. ``concept`` carries ``{name, chebi_id,
    definition}`` for the prompt. ``admit()``, if given, is called right before the
    model is prompted and returns a refusal message (quota used up) or ``None``.
    ``use_cache=False`` skips the library's stored hypothesis and prompts the model.
    Returns ``{"rule", "score", "cached"?, "error"?}``.
    """
    if not pos_mols or not neg_mols:
        return {"rule": None, "score": None,
                "error": "Need at least one positive and one negative example."}

    from . import data_store

    pos_rows, neg_rows = _rows(pos_mols), _rows(neg_mols)
    dataset = _SessionDataset(pd.concat([pos_rows, neg_rows]), data_store.graph())
    chebi_id, info = _build_info(concept or {}, dataset)
    library_dir = config.LLM_LIBRARY_DIR

    # The library doubles as a cache: a ChEBI class that already has a stored
    # hypothesis is served from it instead of a new model call. (Not for the generic
    # id, which every ad-hoc SMILES-only concept shares.)
    if use_cache and chebi_id != _GENERIC_ID and os.path.isdir(library_dir):
        cached = _stored_result(chebi_id, library_dir)
        if cached is not None:
            on_line(f"The rule library already holds an LLM-generated rule for "
                    f"CHEBI:{chebi_id} — using this cached result (no LLM call).")
            on_line("It was generated earlier from the original ChEBI class, not from "
                    "your current examples or definition edits.")
            on_line('To generate a fresh rule for your current examples and definition, '
                    'click "Prompt the LLM".')
            cached.pop("repaired", None)
            cached["score"] = _session_score(cached["rule"], pos_mols, neg_mols)
            cached["cached"] = True
            return cached

    if admit is not None:
        refusal = admit()
        if refusal:
            on_line(refusal)  # keep it visible in the output panel, not only the toast
            return {"rule": None, "score": None, "error": refusal}

    from . import ilp

    # The pipeline writes what it generates into its library; give it a throwaway
    # copy so the shared library stays read-only (and concurrent runs can't collide).
    with ilp.run_dir("llm_") as work_dir:
        run_library = os.path.join(work_dir, "rule_library")
        if os.path.isdir(library_dir):
            shutil.copytree(library_dir, run_library,
                            ignore=shutil.ignore_patterns("generation_logs"))
        else:
            os.makedirs(os.path.join(run_library, "programs"))
        use_aux, use_efg = ilp.library_options(concept)
        if use_efg:
            # Offer the Extended Functional Groups for reuse, as chebILP's
            # --seed_predicates efg does (header-only efg_* programs in the library).
            seed_fg_library(run_library, "efg")
        result = _generate(chebi_id, info, dataset, pos_rows, neg_rows, on_line, model,
                           run_library, work_dir, use_aux=use_aux)
    if result.pop("repaired", False):
        result["score"] = _session_score(result["rule"], pos_mols, neg_mols)
    return result


def _generate(chebi_id, info, dataset, pos_rows, neg_rows, on_line, model, library_dir,
              work_dir, use_aux=True) -> dict:
    problem_dir = os.path.join(work_dir, "problems")  # unused (class_context overridden)

    gen = _SessionRuleGenerator(
        library_dir, model, config.LLM_N_PREDICATES, config.LLM_TOP_K,
        dataset=dataset, problem_dir=problem_dir,
        computed_facts=config.LLM_COMPUTED_FACTS, prompt_samples=config.LLM_PROMPT_SAMPLES,
        pos_rows=pos_rows, neg_rows=neg_rows, use_aux=use_aux,
    )
    gen.retriever = gen.build_retriever()

    on_line(f"Running the chebILP LLM rule pipeline with {model}"
            + (f" (concept: {info['name']})" if info.get("name") else "") + "…")
    # The pipeline reports its progress (predicates kept/repaired, hypothesis F1) via
    # print(); forward those lines into the live log.
    timeout_msg = (f"The LLM did not finish within the time limit of "
                   f"{config.LLM_TIMEOUT} s; no rule was learned. Try again, or use "
                   "fewer/simpler examples.")
    try:
        with llm_guard.deadline(config.LLM_TIMEOUT), \
                llm_guard.progress(on_line, info.get("name"), len(pos_rows), len(neg_rows)), \
                contextlib.redirect_stdout(_StreamToLog(on_line)):
            try:
                gen.generate_for_class(chebi_id, info)
            finally:
                timed_out = llm_guard.expired()
    except llm_guard.LLMTimeout:
        return {"rule": None, "score": None, "error": timeout_msg}
    except Exception as e:
        return {"rule": None, "score": None, "error": f"LLM pipeline failed: {e}"}

    # Read the pipeline's stored result back out and assemble the icarus rule.
    result = _stored_result(chebi_id, library_dir)
    if result is None:
        if timed_out:
            return {"rule": None, "score": None, "error": timeout_msg}
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

    repaired = _repair_hypothesis(hypothesis, programs, dependencies, library_dir)
    if repaired is None:
        return None
    from . import ilp

    assembled = _assemble_rule(programs + dependencies, repaired, chebi_id)
    # Library predicates the hypothesis calls by name but the class does not define.
    rule = ilp.expand_aux_definitions(assembled)
    # A repaired rule classifies differently from the one the stored score is for.
    changed = repaired != hypothesis or rule != assembled
    score = None
    if hyp_entry.get("tp") is not None and not changed:
        score = {"TP": hyp_entry["tp"], "FN": hyp_entry["fn"],
                 "TN": hyp_entry["tn"], "FP": hyp_entry["fp"]}
    return {"rule": rule, "score": score, "repaired": changed}


_AUX_CALL_RE = re.compile(r"\b(aux_[A-Za-z0-9_]*)(?=\s*\()")


def _repair_hypothesis(hypothesis: str, programs, dependencies, library_dir: str) -> str | None:
    """The stored hypothesis with its calls matched to the predicates the class has.

    chebILP stores a new predicate whose name is taken in the library under a
    class-suffixed name (``aux_no_nitrogen`` -> ``aux_no_nitrogen_78206``), but the
    stored hypothesis keeps the name the model wrote, so it calls another class's
    predicate, or one that does not exist (and then never holds). Each such call is
    renamed to the class's own suffixed predicate when there is exactly one; a call to
    a library predicate of that name is left to its library definition; a call to a
    predicate that exists nowhere is dropped as chebILP's ``prune_hypothesis`` does.
    Returns ``None`` if no usable clause is left.
    """
    available = {p.name for p in programs + dependencies}
    own = [p.name for p in programs]
    head, sep, body = hypothesis.partition(":-")
    if not sep:
        return hypothesis
    renames, missing = {}, set()
    for name in dict.fromkeys(_AUX_CALL_RE.findall(body)):
        if name in available:
            continue
        # chebILP's collision names: <name>_<class id>, then <name>_<class id>__<n>.
        suffix = re.compile(rf"{re.escape(name)}(_\d+)(__\d+)?|{re.escape(name)}__\d+")
        suffixed = [n for n in own if suffix.fullmatch(n)]
        if len(suffixed) == 1:
            renames[name] = suffixed[0]
        elif not os.path.exists(aux_rule_path(name, library_dir)):
            missing.add(name)
        # else: a library predicate of that name; its definition is added to the rule
    if renames:
        body = _AUX_CALL_RE.sub(lambda m: renames.get(m.group(1), m.group(1)), body)
    repaired = f"{head.strip()} :- {body.strip()}" if renames else hypothesis
    if not missing:
        return repaired
    return prune_hypothesis(repaired, {n for n in _AUX_CALL_RE.findall(body)} - missing)


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
