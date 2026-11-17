"""Merge duplicate predicates in the rule library (``config.LLM_LIBRARY_DIR``).

The LLM pipeline regenerates the same predicate again and again under the same or a
similar name (``aux_carboxy_carbon`` / ``aux_carboxylic_acid_carbon``,
``aux_x`` / ``aux_x_12345``). Two library predicates are merged when

* **syntactic**: their programs are identical up to variable names, literal order,
  helper-predicate names and the names of (already merged) dependencies, or
* **extensional**: they derive exactly the same atoms on a sample of ChEBI molecules
  (firing on at least ``--min-support`` of them) *and* on a second, larger
  independent sample (rare features can coincide on one sample), *and* their names
  are similar (same name after dropping the ``_<chebi id>`` suffix and normalising
  synonyms like carboxyl/carboxylic/carboxylate, or high token overlap). The samples
  use the same background icarus grounds rules on (``ilp.build_background``: atom
  facts plus the computed ``mol_weight``/``ring_size`` facts).

Of each group one predicate is kept (preferably one without a ``_<id>`` suffix, then
the one most classes use, then the shortest name). Every reference to a dropped
predicate in other programs, ``class_map.json`` and ``hypotheses.json`` is rewritten to
the kept one and the dropped programs are deleted.

Usage (from the project root, with the app's Python env)::

    python scripts/dedup_library.py              # dry run: report the groups
    python scripts/dedup_library.py --apply      # back up the library, then merge

The extensions are cached in ``data/work/dedup_extensions_cf_<n>_<seed>.pkl`` (keyed by
the program sources), so re-running with other thresholds is fast. The report of the
groups (and of the rejected candidate pairs) goes to ``data/work/dedup_report.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icarus import config  # noqa: E402

LIB = config.LLM_LIBRARY_DIR
PROGRAMS = os.path.join(LIB, "programs")
CACHE = os.path.join(config.WORK_DIR, "dedup_extensions_cf_{n}_{seed}.pkl")
BATCH = 25  # programs ground together per clingo call

_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z_][A-Za-z0-9_]*)")
_PRED_RE = re.compile(r"\b(aux_[A-Za-z0-9_]*)\b")
_ID_SUFFIX_RE = re.compile(r"_\d{3,}$")


# ── loading ─────────────────────────────────────────────────────────────────

def load_programs() -> dict:
    from chebILP.predicate_generation.auxiliary_rules import parse_rule_program

    progs = {}
    for fn in sorted(os.listdir(PROGRAMS)):
        if not fn.endswith(".pl"):
            continue
        path = os.path.join(PROGRAMS, fn)
        with open(path, encoding="utf-8") as f:
            prog = parse_rule_program(f.read(), source_file=path)
        if prog is not None and prog.name == fn[:-3]:
            progs[prog.name] = prog
    return progs


def _clauses(source: str) -> list[str]:
    from icarus.ilp import _split_clauses

    return _split_clauses(source)


def _head(clause: str) -> str | None:
    from icarus.ilp import _clause_head

    return _clause_head(clause)


# ── extensional signatures ──────────────────────────────────────────────────

_FACTS = None
_IDS = None


def _init_worker(facts, ids):
    global _FACTS, _IDS
    _FACTS, _IDS = facts, ids


def _namespaced(i: int, sources) -> str:
    """The program + its dependencies with every predicate they define prefixed ``p<i>_``,
    so several programs (whose helpers may share names) can be ground in one instance."""
    defined = set()
    for _n, src in sources:
        defined |= {h for h in (_head(c) for c in _clauses(src)) if h}
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(defined, key=len, reverse=True)))
                     + r")\b(?=\s*\()") if defined else None
    text = "\n".join("\n".join(_clauses(src)) for _n, src in sources)
    return pat.sub(lambda m: f"p{i}_{m.group(1)}", text) if pat else text


def _extensions(batch):
    """``{name: sorted derived atoms}`` for a batch of ``(name, sources)`` on the sample.

    The batch is ground in one clingo instance; if that fails, each program alone.
    """
    from chebILP.predicate_generation.auxiliary_rules import RuleProgram, derive_rule_extensions

    def run(items):
        progs = [RuleProgram(name=f"p{i}_{name}", source=_namespaced(i, sources))
                 for i, (name, sources) in enumerate(items)]
        ext = derive_rule_extensions(progs, _FACTS, _IDS, timeout=300)
        return {name: sorted({t for ts in ext.get(f"p{i}_{name}", {}).values() for t in ts})
                for i, (name, _s) in enumerate(items)}

    try:
        return run(batch), []
    except Exception:  # noqa: BLE001
        pass
    out, errors = {}, []
    for item in batch:
        try:
            out.update(run([item]))
        except Exception as e:  # noqa: BLE001
            out[item[0]] = None
            errors.append(f"{item[0]}: {str(e)[:200]}")
    return out, errors


def sample_background(n: int, seed: int):
    import types

    import pandas as pd

    from icarus.ilp import build_background

    df = pd.read_pickle(config.MOLECULES_PKL)
    df = df[df["mol"].notna()]
    rows = df.sample(n=min(n, len(df)), random_state=seed)
    # The bk uses the molecule id in atom ids; keep it compact and stable.
    mols = [types.SimpleNamespace(id=f"m{i}", mol=m) for i, m in zip(rows.index, rows["mol"])]
    facts, _ = build_background(mols)
    return facts, [m.id for m in mols]


def compute_extensions(progs: dict, n_mols: int, seed: int, workers: int) -> dict:
    cache_path = CACHE.format(n=n_mols, seed=seed)
    from chebILP.predicate_generation.auxiliary_rules import resolve_rule_dependencies

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    key_meta = (n_mols, seed)
    if cache.get("_meta") != key_meta:
        cache = {"_meta": key_meta}

    jobs = {}
    for name, prog in progs.items():
        deps = resolve_rule_dependencies([prog], LIB)
        sources = [(name, prog.source)] + [(d.name, d.source) for d in deps]
        key = hashlib.sha1(json.dumps(sources).encode()).hexdigest()
        if key not in cache:
            jobs[name] = (key, sources)
    print(f"{len(progs)} programs, {len(progs) - len(jobs)} cached, {len(jobs)} to ground",
          flush=True)

    if jobs:
        t0 = time.time()
        facts, ids = sample_background(n_mols, seed)
        print(f"background for {len(ids)} molecules: {len(facts)} facts "
              f"({time.time() - t0:.0f} s)", flush=True)
        items = [(n, src) for n, (_k, src) in jobs.items()]
        batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
        done = 0
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(facts, ids)) as pool:
            for fut in as_completed([pool.submit(_extensions, b) for b in batches]):
                result, errors = fut.result()
                for err in errors:
                    print(f"  ! {err}", flush=True)
                for name, tuples in result.items():
                    cache[jobs[name][0]] = tuples
                done += len(result)
                print(f"  {done}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    out = {}
    for name, prog in progs.items():
        deps = resolve_rule_dependencies([prog], LIB)
        sources = [(name, prog.source)] + [(d.name, d.source) for d in deps]
        out[name] = cache.get(hashlib.sha1(json.dumps(sources).encode()).hexdigest())
    return out


# ── syntactic canonical form ────────────────────────────────────────────────

def _split_literals(body: str) -> list[str]:
    from chebILP.utils import split_prolog_literals

    return [l.strip() for l in split_prolog_literals(body) if l.strip()]


def canonical_form(prog, rename: dict[str, str]) -> str:
    """The program up to variable names, literal order and helper names.

    ``rename`` maps library predicate names to their group representative, so programs
    that build on duplicates of each other compare equal too. The defined predicates
    (head + local helpers) become ``P0, P1, …`` in order of definition.
    """
    clauses = _clauses(prog.source)
    defined = []
    for c in clauses:
        h = _head(c)
        if h and h not in defined:
            defined.append(h)
    # The head first, so the program's own predicate is always P0.
    if prog.name in defined:
        defined.remove(prog.name)
    defined.insert(0, prog.name)
    local = {n: f"P{i}" for i, n in enumerate(defined)}

    def pred(m):
        n = m.group(1)
        return local.get(n) or rename.get(n, n)

    out = []
    for c in clauses:
        c = c.rstrip(".")
        head, _, body = c.partition(":-")
        head = _PRED_RE.sub(pred, head.strip())
        lits = [_PRED_RE.sub(pred, l) for l in _split_literals(body)] if body else []
        # Order literals by their variable-free shape, then number the variables by
        # first occurrence (head first). Ties make this an approximation (two
        # equivalent programs may get different forms), never a false merge: equal
        # forms are always renamings of each other.
        lits.sort(key=lambda l: _VAR_RE.sub("V", l))
        names: dict[str, str] = {}

        def var(m):
            v = m.group(1)
            if v == "_":
                return "_"
            names.setdefault(v, f"V{len(names)}")
            return names[v]

        text = _VAR_RE.sub(var, head) + (" :- " + ", ".join(_VAR_RE.sub(var, l) for l in lits)
                                         if lits else "")
        out.append(text)
    return "\n".join(sorted(out))


# ── name similarity ─────────────────────────────────────────────────────────

_SYNONYMS = {
    "carboxyl": "carboxy", "carboxylic": "carboxy", "carboxylate": "carboxy",
    "carboxylates": "carboxy", "acid": "carboxy",
    "hydroxyl": "hydroxy", "hydroxylated": "hydroxy",
    "amino": "amine", "ammonium": "amine",
    "phosphate": "phospho", "phosphoryl": "phospho", "phosphono": "phospho",
    "phosphoric": "phospho",
    "sulfonate": "sulfo", "sulfonic": "sulfo", "sulfate": "sulfo",
    "ketone": "oxo", "keto": "oxo", "carbonyl": "oxo",
    "nitrogen": "n", "oxygen": "o", "carbon": "c", "sulfur": "s",
    "atom": "", "group": "", "groups": "", "has": "", "any": "", "is": "", "a": "",
    "the": "", "of": "", "on": "", "in": "", "with": "", "mol": "", "molecule": "",
    "contains": "", "present": "",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "single": "1", "double": "2", "triple": "3",
}


def _singular(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def name_tokens(name: str) -> frozenset[str]:
    base = _ID_SUFFIX_RE.sub("", name)
    base = base[4:] if base.startswith("aux_") else base
    toks = set()
    for t in base.split("_"):
        t = _singular(t.lower())
        t = _SYNONYMS.get(t, t)
        if t:
            toks.add(t)
    return frozenset(toks)


def names_similar(a: str, b: str) -> bool:
    ta, tb = name_tokens(a), name_tokens(b)
    if ta == tb:
        return True
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


# ── grouping ────────────────────────────────────────────────────────────────

class _UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _usage(class_map: dict) -> dict[str, int]:
    use = defaultdict(int)
    for names in class_map.values():
        for n in names:
            use[n] += 1
    return use


def representative(group, use: dict[str, int]) -> str:
    return min(group, key=lambda n: (bool(_ID_SUFFIX_RE.search(n)), -use.get(n, 0), len(n), n))


_ATOM_ID_RE = re.compile(r"^a(m\d+)_\d+$")


def _molecules(tuples) -> set[str]:
    """Sample molecules (``m<id>``) an extension touches: molecule or atom (``am<id>_<i>``) args."""
    mols = set()
    for t in tuples:
        for a in t:
            m = _ATOM_ID_RE.match(a)
            if m:
                mols.add(m.group(1))
            elif re.fullmatch(r"m\d+", a):
                mols.add(a)
    return mols


def _ext_key(tuples):
    return hashlib.sha1(repr(tuples).encode()).hexdigest() if tuples is not None else None


def extension_pairs(extensions, min_support: int) -> list[tuple[str, str]]:
    """Pairs of similarly named predicates with the same non-trivial extension."""
    by_ext = defaultdict(list)
    for name, tuples in extensions.items():
        if not tuples or len(_molecules(tuples)) < min_support:
            continue
        by_ext[(len(tuples[0]), _ext_key(tuples))].append(name)
    pairs = []
    for names in by_ext.values():
        names.sort()
        pairs += [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                  if names_similar(a, b)]
    return pairs


def find_groups(progs, pairs, class_map: dict):
    """Union the (verified) extensional ``pairs`` with syntactic duplicates."""
    uf = _UF()
    reasons = {}
    for a, b in pairs:
        if uf.find(a) != uf.find(b):
            reasons[(a, b)] = "extension"
        uf.union(a, b)

    # Syntactic, to a fixpoint (a merge can make programs that build on the merged
    # predicates equal too).
    while True:
        groups = defaultdict(list)
        for n in progs:
            groups[uf.find(n)].append(n)
        use = _usage(class_map)
        rename = {n: representative(g, use) for g in groups.values() for n in g}
        by_form = defaultdict(list)
        for name, prog in progs.items():
            by_form[canonical_form(prog, rename)].append(name)
        changed = False
        for names in by_form.values():
            for b in names[1:]:
                if uf.find(names[0]) != uf.find(b):
                    reasons[(names[0], b)] = "syntax"
                    uf.union(names[0], b)
                    changed = True
        if not changed:
            break

    groups = defaultdict(list)
    for n in progs:
        groups[uf.find(n)].append(n)
    use = _usage(class_map)
    return [sorted(g) for g in groups.values() if len(g) > 1], reasons, use


# ── applying ────────────────────────────────────────────────────────────────

def _local_helpers(prog) -> set[str]:
    return {h for h in (_head(c) for c in _clauses(prog.source)) if h and h != prog.name}


def apply(groups, progs, use, class_map, hypotheses):
    rename = {}
    for g in groups:
        keep = representative(g, use)
        for n in g:
            if n != keep:
                rename[n] = keep
    if not rename:
        return rename

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{LIB}.bak-{stamp}"
    shutil.copytree(LIB, backup)
    print(f"backed up the library to {backup}")

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(rename, key=len,
                                                                  reverse=True))) + r")\b")

    def sub(text, keep_local=frozenset()):
        return pattern.sub(lambda m: m.group(1) if m.group(1) in keep_local
                           else rename[m.group(1)], text)

    for name, prog in progs.items():
        path = os.path.join(PROGRAMS, f"{name}.pl")
        if name in rename:
            os.remove(path)
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # A program defining a dropped name itself (as a local helper) keeps it: that
        # helper is its own definition, not a reference to the dropped program.
        new = sub(text, keep_local=frozenset(_local_helpers(prog)))
        if new != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)

    new_map = {cid: list(dict.fromkeys(rename.get(n, n) for n in names))
               for cid, names in class_map.items()}
    with open(os.path.join(LIB, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump(new_map, f, indent=2)
    for entry in hypotheses.values():
        if entry.get("hypothesis"):
            entry["hypothesis"] = sub(entry["hypothesis"])
    with open(os.path.join(LIB, "hypotheses.json"), "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2)
    return rename


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="rewrite the library (after a backup)")
    ap.add_argument("--molecules", type=int, default=3000, help="sample size for extensions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--verify-molecules", type=int, default=15000,
                    help="independent sample every extensional match must also agree on "
                         "(0 = skip)")
    ap.add_argument("--verify-workers", type=int, default=6,
                    help="workers for the (memory-hungry) verification sample")
    ap.add_argument("--min-support", type=int, default=3,
                    help="molecules an extension must fire on to count as evidence")
    ap.add_argument("--report", default=os.path.join(config.WORK_DIR, "dedup_report.txt"))
    args = ap.parse_args()

    progs = load_programs()
    with open(os.path.join(LIB, "class_map.json"), encoding="utf-8") as f:
        class_map = json.load(f)
    with open(os.path.join(LIB, "hypotheses.json"), encoding="utf-8") as f:
        hypotheses = json.load(f)

    extensions = compute_extensions(progs, args.molecules, args.seed, args.workers)
    pairs = extension_pairs(extensions, args.min_support)
    # Rare features can coincide on one sample ("no ring > 8" vs "no ring > 10"): every
    # extensional candidate must also agree on a larger, independent sample.
    if args.verify_molecules and pairs:
        cand = {n for p in pairs for n in p}
        print(f"verifying {len(pairs)} extensional pairs ({len(cand)} predicates) on "
              f"{args.verify_molecules} more molecules", flush=True)
        verify = compute_extensions({n: progs[n] for n in cand}, args.verify_molecules,
                                    args.seed + 1, args.verify_workers)
        rejected = [(a, b) for a, b in pairs
                    if verify[a] is None or _ext_key(verify[a]) != _ext_key(verify[b])]
        with open(args.report + ".rejected", "w", encoding="utf-8") as f:
            for a, b in rejected:
                f.write(f"{a}  ({progs[a].description})\n{b}  ({progs[b].description})\n\n")
        pairs = [p for p in pairs if p not in set(rejected)]
        print(f"  {len(rejected)} pairs rejected (see {args.report}.rejected)")
    groups, reasons, use = find_groups(progs, pairs, class_map)

    # Programs whose local helpers other programs build on can't be dropped safely
    # (the helper would vanish); they still serve as group representatives.
    referenced = defaultdict(set)
    for name, prog in progs.items():
        for ref in set(_PRED_RE.findall(prog.source)):
            referenced[ref].add(name)
    safe_groups = []
    for g in groups:
        keep = representative(g, use)
        ok = [n for n in g if n == keep or not any(
            referenced[h] - {n} for h in _local_helpers(progs[n]))]
        if len(ok) > 1:
            safe_groups.append(ok)

    n_drop = sum(len(g) - 1 for g in safe_groups)
    with open(args.report, "w", encoding="utf-8") as f:
        for g in sorted(safe_groups, key=lambda g: -len(g)):
            keep = representative(g, use)
            f.write(f"KEEP {keep}  ({progs[keep].description})\n")
            for n in g:
                if n != keep:
                    f.write(f"  drop {n}  ({progs[n].description})\n")
            f.write("\n")
    by_reason = defaultdict(int)
    for r in reasons.values():
        by_reason[r] += 1
    print(f"{len(safe_groups)} duplicate groups, {n_drop} of {len(progs)} predicates to drop "
          f"(merges by {dict(by_reason)}); report: {args.report}")

    if args.apply:
        rename = apply(safe_groups, progs, use, class_map, hypotheses)
        print(f"merged {len(rename)} predicates into {len(set(rename.values()))}")


if __name__ == "__main__":
    main()
