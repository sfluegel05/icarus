"""Starlette backend for the ICaRuS demo.

A single in-memory session drives the whole flow:
  1. seed positive/negative examples from a ChEBI class and/or SMILES/InChI input
  2. get fingerprint-similar suggestions (with ILP predictions once a rule exists)
  3. learn a Prolog rule with Popper, see which molecules it misclassifies
  4. hand-edit the rule and see which classifications changed
  5. iterate: add samples, re-suggest, re-learn
"""

import contextlib
import threading
import traceback

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from chebi_utils.read_molecule import smiles_or_inchi_to_mol

from . import config, data_store, ilp, render, similarity
from .session import SESSION


# ── prediction / reporting helpers ─────────────────────────────────────────
def _outcome(label: str, predicted: bool) -> str:
    if label == "pos":
        return "TP" if predicted else "FN"
    if label == "neg":
        return "FP" if predicted else "TN"
    return "pos_pred" if predicted else "neg_pred"  # unlabeled


def recompute_predictions() -> dict[str, bool]:
    """Classify every session molecule under the current rule and cache it."""
    if not SESSION.current_rule:
        SESSION.last_predictions = {}
        return {}
    mols = list(SESSION.molecules.values())
    preds = ilp.classify(SESSION.current_rule, mols)
    SESSION.last_predictions = preds
    return preds


def _report(preds: dict[str, bool]) -> dict:
    """Confusion matrix + misclassified molecule ids over labelled molecules."""
    conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    mismatches = []
    for m in SESSION.labeled():
        o = _outcome(m.label, preds.get(m.id, False))
        conf[o] += 1
        if o in ("FP", "FN"):
            mismatches.append({"id": m.id, "name": m.name, "smiles": m.smiles,
                               "label": m.label, "outcome": o})
    return {"confusion": conf, "mismatches": mismatches}


def state_dict() -> dict:
    preds = SESSION.last_predictions if SESSION.current_rule else {}
    mols = []
    for m in SESSION.molecules.values():
        d = m.to_dict()
        if SESSION.current_rule:
            p = preds.get(m.id, False)
            d["prediction"] = p
            d["outcome"] = _outcome(m.label, p)
        else:
            d["prediction"] = None
            d["outcome"] = None
        mols.append(d)
    n_pos = sum(1 for m in SESSION.molecules.values() if m.label == "pos")
    n_neg = sum(1 for m in SESSION.molecules.values() if m.label == "neg")
    out = {
        "molecules": mols,
        "counts": {"pos": n_pos, "neg": n_neg,
                   "unlabeled": len(SESSION.molecules) - n_pos - n_neg},
        "learned_rule": SESSION.learned_rule,
        "current_rule": SESSION.current_rule,
        "has_rule": bool(SESSION.current_rule),
        "edited": bool(SESSION.current_rule and SESSION.current_rule != SESSION.learned_rule),
    }
    if SESSION.current_rule:
        out["report"] = _report(preds)
        out["rule_nl"] = ilp.rule_to_nl(SESSION.current_rule)
    return out


def _int(v, default):
    """Parse an int from user input, falling back to ``default`` on bad values."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def ok(**extra):
    # Refresh predictions so any molecule added/relabelled since the last learn is
    # classified before the report is built — otherwise new molecules default to
    # "negative" and skew the confusion matrix / mismatch list.
    if SESSION.current_rule:
        try:
            recompute_predictions()
        except ilp.RuleError:
            pass
    d = state_dict()
    d.update(extra)
    return JSONResponse(d)


# ── endpoints ───────────────────────────────────────────────────────────────
async def index(request: Request):
    return FileResponse(f"{config.WEB_DIR}/index.html")


async def depict(request: Request):
    """Return an SVG 2D depiction for a session molecule (``id``), a ChEBI
    molecule (``chebi_id``), or an ad-hoc ``smiles``/InChI string."""
    q = request.query_params
    w = _int(q.get("w"), 280)
    h = _int(q.get("h"), 220)
    mol = None
    if q.get("id"):
        m = SESSION.molecules.get(q["id"])
        mol = m.mol if m else None
    elif q.get("chebi_id"):
        df = data_store.molecules()
        cid = str(q["chebi_id"])
        if cid in df.index:
            mol = df.loc[cid, "mol"]
    elif q.get("smiles"):
        mol = smiles_or_inchi_to_mol(q["smiles"])
    svg = render.mol_to_svg(mol, w, h)
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=3600"})


async def explain(request: Request):
    """Explain why the current rule classifies a session molecule as positive."""
    mid = request.query_params.get("id", "")
    # Snapshot the molecule + rule under the lock, then run the (slow) explanation
    # outside it so a concurrent reset/remove can't race the read.
    with SESSION.lock:
        m = SESSION.molecules.get(mid)
        rule = SESSION.current_rule
        smiles = m.smiles if m else None
        name = m.name if m else None
    if m is None:
        return JSONResponse({"error": "Unknown molecule."}, status_code=404)
    if not rule:
        return JSONResponse({"error": "No rule yet."}, status_code=400)
    try:
        data = ilp.explain_molecule(smiles, rule)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Could not build explanation: {e}"}, status_code=500)
    data["name"] = name
    return JSONResponse(data)


async def api_state(request: Request):
    with SESSION.lock:
        return ok()  # ok() refreshes predictions when a rule exists


async def classes_search(request: Request):
    q = request.query_params.get("q", "")
    return JSONResponse({"results": data_store.search_classes(q)})


async def session_from_class(request: Request):
    body = await request.json()
    chebi_id = str(body.get("chebi_id", "")).strip()
    max_pos = _int(body.get("max_pos"), config.DEFAULT_MAX_POS)
    max_neg = _int(body.get("max_neg"), config.DEFAULT_MAX_NEG)
    if not chebi_id:
        return JSONResponse({"error": "No ChEBI id given."}, status_code=400)
    with SESSION.lock:
        pos_ids, neg_ids = data_store.gather_class_examples(chebi_id, max_pos, max_neg)
        added_pos = sum(1 for cid in pos_ids if SESSION.add_from_chebi_id(cid, "pos"))
        added_neg = sum(1 for cid in neg_ids if SESSION.add_from_chebi_id(cid, "neg"))
        return ok(message=f"Added {added_pos} positive and {added_neg} negative examples "
                          f"from {data_store.class_name(chebi_id)}.")


async def session_add(request: Request):
    body = await request.json()
    text = body.get("text", "")
    label = body.get("label", "pos")
    with SESSION.lock:
        added, errors = SESSION.add_from_text(text, label)
        msg = f"Added {len(added)} molecule(s)."
        if errors:
            msg += f" Could not parse: {', '.join(errors[:5])}"
        return ok(message=msg)


async def session_set_label(request: Request):
    body = await request.json()
    with SESSION.lock:
        SESSION.set_label(str(body.get("id")), body.get("label"))
        return ok()


async def session_remove(request: Request):
    body = await request.json()
    with SESSION.lock:
        SESSION.remove(str(body.get("id")))
        return ok()


async def session_reset(request: Request):
    with SESSION.lock:
        SESSION.clear()
        return ok(message="Session cleared.")


async def suggest(request: Request):
    body = await request.json()
    n = _int(body.get("n"), 10)
    with SESSION.lock:
        positives = SESSION.positives()
        if not positives:
            return JSONResponse({"error": "Add at least one positive example first."},
                                status_code=400)
        exclude_ids = {m.chebi_id for m in SESSION.molecules.values() if m.chebi_id}
        exclude_smiles = {m.smiles for m in SESSION.molecules.values()}
        candidates = similarity.suggest(positives, exclude_ids, exclude_smiles, n=n)
        # If a rule exists, attach its prediction for each candidate.
        if SESSION.current_rule and candidates:
            from chebi_utils.read_molecule import smiles_or_inchi_to_mol

            class _Tmp:
                def __init__(self, cid, mol):
                    self.id = cid
                    self.mol = mol

            tmp = []
            for c in candidates:
                mol = smiles_or_inchi_to_mol(c["smiles"])
                if mol is not None:
                    tmp.append(_Tmp(c["chebi_id"], mol))
            if tmp:
                preds = ilp.classify(SESSION.current_rule, tmp)
                for c in candidates:
                    c["prediction"] = preds.get(c["chebi_id"])
        return JSONResponse({"candidates": candidates})


async def suggest_accept(request: Request):
    body = await request.json()
    chebi_id = body.get("chebi_id")
    smiles = body.get("smiles")
    label = body.get("label", "pos")
    with SESSION.lock:
        if chebi_id:
            SESSION.add_from_chebi_id(str(chebi_id), label)
        elif smiles:
            SESSION.add_from_text(smiles, label)
        return ok()


# Live-learning state, updated by the background Popper thread and polled by the UI.
LEARN_LOCK = threading.Lock()
LEARN = {"running": False, "done": False, "error": None, "lines": [],
         "message": None, "score": None}


def _learn_worker(pos, neg, timeout):
    def on_line(line):
        with LEARN_LOCK:
            LEARN["lines"].append(line)

    try:
        result = ilp.learn_streaming(pos, neg, timeout, on_line)
    except Exception as e:
        traceback.print_exc()
        with LEARN_LOCK:
            LEARN.update(running=False, done=True, error=str(e))
        return

    rule = result.get("rule")
    with SESSION.lock:
        SESSION.learned_rule = rule
        SESSION.current_rule = rule
        if rule:
            try:
                recompute_predictions()
            except ilp.RuleError:
                pass
    with LEARN_LOCK:
        LEARN.update(
            running=False, done=True, score=result.get("score"),
            message=("Learned a rule." if rule else
                     result.get("error") or
                     "No rule found within the time limit "
                     "(try more/cleaner examples or a longer timeout)."),
        )


async def learn(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    timeout = _int(body.get("timeout"), config.DEFAULT_TIMEOUT)

    with LEARN_LOCK:
        if LEARN["running"]:
            return JSONResponse({"error": "Already learning."}, status_code=409)
        LEARN.update(running=True, done=False, error=None, lines=[], message=None, score=None)

    with SESSION.lock:
        pos = SESSION.positives()
        neg = [m for m in SESSION.molecules.values() if m.label == "neg"]

    if not pos or not neg:
        with LEARN_LOCK:
            LEARN.update(running=False, done=True,
                         error="Need at least one positive and one negative example.")
        return JSONResponse(
            {"error": "Need at least one positive and one negative example."}, status_code=400)

    threading.Thread(target=_learn_worker, args=(pos, neg, timeout), daemon=True).start()
    return JSONResponse({"started": True})


async def learn_progress(request: Request):
    """Poll Popper's live output; when done, includes the final session state."""
    with LEARN_LOCK:
        lines = list(LEARN["lines"])
        running, done, error = LEARN["running"], LEARN["done"], LEARN["error"]
        message, score = LEARN["message"], LEARN["score"]
    resp = {
        "running": running, "done": done, "error": error, "message": message,
        "score": score, "tail": lines[-6:], "log": "\n".join(lines),
        "total_lines": len(lines),
    }
    if done and not error:
        resp["state"] = state_dict()
    return JSONResponse(resp)


async def rule_edit(request: Request):
    body = await request.json()
    new_rule = (body.get("rule") or "").strip()
    with SESSION.lock:
        if not ilp.rule_targets_ok(new_rule):
            return JSONResponse(
                {"error": f"Rule must define the target predicate "
                          f"'{config.TARGET_LABEL}(...) :- ...'."}, status_code=400)
        old_preds = dict(SESSION.last_predictions)
        prev_rule = SESSION.current_rule
        SESSION.current_rule = new_rule
        try:
            new_preds = recompute_predictions()
        except ilp.RuleError as e:
            SESSION.current_rule = prev_rule
            return JSONResponse({"error": f"Invalid rule: {e}"}, status_code=400)
        changed = []
        for m in SESSION.molecules.values():
            o, n = old_preds.get(m.id), new_preds.get(m.id)
            if o is not None and o != n:
                changed.append({"id": m.id, "name": m.name, "label": m.label,
                                "from": o, "to": n})
        return ok(message=f"Rule updated. {len(changed)} classification(s) changed.",
                  changed=changed)


async def rule_reset(request: Request):
    with SESSION.lock:
        SESSION.current_rule = SESSION.learned_rule
        if SESSION.current_rule:
            try:
                recompute_predictions()
            except ilp.RuleError:
                pass
        return ok(message="Reverted to the learned rule.")


routes = [
    Route("/", index),
    Route("/api/depict", depict),
    Route("/api/explain", explain),
    Route("/api/state", api_state),
    Route("/api/classes/search", classes_search),
    Route("/api/session/from_class", session_from_class, methods=["POST"]),
    Route("/api/session/add", session_add, methods=["POST"]),
    Route("/api/session/set_label", session_set_label, methods=["POST"]),
    Route("/api/session/remove", session_remove, methods=["POST"]),
    Route("/api/session/reset", session_reset, methods=["POST"]),
    Route("/api/suggest", suggest, methods=["POST"]),
    Route("/api/suggest/accept", suggest_accept, methods=["POST"]),
    Route("/api/learn", learn, methods=["POST"]),
    Route("/api/learn/progress", learn_progress),
    Route("/api/rule/edit", rule_edit, methods=["POST"]),
    Route("/api/rule/reset", rule_reset, methods=["POST"]),
]

@contextlib.asynccontextmanager
async def _lifespan(app):
    missing = config.check_data_files()
    if missing:
        print(
            "\n[ICaRuS] WARNING — required data file(s) not found:\n"
            + "\n".join(missing)
            + "\n  Set ICARUS_CHEBILP_DIR to the chebILP checkout that holds them "
            "(see README → Running on another system).\n",
            flush=True,
        )
    else:
        print("[ICaRuS] data files located; ready.", flush=True)
    yield


app = Starlette(routes=routes, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
