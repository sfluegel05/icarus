"""The rule editor and explanation endpoints, on a small in-memory session.

Needs no ChEBI data: molecules are added from SMILES, rules carry their own auxiliary
definitions (the rule library under data/ is optional)."""

from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from icarus.app import app
from icarus.session import SESSIONS

AUX_RULE = """concept(M) :- has_atom(M,A), n(A), aux_charged(A).
% PREDICATE_NAME: aux_charged
% DESCRIPTION: the atom carries a formal charge
aux_charged(A) :- charge_p(A).
aux_charged(A) :- charge_n(A)."""


@pytest.fixture
def session():
    sid = "test-" + uuid.uuid4().hex[:12]
    return sid, SESSIONS.get(sid)


@pytest.fixture
def client(session):
    sid, _ = session
    return TestClient(app, headers={"X-Session-Id": sid})


def test_page_loads_the_viloch_editor(client):
    html = client.get("/").text
    assert 'src="viloch/blocks.js"' in html and 'href="viloch/blocks.css"' in html
    assert "getModel" in client.get("/viloch/blocks.js").text
    assert client.get("/viloch/blocks.css").status_code == 200


def test_compile_model(client):
    model = {"clauses": [{"atoms": [{"id": "a1", "chips": [{"pred": "o"}], "x": 0, "y": 0}],
                          "molChips": [{"pred": "aromatic"}], "bonds": [], "groups": []}]}
    res = client.post("/api/rule/compile", json={"model": model})
    assert res.json() == {"rule": "concept(V0) :- has_atom(V0,V1), o(V1), aromatic(V0)."}


@pytest.mark.parametrize("body", [{"model": "x"}, {"model": {"clauses": [1]}},
                                  {"model": {"clauses": [{"atoms": [{}]}]}}])
def test_compile_rejects_bad_models(client, body):
    assert client.post("/api/rule/compile", json=body).status_code == 400


def test_unfold_program_returns_model(client):
    data = client.post("/api/rule/unfold", json={"rule": AUX_RULE}).json()
    assert len(data["model"]["clauses"]) == 2  # one per clause of aux_charged
    assert all("rawClause" not in c for c in data["model"]["clauses"])


def test_unfold_one_clause_model(client):
    clause = client.post("/api/rule/unfold", json={"rule": "concept(M) :- has_atom(M,A), n(A), aux_charged(A)."}).json()
    # aux_charged is not defined there (no library in tests): the chip stays
    aux_text = AUX_RULE.split("\n", 1)[1]
    model = {"clauses": [clause["model"]["clauses"][0]]}
    data = client.post("/api/rule/unfold",
                       json={"rule": aux_text, "model": model, "pred": "aux_charged"}).json()
    assert "error" not in data
    assert len(data["model"]["clauses"]) == 2


def test_unfold_rejects_intervals(client):
    res = client.post("/api/rule/unfold", json={"rule": "concept(M) :- X = 1..9, has_atom(M,X)."})
    assert res.status_code == 400 and "Intervals" in res.json()["error"]


def test_rule_editor_of_session_rule(client, session):
    _, s = session
    assert client.get("/api/rule/editor").json()["model"] is None
    s.add_rule("edited", AUX_RULE)
    data = client.get("/api/rule/editor").json()
    assert data["source"] == AUX_RULE
    assert len(data["model"]["clauses"]) == 2


def test_apply_edit_and_explain(client, session):
    _, s = session
    s.add_from_text("C[NH3+]\nCCO", "pos")
    res = client.post("/api/rule/edit", json={"rule": AUX_RULE})
    assert res.status_code == 200, res.text
    expl = client.get("/api/rule/explain").json()
    assert expl["image"].startswith("data:image/png;base64,") and expl["text"]
    mol = next(m for m in s.molecules.values() if m.smiles == "C[NH3+]")
    local = client.get("/api/explain", params={"id": mol.id}).json()
    assert local["fires"] is True
