"""Quick end-to-end check of the ICaRuS ILP pipeline (no web server).

Run from this repo under its WSL venv:
    PYTHONPATH=. .wslvenv/bin/python smoke_test.py
"""

from icarus.session import Session
from icarus import ilp


def main():
    s = Session()
    # Positives: simple alcohols (contain O). Negatives: pure hydrocarbons.
    for smi in ["CCO", "CCCO", "CC(O)C", "OCCO"]:
        s.add_from_text(smi, "pos")
    for smi in ["CC", "CCC", "CCCC", "C1CCCCC1"]:
        s.add_from_text(smi, "neg")

    pos = s.positives()
    neg = [m for m in s.molecules.values() if m.label == "neg"]
    print(f"pos={len(pos)} neg={len(neg)}")

    print("Learning ...")
    result = ilp.learn(pos, neg, timeout=20)
    print("rule:", repr(result["rule"]))
    print("score:", result["score"])

    if result["rule"]:
        preds = ilp.classify(result["rule"], list(s.molecules.values()))
        for m in s.molecules.values():
            print(f"  {m.label:4s} {m.smiles:10s} -> {'POS' if preds[m.id] else 'neg'}")


if __name__ == "__main__":
    main()
