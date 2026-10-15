"""Quick end-to-end check of the ICaRuS ILP pipeline (no web server, no LLM).

Run from this repo with the app's interpreter:
    PYTHONPATH=. python smoke_test.py
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

    print("Learning with Popper ...")
    result = ilp.learn_streaming(pos, neg, timeout=20, on_line=lambda line: None)
    print("rule:", repr(result["rule"]))
    print("score:", result["score"])
    if not result["rule"]:
        raise SystemExit("FAIL: no rule learned")

    preds = ilp.classify(result["rule"], list(s.molecules.values()))
    for m in s.molecules.values():
        print(f"  {m.label:4s} {m.smiles:10s} -> {'POS' if preds[m.id] else 'neg'}")
    wrong = sum(preds[m.id] != (m.label == "pos") for m in s.molecules.values())
    print("OK" if wrong == 0 else f"WARN: {wrong} misclassified")


if __name__ == "__main__":
    main()
