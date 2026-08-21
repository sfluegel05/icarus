"""Server-side 2D structure depiction (SVG) via RDKit."""

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

_EMPTY = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
    '<text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" '
    'font-family="sans-serif" font-size="12" fill="#9aa5b1">no structure</text></svg>'
)


def mol_to_svg(mol, w: int = 280, h: int = 220) -> str:
    """Render an RDKit mol to a clean, transparent-background SVG.

    Explicit hydrogens are dropped for legibility; falls back gracefully when a
    molecule cannot be sanitized/kekulized.
    """
    if mol is None:
        return _EMPTY.format(w=w, h=h)
    try:
        m = Chem.Mol(mol)
        try:
            m = Chem.RemoveHs(m)
        except Exception:
            pass
        try:
            rdDepictor.Compute2DCoords(m)
        except Exception:
            pass
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = d.drawOptions()
        opts.clearBackground = False
        opts.padding = 0.08
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(d, m)
        except Exception:
            d.DrawMolecule(m)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return _EMPTY.format(w=w, h=h)
