"""Extended Functional Groups (EFG) as library predicates: :mod:`viloch.chem.efg`.

viloch packages the EFG tables and points chebILP's ``efg`` seed source at them
(:func:`install`); icarus uses its matching for classification facts
(:func:`rule_facts`), retrieval (:func:`retrieval_entries`) and the palette.
"""

from viloch.chem.efg import (  # noqa: F401
    fact_lines,
    install,
    is_efg,
    referenced,
    retrieval_entries,
    rule_facts,
)
