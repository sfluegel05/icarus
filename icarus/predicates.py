"""The predicate catalog that drives the block editor: :mod:`viloch.editor.catalog`.

The static blocks (elements, hydrogens, charge, rings, stereo, bonds, molecule
properties), the blocks of the generated ``aux_*`` predicates of the session's rule
library, and the Extended Functional Groups all come from viloch; icarus only picks
which library predicates are relevant to the session (``app._class_library_names``).
"""

from viloch.editor.catalog import (  # noqa: F401
    aux_blocks,
    aux_index,
    catalog,
    efg_blocks,
    search_aux_blocks,
)
