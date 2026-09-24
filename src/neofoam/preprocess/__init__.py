# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pre-processing: initialise a case's ``0/`` fields region by region.

The Python replacement for ``setFieldsDict``: a case declares a default per
field plus a list of regions in ``system/setFields.yaml`` or
``system/setFields.py``, and the ``setFields`` preprocessing tool
(:mod:`neofoam.tools.set_fields`) writes the values into the fields before the
run. The regions are the *post-processing* selectors
(:class:`~neofoam.postprocess.nodes.selectors.Box`,
:class:`~neofoam.postprocess.nodes.selectors.Sphere`, ``&``/``|``/``~``) — one
region language for both ends of a case.

The declaration (:mod:`~neofoam.preprocess.config`) and the script API
(:mod:`~neofoam.preprocess.script`) are pure numpy; pybFoam appears only in
:mod:`~neofoam.preprocess.apply`, which is what makes this pybFoam-only — see
its docstring. The case loaders behind the front doors are the tool's, and are
imported from their own modules.
"""

from neofoam.preprocess.apply import apply_set_fields
from neofoam.preprocess.config import SetFieldsConfig, resolve_regions
from neofoam.preprocess.script import SetFields

__all__ = [
    # The two front doors
    "SetFields",
    "SetFieldsConfig",
    # Resolve and apply
    "resolve_regions",
    "apply_set_fields",
]
