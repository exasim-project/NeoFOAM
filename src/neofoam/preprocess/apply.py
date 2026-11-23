# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The write-back: read the case's fields, assign the declared values, write them.

The only module of the package that touches pybFoam, and the reason
``setFields`` is **pybFoam-only**: a pybFoam volume field hands its cell values
over as a writable, zero-copy numpy view of the OpenFOAM memory
(``np.asarray(field.internalField())``), so assigning through the view *is*
setting the field. A NeoN field keeps its values on an executor that may be a
device and offers no such view, so a NeoN case has no write path here.

The field type follows the value: a scalar sets a ``volScalarField``, a 3-vector
a ``volVectorField``, and the on-disk class is checked against it *before* the
read — OpenFOAM answers a wrong-class read with a fatal error that takes the
process down, not an exception.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.postprocess.nodes.selectors import Selector
from neofoam.postprocess.sources.geometry import CellGeometry
from neofoam.preprocess.config import RegionValue

#: Number of components a value has -> the OpenFOAM class that holds it.
_FIELD_CLASS = {1: "volScalarField", 3: "volVectorField"}

#: The ``class`` entry of a FoamFile header.
_CLASS_ENTRY = re.compile(r"\bclass\s+(\w+)\s*;")

#: OpenFOAM class -> the pybFoam class that reads it.
_READERS: dict[str, Any] = {
    "volScalarField": pyf.volScalarField,
    "volVectorField": pyf.volVectorField,
}

#: A region as :func:`~neofoam.preprocess.config.resolve_regions` hands it over.
Region = tuple[Selector, Mapping[str, RegionValue]]


def apply_set_fields(
    mesh: Any,
    defaults: Mapping[str, RegionValue],
    regions: Sequence[Region],
) -> list[str]:
    """Write every field named in ``defaults`` or ``regions`` and return their names.

    Each field is read from the mesh's current time directory, filled with its
    default (when it has one), overwritten in every region that names it — in
    order, so a later region wins — and written back. Paths are relative to the
    working directory, like every preprocessing tool's dict file, so the caller
    runs from the case::

        apply_set_fields(mesh, {"alpha.water": 0.0}, resolve_regions(config))
    """
    positions = CellGeometry(mesh).positions()
    # a selector answers with the kernels' 0/1 labels; indexing wants booleans.
    masks = [
        (np.asarray(selector.select(positions)).astype(bool), values)
        for selector, values in regions
    ]

    written: list[str] = []
    for name in _field_names(defaults, regions):
        field = _read_field(mesh, name, _components(name, defaults, regions))
        values = np.asarray(field.internalField())
        if name in defaults:
            values[:] = defaults[name]
        for mask, region_values in masks:
            if name in region_values:
                values[mask] = region_values[name]
        field.correctBoundaryConditions()
        pyf.write(field)
        written.append(name)
    return written


@dataclass(frozen=True)
class IsoVolumeFractions:
    """The volume fractions of one iso-surface cut, as ``setAlphaField`` writes them.

    ``cells`` holds one value in [0, 1] per cell; ``patches`` holds the wetted
    area fraction of every patch face, by patch name. Build one with
    :func:`iso_volume_fractions` and return it from a case's ``set_alpha``::

        return {"alpha.water": iso_volume_fractions(mesh, level_set)}
    """

    cells: npt.NDArray[np.float64]
    patches: dict[str, npt.NDArray[np.float64]]


def iso_volume_fractions(
    mesh: Any, point_values: npt.ArrayLike, iso_value: float = 0.0
) -> IsoVolumeFractions:
    """Cut every cell and boundary face at the iso-surface of a point field.

    The region where ``point_values > iso_value`` is the phase, so a level set is
    positive inside it. Use this in ``system/setAlphaField.py`` for a sharp
    interface; use :class:`SetFields` for a region of constant values. Reads the
    mesh only and writes nothing::

        points = np.asarray(mesh.points())
        level_set = 0.15 - np.linalg.norm(points[:, :2] - (0.3, 0.35), axis=1)
        fractions = iso_volume_fractions(mesh, level_set)
    """
    values = np.asarray(point_values, dtype=np.float64)
    cells, patches = nfb.iso_volume_fractions(mesh, values, iso_value)
    return IsoVolumeFractions(cells=cells, patches=patches)


def apply_volume_fractions(mesh: Any, fields: Mapping[str, IsoVolumeFractions]) -> list[str]:
    """Write each field's cell and patch fractions to the mesh's current time directory.

    The field must already exist as a ``volScalarField``; its boundary condition
    types are kept. Returns the names written::

        apply_volume_fractions(mesh, {"alpha.water": fractions})
    """
    for name, fractions in fields.items():
        field = _read_field(mesh, name, components=1)
        nfb.write_volume_fractions(field, fractions.cells, fractions.patches)
    return list(fields)


def _field_names(defaults: Mapping[str, RegionValue], regions: Sequence[Region]) -> list[str]:
    """Every field named anywhere, in declaration order."""
    names = list(defaults)
    for _, values in regions:
        names.extend(name for name in values if name not in names)
    return names


def _components(name: str, defaults: Mapping[str, RegionValue], regions: Sequence[Region]) -> int:
    """How many components every value of ``name`` has; a disagreement is an error."""
    assigned = [defaults[name]] if name in defaults else []
    assigned.extend(values[name] for _, values in regions if name in values)
    # np.size, not len: a script may compute a value with numpy, where a scalar
    # has no len() and a 3-vector may be an array rather than a tuple.
    sizes = {int(np.size(value)) for value in assigned}
    if len(sizes) != 1:
        raise ValueError(
            f"setFields: field {name!r} is set to values of {sorted(sizes)} components — "
            f"a field is either a scalar or a 3-vector, never both"
        )
    components = sizes.pop()
    if components not in _FIELD_CLASS:
        raise ValueError(
            f"setFields: field {name!r} is set to a value of {components} components — "
            f"a field is either a scalar or a 3-vector"
        )
    return components


def _read_field(mesh: Any, name: str, components: int) -> Any:
    """The registered field ``name``, read as the class its declared values imply."""
    expected = _FIELD_CLASS[components]
    path = Path(str(mesh.time().timeName())) / name
    if not path.is_file():
        raise ValueError(f"setFields: field {name!r} is not in {path.parent}/")
    on_disk = _class_on_disk(path)
    if on_disk != expected:
        kind = "scalar" if components == 1 else "3-vector"
        raise ValueError(
            f"setFields: field {name!r} is a {on_disk} on disk but is set to a {kind} "
            f"value, which needs a {expected}"
        )
    # Unregistered: setFields owns the field for the length of this call, and
    # registering it would collide with the solver's own read of the same name.
    return _READERS[expected].read_field(mesh, name, False)


def _class_on_disk(path: Path) -> str:
    """The ``class`` of a field file, read from its header without OpenFOAM.

    The header is ASCII even in a ``writeFormat binary`` file, so it is parsed
    here rather than handed to ``pyf.dictionary.read``: that reads the *whole*
    file and answers a binary body with the fatal error this pre-check exists to
    prevent (and tokenises a multi-million-cell list twice on an ASCII one).
    """
    head = path.open("rb").read(1024).decode("ascii", errors="replace").split("}", 1)[0]
    match = _CLASS_ENTRY.search(head)
    if match is None:
        raise ValueError(f"setFields: {path} has no FoamFile header — it is not a field file")
    return match.group(1)
