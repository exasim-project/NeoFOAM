# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling's view of an OpenFOAM region: its mesh geometry and its fields, through pybFoam."""

from typing import Any, Mapping

import numpy as np

from neofoam.framework.coupling import Patch


class FoamGeometry:
    """The :class:`~neofoam.framework.coupling.MeshGeometry` of an OpenFOAM ``fvMesh``.

    Example::

        ctx.models["coupling"] = Participant(coupling, FoamGeometry(mesh), FoamFields(ctx.fields))
    """

    def __init__(self, mesh: Any) -> None:
        self.mesh = mesh

    def face_centres(self, patch: str) -> Any:
        return np.array(np.asarray(self.mesh.Cf()[patch]))

    def face_areas(self, patch: str) -> Any:
        return np.array(np.asarray(self.mesh.magSf()[patch]))

    def cell_centres(self) -> Any:
        return np.array(np.asarray(self.mesh.C().internalField()))

    def cell_volumes(self) -> Any:
        return np.array(np.asarray(self.mesh.V()))


class FoamFields:
    """The :class:`~neofoam.framework.coupling.FieldAccess` of OpenFOAM volume fields.

    Reads copy; writes go through the numpy view of the patch or the cells, since
    OpenFOAM disables ``operator=`` of a ``fixedValue`` patch.
    """

    def __init__(self, fields: Mapping[str, Any]) -> None:
        self.fields = fields

    def _values(self, field: str, location: Any) -> Any:
        if isinstance(location, Patch):
            return np.asarray(self.fields[field][location.name])
        return np.asarray(self.fields[field].internalField())

    def read(self, field: str, location: Any) -> Any:
        return np.array(self._values(field, location))

    def write(self, field: str, location: Any, values: Any) -> None:
        self._values(field, location)[:] = np.asarray(values, dtype=float)
