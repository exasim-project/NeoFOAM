# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Choose the VTKHDF writer by name: :func:`vtkhdf_writer`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

from neofoam.postprocess.writers.vtkhdf import MeshExport, VtkHdfWriter
from neofoam.postprocess.writers.vtkhdf_files import Group, VtkHdfFilesWriter

WriterKind = Literal["parallel", "files"]


def vtkhdf_writer(
    path: Path,
    mesh: MeshExport,
    comm: Any = None,
    kind: WriterKind = "parallel",
    group: Group = "node",
) -> Union[VtkHdfWriter, VtkHdfFilesWriter]:
    """``"parallel"``: :class:`VtkHdfWriter`, one file (``group`` unused); ``"files"``:
    :class:`VtkHdfFilesWriter`, a file per ``group`` and the index ``path``."""
    if kind == "parallel":
        return VtkHdfWriter(path, mesh, comm)
    if kind == "files":
        return VtkHdfFilesWriter(path, mesh, comm, group=group)
    raise ValueError(f"VTKHDF writer kind must be 'parallel' or 'files', not {kind!r}")
