# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-style patch BCs → ``neon.blockamr`` ``VectorBC``.

The block-structured domain is a Cartesian box whose six faces are keyed
``xlo/xhi/ylo/yhi/zlo/zhi`` (AMReX convention). A patch spec is a small dict
``{"type": ..., "value": [...]}`` mirroring an OpenFOAM ``boundaryField`` entry:

* ``fixedValue``              → :func:`neon.blockamr.bc.fixedValue` (Dirichlet)
* ``noSlip``                  → :func:`neon.blockamr.bc.noSlip`
* ``zeroGradient`` / ``Neumann`` → :class:`neon.blockamr.bc.NeumannBC`

``slip`` / ``symmetry`` / ``symmetryPlane`` have **no** counterpart in the
vendored engine (only Dirichlet + Neumann ghost fills exist), so they raise
:class:`NotImplementedError` — a genuine gap flagged for a later spec. Periodic
faces need no entry: the engine skips them from ``geom.is_periodic()``.
"""

from typing import Any, Mapping

_FACES = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")


def map_patch(spec: Mapping[str, Any]) -> Any:
    """Map one OpenFOAM-style patch spec to a ``neon.blockamr`` face BC object."""
    from neon.blockamr.bc import NeumannBC, fixedValue, noSlip

    bc_type = spec.get("type")
    if bc_type == "fixedValue":
        value = spec.get("value")
        if value is None or len(value) != 3:
            raise ValueError(
                f"fixedValue patch needs a 3-vector 'value'; got {value!r}"
            )
        return fixedValue([float(v) for v in value])
    if bc_type == "noSlip":
        return noSlip()
    if bc_type in ("zeroGradient", "Neumann"):
        return NeumannBC()
    if bc_type in ("slip", "symmetry", "symmetryPlane"):
        raise NotImplementedError(
            f"BC type {bc_type!r} is not supported by the neon.blockamr engine "
            "(only Dirichlet fixedValue/noSlip and Neumann zeroGradient exist). "
            "Slip/symmetry ghost fills are deferred to a later spec."
        )
    raise ValueError(f"unknown/unsupported patch BC type {bc_type!r}")


def build_vector_bc(patches: Mapping[str, Mapping[str, Any]]) -> Any:
    """Build a ``neon.blockamr.VectorBC`` from a ``{face: patch-spec}`` mapping.

    Faces are keyed ``xlo/xhi/ylo/yhi/zlo/zhi``; any omitted face defaults to
    ``noSlip`` inside ``VectorBC`` (and periodic faces are skipped by the engine).
    """
    from neon.blockamr.bc import VectorBC

    unknown = set(patches) - set(_FACES)
    if unknown:
        raise ValueError(
            f"unknown boundary face(s) {sorted(unknown)}; expected {list(_FACES)}"
        )

    faces = {face: map_patch(spec) for face, spec in patches.items()}
    return VectorBC(**faces)
