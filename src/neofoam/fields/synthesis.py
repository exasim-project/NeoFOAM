# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Auto-synthesize :class:`InitStep` objects from :class:`FieldDecl`s.

This module is the framework-side counterpart to the disk-schema
synthesizer in :mod:`neofoam.fields.schema`. Where ``schema_for``
synthesizes the per-field ``BaseConfig`` bound to ``0/<name>``, this
synthesizer produces the runtime :class:`InitStep` that calls the right
pybFoam ``read_field`` for the declared value type.

A model author writes a single ``Model.field(name, value_type=..., ...)``
declaration; the framework picks up that declaration in
``ModelRuntime.run_build`` and emits the equivalent ``InitStep`` — no
cross-link from ``@build`` required.

The pybFoam import is deferred to factory invocation time so importing
:mod:`neofoam.fields` does not pull in the OpenFOAM bindings.
"""

from __future__ import annotations

from typing import Any, Callable

from neofoam.fields.decl import FieldDecl
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.initialization.helpers import field as _field_step
from neofoam.framework.initialization.init_step import InitStep


def _resolve_read_field(value_type: type) -> Callable[[Any, str], Any]:
    """Return the pybFoam ``<Type>.read_field`` callable for ``value_type``.

    The lookup is deferred to call time so ``import neofoam.fields`` does
    not transitively import ``pybFoam`` — only models that actually
    materialise a runtime pay that import cost.
    """
    import pybFoam as pyf  # local import, see module docstring

    dispatch: dict[type, Callable[[Any, str], Any]] = {
        Scalar: pyf.volScalarField.read_field,
        Vector: pyf.volVectorField.read_field,
        # Tensor reserved; not exercised in Phase 1 / 2.
    }
    fn = dispatch.get(value_type)
    if fn is None:
        raise TypeError(
            f"auto-synthesize: no read_field dispatch for value_type "
            f"{value_type!r}. Add it to "
            "neofoam.fields.synthesis._resolve_read_field "
            "(and update neofoam.fields.schema._FOAM_CLASS in lock-step)."
        )
    return fn


def synthesize_init_step(decl: FieldDecl) -> InitStep:
    """Build the framework's default ``InitStep`` for one ``FieldDecl``.

    The returned step calls
    ``<value_type>.read_field(ctx["mesh"], decl.name)`` and forwards
    ``depends_on`` / ``write`` straight from the declaration. Equivalent
    to the now-removed ``FieldDecl.create(<read_fn>)`` adapter, but with
    the factory generated from the declared value type rather than
    supplied by each model.
    """

    def factory(ctx: dict[str, Any]) -> Any:
        read_field = _resolve_read_field(decl.value_type)
        return read_field(ctx["mesh"], decl.name)

    return _field_step(
        decl.name,
        factory,
        depends_on=list(decl.depends_on),
        write=decl.write,
    )
