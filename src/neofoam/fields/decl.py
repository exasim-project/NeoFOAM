# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field declarations attached to :class:`~neofoam.framework.model.ModelSpec`.

A :class:`FieldDecl` is the immutable record :meth:`ModelSpec.field`
returns. It carries the registration-time facts about an on-disk field
— ``name``, ``dimensions``, ``value_type``, ``allowed_bcs``, the
``write`` flag, and ``depends_on`` for the runtime init graph — and
nothing else: instantiating the field at runtime is the framework's
responsibility. :func:`~neofoam.fields.synthesis.synthesize_init_step`
emits the matching :class:`~neofoam.framework.initialization.InitStep`
that calls ``<value_type>.read_field(mesh, name)`` and applies the
declared ``depends_on`` / ``write``. The model author writes the
declaration once and never repeats it inside ``@build``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Optional, Union

from pydantic import BaseModel


@dataclass(frozen=True)
class FieldDecl:
    """Immutable on-disk-field declaration registered via ``Model.field(...)``.

    Attributes:
        name: Field name (e.g. ``"U"``, ``"p"``). Used both for the
            runtime registry key (``"fields.<name>"``) and the on-disk
            file path (``"0/<name>"``).
        dimensions: OpenFOAM dimension exponents
            ``[M, L, T, Θ, N, I, J]``; checked verbatim on disk and
            written back unchanged.
        value_type: :class:`~neofoam.fields.value_types.Scalar` /
            :class:`~neofoam.fields.value_types.Vector` marker.
        allowed_bcs: BC arms permitted on this field's
            ``boundaryField``. Build the discriminated union via
            :func:`~neofoam.fields.bc.build_bc_union`. Include
            :class:`~neofoam.fields.bc.GenericBC` to keep unknown
            arms parsing as the catch-all.
        write: ``True`` if the field should be auto-persisted by the
            runtime (forwarded to the underlying ``InitStep``).
        depends_on: Runtime dependencies for the field's factory.
            Defaults to ``("mesh",)``; a field that reads another
            field declares the extra edge explicitly
            (e.g. ``phi`` depends on ``("fields.U",)``).
        initial_value: Optional default ``internalField`` baked into
            scaffolded cases. ``None`` means "no default" — the loader
            won't synthesise one. The synthesised schema still accepts
            any value supplied by the case author / agent.
    """

    name: str
    dimensions: list[int]
    value_type: type
    allowed_bcs: tuple[type[BaseModel], ...]
    write: bool = False
    depends_on: tuple[str, ...] = ("mesh",)
    initial_value: Optional[Union[float, tuple[float, float, float], str]] = None
    # Kept private and out of the comparison/hash so model authors who
    # cache a ``FieldDecl`` by identity see the same object whether or
    # not a schema has been materialised.
    _schema_cache: dict[str, type] = _dc_field(
        default_factory=dict, compare=False, hash=False, repr=False
    )
