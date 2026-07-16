# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parameter space for generated Snakemake sweep workflows.

A sweep is described by two files:

- ``sweep.csv`` — the flat combination table. One row per *case*; the case
  column holds the case name (the primary ``{case}`` wildcard) and every other
  column names a parameter *variant* for one dimension (a config, e.g.
  ``transport_properties_config``).
- ``params.yaml`` — named, reusable parameter variants per dimension::

      transport_properties_config:
        nu1e-05: {transportModel: Newtonian, nu: 1.0e-05}
        nu2e-05: {transportModel: Newtonian, nu: 2.0e-05}
      control_dict_config:
        short: {endTime: 1.0, deltaT: 0.001}
        long:  {endTime: 5.0, deltaT: 0.001}

:class:`YamlParamSpace` ties both together with an API compatible with
``snakemake.utils.Paramspace`` (``wildcard_pattern``, ``instance_patterns``,
``instance``) and can *materialize* the resolved configuration of each
``(case, rule)`` pair as ``configs/{case}/{rule}.json``. Materialized files are
only rewritten when their content changes, so downstream rules that declare
them as inputs re-run exactly when their parameters change.

Variants may be validated against pydantic models (one model per dimension).

This module is imported by the generated Snakefile at workflow *runtime*, so it
deliberately stays UI-free: stdlib + pydantic only, with PyYAML imported
lazily. It must never import trame or pybFoam.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from pydantic import BaseModel

__all__ = [
    "KeyedDim",
    "YamlParamSpace",
    "read_params_yaml",
    "read_sweep_csv",
    "write_if_changed",
    "write_params_yaml",
    "write_sweep_csv",
]

DEFAULT_CASE_COL = "case"


def _yaml() -> ModuleType:
    """Import PyYAML lazily so the module stays importable without it."""
    try:
        import yaml
    except ImportError as e:
        msg = "PyYAML is required for params.yaml support. Install it with: pip install pyyaml"
        raise ImportError(msg) from e
    return yaml


class _SupportsGet(Protocol):
    """Anything with a ``.get`` — plain dicts and Snakemake wildcards objects."""

    def get(self, key: str, default: Any | None = None) -> Any: ...


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def read_sweep_csv(
    path: str | Path, case_col: str = DEFAULT_CASE_COL
) -> list[dict[str, str]]:
    """Read the sweep combination table.

    Args:
        path: Path to the CSV file.
        case_col: Name of the case-name column.

    Returns:
        One dict per row (column -> cell), in file order.

    Raises:
        ValueError: If the case column is missing or case names repeat.
    """
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if case_col not in fieldnames:
            msg = f"sweep table {path} is missing the '{case_col}' column (found: {fieldnames})"
            raise ValueError(msg)
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]

    seen: set[str] = set()
    for row in rows:
        case = row[case_col]
        if case in seen:
            msg = f"sweep table {path}: duplicate case name '{case}'"
            raise ValueError(msg)
        seen.add(case)
    return rows


def write_sweep_csv(
    path: str | Path,
    rows: list[dict[str, str]],
    columns: list[str] | None = None,
    case_col: str = DEFAULT_CASE_COL,
) -> None:
    """Write the sweep combination table (deterministic column order).

    Args:
        path: Destination CSV path.
        rows: One dict per row (column -> cell).
        columns: Explicit column order; defaults to the case column followed by
            the remaining columns sorted by name.
        case_col: Name of the case-name column.
    """
    if columns is None:
        others = sorted({k for row in rows for k in row} - {case_col})
        columns = [case_col, *others]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_params_yaml(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the named parameter variants: ``{dimension: {variant_name: params}}``.

    Raises:
        ValueError: If the file is not a mapping of mappings.
    """
    data = _yaml().safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        msg = f"params file {path}: expected a mapping at the top level"
        raise ValueError(msg)
    for dim, variants in data.items():
        if not isinstance(variants, dict):
            msg = f"params file {path}: dimension '{dim}' must map variant names to param dicts"
            raise ValueError(msg)
    return data


def write_params_yaml(
    path: str | Path, variants: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Write the named parameter variants (deterministic key order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml().safe_dump(variants, sort_keys=True))


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` only if the content differs.

    This is the provenance primitive behind :meth:`YamlParamSpace.materialize`:
    unchanged configs keep their mtime, so Snakemake does not re-run their jobs.

    Returns:
        True if the file was (re)written.
    """
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


# ---------------------------------------------------------------------------
# YamlParamSpace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyedDim:
    """A resolved *keyed* dimension: rules run once per variant, not per case.

    Returned by :meth:`YamlParamSpace.keyed`. When the dimension is absent from
    the sweep, the axis is a single implicit variant (``space is None``) so the
    pipeline shape stays uniform — the keyed rule still runs exactly once.

    Attributes:
        dim: The dimension name (sweep column / params.yaml key).
        names: The variant names, or ``(default,)`` for the implicit axis.
        space: The owning space, or None for the implicit single-variant axis.
        default: The implicit variant name.
    """

    dim: str
    names: tuple[str, ...]
    space: YamlParamSpace | None
    default: str = "base"

    def of(self, wildcards: _SupportsGet) -> str:
        """The variant name a case routes to (``default`` on the implicit axis)."""
        if self.space is None:
            return self.default
        return self.space.variant_of(wildcards, self.dim)


class YamlParamSpace:
    """A parameter space backed by ``sweep.csv`` + ``params.yaml``.

    Mirrors the ``snakemake.utils.Paramspace`` surface (``wildcard_pattern``,
    ``instance_patterns``, ``instance``) while resolving name columns to nested
    YAML variants and optionally validating them with pydantic models.

    Args:
        sweep_csv: Path to the combination table.
        params_yaml: Path to the named parameter variants.
        models: Optional pydantic model per dimension; every referenced variant
            is validated against its dimension's model at construction time.
        case_col: Name of the case-name column / wildcard.

    Raises:
        ValueError: On an empty sweep, a sweep column with no matching
            dimension in the params file, an unknown variant name, or a variant
            failing pydantic validation.
    """

    def __init__(
        self,
        sweep_csv: str | Path,
        params_yaml: str | Path,
        models: Mapping[str, type[BaseModel]] | None = None,
        case_col: str = DEFAULT_CASE_COL,
    ) -> None:
        self.case_col = case_col
        self.rows = read_sweep_csv(sweep_csv, case_col)
        self.variants = read_params_yaml(params_yaml)
        self.models = dict(models or {})
        if not self.rows:
            msg = f"sweep table {sweep_csv} has no rows"
            raise ValueError(msg)
        self.dims: list[str] = [k for k in self.rows[0] if k != case_col]
        self._by_case = {row[case_col]: row for row in self.rows}
        self._validate()

    def _validate(self) -> None:
        for dim in self.dims:
            if dim not in self.variants:
                msg = (
                    f"sweep column '{dim}' has no matching top-level key in the "
                    f"params file (found: {sorted(self.variants)})"
                )
                raise ValueError(msg)
        for row in self.rows:
            case = row[self.case_col]
            for dim in self.dims:
                variant = row[dim]
                if variant not in self.variants[dim]:
                    msg = (
                        f"case '{case}': unknown variant '{variant}' for dimension "
                        f"'{dim}' (known: {sorted(self.variants[dim])})"
                    )
                    raise ValueError(msg)
        for dim, model in self.models.items():
            for variant, params in self.variants.get(dim, {}).items():
                try:
                    model.model_validate(params)
                except Exception as e:
                    msg = (
                        f"params variant '{dim}.{variant}' failed validation "
                        f"against {model.__name__}: {e}"
                    )
                    raise ValueError(msg) from e

    # -- snakemake.utils.Paramspace-compatible surface ----------------------

    @property
    def wildcard_pattern(self) -> str:
        """The wildcard pattern identifying one instance (``"{case}"``)."""
        return f"{{{self.case_col}}}"

    @property
    def instance_patterns(self) -> list[str]:
        """The wildcard pattern of each instance — here, the case names."""
        return list(self.cases)

    def instance(self, wildcards: _SupportsGet) -> dict[str, dict[str, Any]]:
        """The resolved parameters of one instance: ``{dim: variant params}``.

        Args:
            wildcards: A Snakemake wildcards object or a plain mapping holding
                the case column (both provide ``.get``).
        """
        case = wildcards.get(self.case_col)
        if case is None:
            msg = f"wildcards carry no '{self.case_col}' entry"
            raise KeyError(msg)
        return self.config_for(str(case))

    def variant_of(self, wildcards: _SupportsGet, dim: str) -> str:
        """The variant *name* a case selects for one dimension (e.g. ``"coarse"``).

        Used by generated input functions to route a per-case rule to the
        output of a per-variant rule::

            lambda wc: f"meshes/{space.variant_of(wc, 'mesh')}/..."
        """
        case = wildcards.get(self.case_col)
        if case is None:
            msg = f"wildcards carry no '{self.case_col}' entry"
            raise KeyError(msg)
        row = self._by_case.get(str(case))
        if row is None:
            msg = f"unknown case '{case}' (known: {self.cases})"
            raise KeyError(msg)
        if dim not in self.dims:
            msg = f"unknown dimension '{dim}' (sweep columns: {self.dims})"
            raise ValueError(msg)
        return row[dim]

    # -- extras --------------------------------------------------------------

    @property
    def cases(self) -> list[str]:
        """The case names, in sweep-table order."""
        return [row[self.case_col] for row in self.rows]

    def config_for(
        self, case: str, dims: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """The resolved parameter variants of one case, restricted to ``dims``."""
        row = self._by_case.get(case)
        if row is None:
            msg = f"unknown case '{case}' (known: {self.cases})"
            raise KeyError(msg)
        use = self.dims if dims is None else list(dims)
        for dim in use:
            if dim not in self.dims:
                msg = f"unknown dimension '{dim}' (sweep columns: {self.dims})"
                raise ValueError(msg)
        return {dim: self.variants[dim][row[dim]] for dim in use}

    def materialize(
        self,
        rule_dims: Mapping[str, Sequence[str]],
        out_dir: str | Path = "configs",
    ) -> list[Path]:
        """Write ``{out_dir}/{case}/{rule}.json`` for every (case, rule) pair.

        Each JSON holds only the rule's own dimensions, so editing a variant
        rewrites only the configs of rules that use that dimension — and via
        :func:`write_if_changed` only those whose content actually changed.

        Args:
            rule_dims: Dimensions used by each rule, e.g.
                ``{"setup": ["transport_properties_config"]}``.
            out_dir: Root directory for the materialized configs.

        Returns:
            The paths that were (re)written.
        """
        out_dir = Path(out_dir)
        changed: list[Path] = []
        for case in self.cases:
            for rule, dims in rule_dims.items():
                config = self.config_for(case, dims)
                text = json.dumps(config, sort_keys=True, indent=2) + "\n"
                path = out_dir / case / f"{rule}.json"
                if write_if_changed(path, text):
                    changed.append(path)
        return changed

    def materialize_dims(
        self,
        dims: Sequence[str],
        out_dir: str | Path = "configs",
    ) -> list[Path]:
        """Write ``{out_dir}/{dim}/{variant}.json`` for every variant of ``dims``.

        The per-variant counterpart of :meth:`materialize`, for rules that run
        once per named variant instead of once per case (keyed rules). Editing
        one variant rewrites exactly one JSON, so only that variant's rule
        instance (and its dependents) re-runs.

        Returns:
            The paths that were (re)written.
        """
        out_dir = Path(out_dir)
        changed: list[Path] = []
        for dim in dims:
            if dim not in self.variants:
                msg = f"unknown dimension '{dim}' (params.yaml sections: {sorted(self.variants)})"
                raise ValueError(msg)
            for name, params in sorted(self.variants[dim].items()):
                text = json.dumps(params, sort_keys=True, indent=2) + "\n"
                path = out_dir / dim / f"{name}.json"
                if write_if_changed(path, text):
                    changed.append(path)
        return changed

    def keyed(
        self,
        dim: str,
        out_dir: str | Path = "configs",
        default: str = "base",
    ) -> KeyedDim:
        """Resolve a keyed dimension, with ONE implicit variant when absent.

        Present in the sweep: materializes its variants
        (:meth:`materialize_dims`) and returns the real axis. Absent: writes
        ``{out_dir}/{dim}/{default}.json`` = ``{}`` and returns the implicit
        single-variant axis, so the keyed rule still runs exactly once and is
        shared by every case.
        """
        if dim in self.dims:
            self.materialize_dims([dim], out_dir=out_dir)
            return KeyedDim(
                dim=dim,
                names=tuple(sorted(self.variants[dim])),
                space=self,
                default=default,
            )
        write_if_changed(Path(out_dir) / dim / f"{default}.json", "{}\n")
        return KeyedDim(dim=dim, names=(default,), space=None, default=default)
