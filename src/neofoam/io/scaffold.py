# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Write a set of configs back to a case directory.

:func:`save_configs` takes loaded (or freshly constructed) ``BaseConfig``
instances and writes each to its registered IO path under ``case_dir``.
Together with ``LoadResult.config_classes`` this is the basis for
scaffolding a case from config schemas alone — fill the configs, save
them, and a runnable case directory appears on disk.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Union

from neofoam.io.base import BaseConfig


def collect_config_classes(sources: Iterable[Any]) -> list[type]:
    """Collect the distinct ``BaseConfig`` subclasses from ``sources``.

    Each source may be a ``BaseConfig`` **subclass** (taken as-is), a
    ``BaseConfig`` **instance** (its class), or a model **spec / runtime**
    exposing ``_config_classes`` (declared classes) and/or ``configs``
    (loaded instances). Specs are duck-typed, so this works without
    importing the framework. Results are deduped by identity and
    order-preserved.

    This is the schema set an agent fills and :func:`save_configs` writes
    to scaffold a case — it surfaces classes a model declares even when
    their instances are never loaded.
    """
    seen: list[type] = []

    def _add(cls: Any) -> None:
        if isinstance(cls, type) and issubclass(cls, BaseConfig) and cls not in seen:
            seen.append(cls)

    for source in sources:
        if isinstance(source, type):
            _add(source)
            continue
        if isinstance(source, BaseConfig):
            _add(type(source))
            continue
        # A model spec carries ``_config_classes`` directly; a runtime
        # carries it on ``.spec`` and loaded instances on ``.configs``.
        spec = getattr(source, "spec", source)
        for declared in getattr(spec, "_config_classes", []) or []:
            _add(declared)
        for instance in getattr(source, "configs", []):
            _add(type(instance))
    return seen


def save_configs(
    configs: Iterable[BaseConfig], *, case_dir: Union[Path, str]
) -> list[Path]:
    """Save each config to ``case_dir`` via its registered IO strategy.

    Configs whose class has no ``@IOStrategy`` binding are skipped with a
    warning rather than raising, so a heterogeneous collection (e.g. one
    that includes an unbound config) still writes what it can.

    Args:
        configs: ``BaseConfig`` instances to write.
        case_dir: Target case directory (each config picks its own path
            relative to it via ``get_default_path``).

    Returns:
        The paths actually written, in input order.
    """
    written: list[Path] = []
    for cfg in configs:
        if type(cfg).io_config is None:
            warnings.warn(
                f"Skipping {type(cfg).__name__}: no IO strategy registered",
                stacklevel=2,
            )
            continue
        cfg.save(case_dir=case_dir)
        written.append(type(cfg).get_default_path(case_dir))
    return written
