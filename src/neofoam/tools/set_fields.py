# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""setFields tool: initialise the case's ``0/`` fields from its setFields declaration.

The ``@build`` initializer resolves the prior mesh from the live ``ctx`` at call time
(never captured in a long-lived closure — that would create a mesh-bound reference cycle
that segfaults across in-process runs), and loads the case's declaration there too, so a
pipeline is built without reading the case. ``set_fields_for_case``/``apply_set_fields``
are module globals so tests can monkeypatch them without building a mesh. The case is
read relative to the working directory, like every tool's dict file — ``neofoam
preprocess`` runs the DAG from inside the case. setFields changes no topology; it passes
the prior mesh straight through so the pipeline treats it like any mesh-advancing step,
which also lets it be the pipeline's sink.

**Never on a restart.** The tool reads and writes the time directory the run *starts
from*, so on a restart it would overwrite the very fields the run is restarting from —
and the ``incompressibleFluid`` solver resolves the same ``system/preprocess.yaml``
again at start, not only ``neofoam preprocess``. :func:`_is_restart` is the guard: a
case that holds a time directory older than the one the run starts from is continuing,
so the step logs one line and passes the mesh through untouched. Neither ``Time.timeIndex()`` nor
``Time.startTimeIndex()`` can be used for this — both read 0 at a restart too, since
``startTimeIndex_`` only moves when a previous run left a ``uniform/time`` dictionary.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from neofoam.framework.initialization import InitStep, InitStepExecutionError
from neofoam.framework.tools import Tool
from neofoam.preprocess.apply import apply_set_fields
from neofoam.preprocess.script import set_fields_for_case

from .registry import register_tool

logger = logging.getLogger(__name__)


class SetFieldsStep(BaseModel):
    """Initialise the case's fields from ``system/setFields.yaml`` / ``setFields.py``."""

    tool: Literal["setFields"]


setFieldsTool = Tool("setFields", consumes_mesh=True)


def _is_restart(case_dir: Path, start: str) -> bool:
    """Whether the case holds a time directory older than the one the run starts from.

    ``start`` is the run's time *name*, the string OpenFOAM itself resolved the
    start time to and reads and writes under — comparing ``Time.value()`` against
    directory names calls a ``startTime 0.10000001`` beside a ``0.1/`` written at
    ``timePrecision 6`` a restart, when the two are the same directory.
    """
    start_time = float(start)
    for entry in case_dir.iterdir():
        if not entry.is_dir() or entry.name == start:
            continue
        try:
            written = float(entry.name)
        except ValueError:  # constant, system, 0.orig, processor* …
            continue
        if written < start_time:
            return True
    return False


@setFieldsTool.build
def _build_set_fields(cfg: SetFieldsStep) -> list[InitStep]:
    def run(ctx: dict[str, Any]) -> Any:
        mesh = ctx["_prev_mesh"]
        case_dir = Path(".")
        time = mesh.time()
        if _is_restart(case_dir, str(time.timeName())):
            logger.warning(
                "preprocess.setFields: skipped — the run starts at %s, which is not the "
                "case's earliest time directory, so this is a restart and the fields are "
                "left as they are",
                time.timeName(),
            )
            return mesh
        setup = set_fields_for_case(case_dir)
        try:
            apply_set_fields(mesh, setup.defaults, setup.regions)
        except (ValueError, TypeError):
            raise
        except Exception as exc:  # binding-level failure
            raise InitStepExecutionError("preprocess.setFields", [], exc) from exc
        return mesh

    return [InitStep(name="preprocess.setFields", initializer=run)]


register_tool(setFieldsTool)
