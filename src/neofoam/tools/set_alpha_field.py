# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""setAlphaField tool: initialise volume fractions from a case's ``set_alpha`` function.

The script is loaded and the mesh resolved from the live ``ctx`` at call time, for the
same reason as in :mod:`neofoam.tools.set_fields`, whose restart guard this step shares.
"""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from neofoam.core.case_module import CaseModuleLoader
from neofoam.framework.initialization import InitStep, InitStepExecutionError
from neofoam.framework.tools import Tool
from neofoam.preprocess.apply import IsoVolumeFractions, apply_volume_fractions

from .registry import register_tool
from .set_fields import _is_restart

logger = logging.getLogger(__name__)

SCRIPT_FILE = "system/setAlphaField.py"

_LOADER = CaseModuleLoader(
    module_name="neofoam_case_set_alpha_field",
    families=("Node",),
    kind="setAlphaField",
)

SetAlpha = Callable[[Any], Mapping[str, IsoVolumeFractions]]


class SetAlphaFieldStep(BaseModel):
    """Initialise volume fractions from ``system/setAlphaField.py``."""

    tool: Literal["setAlphaField"]


setAlphaFieldTool = Tool("setAlphaField", consumes_mesh=True)


def _load_set_alpha(case_dir: Path) -> SetAlpha:
    """The case script's ``set_alpha`` function."""
    module = _LOADER.load(case_dir, SCRIPT_FILE)
    if module is None:
        raise ValueError(
            f"setAlphaField: {case_dir} has no {SCRIPT_FILE} — remove the tool from "
            f"system/preprocess.yaml or add the script"
        )
    set_alpha: Any = getattr(module, "set_alpha", None)
    if not callable(set_alpha):
        raise ValueError(f"setAlphaField: {SCRIPT_FILE} must define a function set_alpha(mesh)")
    return set_alpha  # type: ignore[no-any-return]


@setAlphaFieldTool.build
def _build_set_alpha_field(cfg: SetAlphaFieldStep) -> list[InitStep]:
    def run(ctx: dict[str, Any]) -> Any:
        mesh = ctx["_prev_mesh"]
        case_dir = Path(".")
        time_name = str(mesh.time().timeName())
        if _is_restart(case_dir, time_name):
            logger.warning(
                "preprocess.setAlphaField: skipped — the run starts at %s, which is not the "
                "case's earliest time directory, so this is a restart and the fields are "
                "left as they are",
                time_name,
            )
            return mesh
        set_alpha = _load_set_alpha(case_dir)
        try:
            apply_volume_fractions(mesh, set_alpha(mesh))
        except (ValueError, TypeError):
            raise
        except Exception as exc:  # binding-level failure
            raise InitStepExecutionError("preprocess.setAlphaField", [], exc) from exc
        return mesh

    return [InitStep(name="preprocess.setAlphaField", initializer=run)]


register_tool(setAlphaFieldTool)
