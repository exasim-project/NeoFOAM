#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fill this folder by prompting an LLM agent — nothing is mirrored from source.

    export ANTHROPIC_API_KEY=…
    python test/agent/hotRoom/run_fill.py

Three explicit steps, no ``fill_case`` wrapper:

1. wipe this directory (``*.py`` helpers excepted);
2. prompt an Anthropic agent with the upstream
   ``buoyantBoussinesqPimpleFoam/hotRoom`` dictionary text and let it
   return a populated ``CaseSpec``;
3. write every populated config through :func:`save_merged`, which groups
   contributors by ``io_config.file`` and merges them — Boussinesq and
   Transport both land in ``constant/transportProperties`` without
   clobbering each other.

No mesh, ``0/`` fields, or static assets are copied; the on-disk result
is exactly what the agent's saved configs produce. No exception handling
either; a crash is the result.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from neofoam.agent.case_fill import (
    build_case_agent,
    case_spec_to_configs,
)

from save_merged import save_merged

SOURCE = Path(
    "/home/henning/OpenFOAM/develop/openfoam/tutorials/heatTransfer/"
    "buoyantBoussinesqPimpleFoam/hotRoom"
)
TARGET = Path(__file__).resolve().parent


def _wipe(target: Path) -> None:
    """Empty the target dir, but keep helper scripts (``*.py``) alongside."""
    for child in target.iterdir():
        if child.is_file() and child.suffix == ".py":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


if __name__ == "__main__":
    print(f"source (prompt only, not copied): {SOURCE}")
    print(f"target: {TARGET}")
    _wipe(TARGET)

    # 1) Tell the agent to fill the configs.
    agent = build_case_agent(model_name="claude-haiku-4-5")
    prompt = "fill out the config files that match the schema and dont select the Boussinesq model"
    # prompt = _agent_prompt_from_source(SOURCE, incompressibleFluid)
    print("=== prompt ===")
    print(prompt)
    print("\n=== agent response ===")
    result = agent.run_sync(prompt)
    case_spec = result.output

    # 2) Write the configs to disc — grouped + merged per target file so
    # multi-owner files (transportProperties, fvSchemes, fvSolution) keep
    # every contribution. Per-instance ``cfg.save()`` would lose all but
    # the last writer (OpenFOAMStrategy.write clears before emitting).
    configs = case_spec_to_configs(case_spec)
    report = save_merged(configs, case_dir=TARGET)
    for file, contribs in report.items():
        print(f"WRITE {file} ← {', '.join(contribs)}")

    print("\n=== spec ===")
    print(case_spec.model_dump_json(indent=2, exclude_none=True))
