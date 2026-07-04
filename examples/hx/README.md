<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# HX example — from surfaces to a solved case

The **starting point is only the geometry surfaces**: the five STL patches under
`constant/triSurface/` (a quasi-2D tube-bank heat exchanger), plus `manifest.json`
naming each surface and its role (inlet / outlet / walls / tubes / front-back slab).

Nothing else is authored by hand. `run.sh` drives the rest of the pipeline:

```
constant/triSurface/*.stl        (this example — the only input)
        │
        ▼  manifest.json  ──►  blockMeshDict + snappyHexMeshDict   (deterministic)
        │
        ▼  AI prompt  ──►  0/U, 0/p, transport/turbulence, fvSchemes,
        │                  fvSolution, controlDict   (LLM via the neofoam MCP)
        ▼
   blockMesh → snappyHexMesh → pimpleFoam solve   (one `./Allrun`)
```

The physics is written by an LLM that is handed the **neofoam MCP toolset** and a
natural-language prompt (from `test/e2e/cases/prompts.json`). The model discovers the
patches (`case_patches`), the config shapes (`list_configs` / `config_schema`) and
writes the case (`save_case`); a `validate_case` → fix loop runs until it is clean.
The geometry is a given — the prompt tells the model **not** to author the mesh dicts.

## Run it

Prerequisites: `ANTHROPIC_API_KEY` set, and a sourced OpenFOAM (for the solve).

```bash
examples/hx/run.sh                 # laminar tube_bank (physics fully specified)
examples/hx/run.sh heat_exchanger  # autonomous: the LLM chooses the physics
```

`run.sh` stages the mesh dicts from this example's own surfaces, lets the AI author
the physics, copies in `Allrun`, then meshes and solves — writing time directories
into this folder. The `tube_bank` prompt is laminar and solves end-to-end; the
`heat_exchanger` prompt is autonomous (water, the LLM decides turbulence / heat
transfer) and is the harder, exploratory case.

## Notes

- The AI-authored files (`0/`, `system/*`, `constant/{transport,turbulence}Properties`,
  the mesh dicts) and the solved time directories are **generated** by `run.sh` — only
  the surfaces + manifest are the checked-in starting point.
- The driver lives in `test/e2e/cases/fill_tube_bank_mcp.py`; the prompts in
  `test/e2e/cases/prompts.json`.
