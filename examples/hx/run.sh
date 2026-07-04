#!/usr/bin/env bash
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Drive this example end-to-end FROM ONLY THE SURFACES in constant/triSurface:
#   surfaces -> blockMesh/snappyHexMesh dicts (from manifest) -> AI-authored
#   physics (via the neofoam MCP) -> mesh -> solve.
#
# The physics is written by an LLM handed the neofoam MCP toolset and a prompt
# from test/e2e/cases/prompts.json (default: tube_bank, laminar). The geometry is
# deterministic: the mesh dicts are built from manifest.json and the LLM is told
# NOT to author them.
#
# Prerequisites: ANTHROPIC_API_KEY set, and a sourced OpenFOAM (for the solve).
#
# Usage (from the repo root or anywhere):
#   examples/hx/run.sh                 # laminar tube_bank prompt
#   examples/hx/run.sh heat_exchanger  # autonomous prompt (LLM picks the physics)
#
set -euo pipefail
HERE="$(cd "${0%/*}" && pwd)"
PROMPT="${1:-tube_bank}"

# 1) Author the case (stage geometry from THIS example's surfaces + let the AI
#    fill the physics via the MCP tools).
python - "$HERE" "$PROMPT" <<'PY'
import sys
from pathlib import Path

here, prompt = Path(sys.argv[1]), sys.argv[2]
repo = here.parents[1]
sys.path.insert(0, str(repo / "test" / "e2e" / "cases"))

import fill_tube_bank_mcp as driver
from neofoam.e2e.patch_set import PatchSet

# Start from this example's OWN surfaces + manifest (not the driver's fixtures).
driver.TRI_SURFACE = here / "constant" / "triSurface"
patch_set = PatchSet.load(here / "manifest.json")

case = driver.build_tube_bank(here, prompt=prompt, patch_set=patch_set)
print(f"case authored from surfaces ({prompt}): {case}")
PY

# 2) Mesh + solve in one command (in-process blockMesh -> snappyHexMesh -> solve).
cd "$HERE"
NEOFOAM_PYTHON="$(command -v python)" ./Allrun
