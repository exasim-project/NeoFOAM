# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Regenerate the STL geometry a mesh variant meshes, from a parametric CAD
# model. Runs first in the mesh chain (before blockMesh) when the sweep carries
# a CAD axis; the blockMesh/snappyHexMesh chain then meshes the freshly written
# constant/triSurface/*.stl. Only ever included when a CAD axis is present, so
# MESH_STEM carries the `{cad}` wildcard the per-variant config is keyed on.
# Consumes header globals: CAD_MODEL, MESH_STEM, _mesh_case, MESH_TOOL_INPUT.
rule cad_geometry:
    input:
        MESH_STEM + "/" + MESH_TOOL_INPUT['cad_geometry']
    output:
        MESH_STEM + "/constant/triSurface/.cad.done"
    params:
        model=lambda wc: CAD_MODEL,
        cad_json=lambda wc: f"configs/cad/{wc.cad}.json",
        case=lambda wc: _mesh_case(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner cad"
        " --model '{params.model}' --params '{params.cad_json}'"
        " --case '{params.case}' --stamp '{output}'"
        " > '{params.case}/log.cad' 2>&1"
