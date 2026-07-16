# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Build the background mesh in the staged variant dir (single-tool
# preprocess.yaml slice; options come from the base case's entry). Keyed by
# MESH_STEM (`{cad}__{mesh}` with a CAD axis, else `{mesh}`).
# Consumes header globals: BASE_CASE, MESH_STEM, _mesh_case, MESH_TOOL_INPUT.
rule blockMesh:
    input:
        MESH_STEM + "/" + MESH_TOOL_INPUT['blockMesh']
    output:
        MESH_STEM + "/.blockMesh.done"
    params:
        base=lambda wc: BASE_CASE,
        case=lambda wc: _mesh_case(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner tool"
        " --tool blockMesh --base '{params.base}'"
        " --case '{params.case}' --stamp '{output}'"
        " > '{params.case}/log.blockMesh' 2>&1"
