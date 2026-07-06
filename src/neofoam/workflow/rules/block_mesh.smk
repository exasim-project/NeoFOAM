# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Build the background mesh in the staged variant dir (single-tool
# preprocess.yaml slice; options come from the base case's entry).
# Consumes header globals: BASE_CASE, MESH_TOOL_INPUT.
rule blockMesh:
    input:
        lambda wc: f"meshes/{wc.mesh}/{MESH_TOOL_INPUT['blockMesh']}"
    output:
        "meshes/{mesh}/.blockMesh.done"
    params:
        base=lambda wc: BASE_CASE,
    shell:
        "python -m neofoam.workflow.sweep_runner tool"
        " --tool blockMesh --base '{params.base}'"
        " --case 'meshes/{wildcards.mesh}' --stamp '{output}'"
        " > 'meshes/{wildcards.mesh}/log.blockMesh' 2>&1"
