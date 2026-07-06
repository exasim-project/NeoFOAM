# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Refine/snap the variant's mesh around the STL surfaces (single-tool slice;
# resumes from the constant/polyMesh its predecessor left on disk).
# Consumes header globals: BASE_CASE, MESH_TOOL_INPUT.
rule snappyHexMesh:
    input:
        lambda wc: f"meshes/{wc.mesh}/{MESH_TOOL_INPUT['snappyHexMesh']}"
    output:
        "meshes/{mesh}/.snappyHexMesh.done"
    params:
        base=lambda wc: BASE_CASE,
    shell:
        "python -m neofoam.workflow.sweep_runner tool"
        " --tool snappyHexMesh --base '{params.base}'"
        " --case 'meshes/{wildcards.mesh}' --stamp '{output}'"
        " > 'meshes/{wildcards.mesh}/log.snappyHexMesh' 2>&1"
