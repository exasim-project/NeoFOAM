# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Validate the variant's mesh (fails the chain on mesh errors).
# Consumes header globals: BASE_CASE, MESH_TOOL_INPUT.
rule checkMesh:
    input:
        lambda wc: f"meshes/{wc.mesh}/{MESH_TOOL_INPUT['checkMesh']}"
    output:
        "meshes/{mesh}/.checkMesh.done"
    params:
        base=lambda wc: BASE_CASE,
    shell:
        "python -m neofoam.workflow.sweep_runner tool"
        " --tool checkMesh --base '{params.base}'"
        " --case 'meshes/{wildcards.mesh}' --stamp '{output}'"
        " > 'meshes/{wildcards.mesh}/log.checkMesh' 2>&1"
