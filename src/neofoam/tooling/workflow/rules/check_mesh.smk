# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Validate the variant's mesh (fails the chain on mesh errors). Keyed by
# MESH_STEM (`{cad}__{mesh}` with a CAD axis, else `{mesh}`).
# Consumes header globals: BASE_CASE, MESH_STEM, _mesh_case, MESH_TOOL_INPUT.
rule checkMesh:
    input:
        MESH_STEM + "/" + MESH_TOOL_INPUT['checkMesh']
    output:
        MESH_STEM + "/.checkMesh.done"
    params:
        base=lambda wc: BASE_CASE,
        case=lambda wc: _mesh_case(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner tool"
        " --tool checkMesh --base '{params.base}'"
        " --case '{params.case}' --stamp '{output}'"
        " > '{params.case}/log.checkMesh' 2>&1"
