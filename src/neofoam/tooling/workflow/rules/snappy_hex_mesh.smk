# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Refine/snap the variant's mesh around the STL surfaces (single-tool slice;
# resumes from the constant/polyMesh its predecessor left on disk). Keyed by
# MESH_STEM (`{cad}__{mesh}` with a CAD axis, else `{mesh}`).
# Consumes header globals: BASE_CASE, MESH_STEM, _mesh_case, MESH_TOOL_INPUT.
rule snappyHexMesh:
    input:
        MESH_STEM + "/" + MESH_TOOL_INPUT['snappyHexMesh']
    output:
        MESH_STEM + "/.snappyHexMesh.done"
    params:
        base=lambda wc: BASE_CASE,
        case=lambda wc: _mesh_case(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner tool"
        " --tool snappyHexMesh --base '{params.base}'"
        " --case '{params.case}' --stamp '{output}'"
        " > '{params.case}/log.snappyHexMesh' 2>&1"
