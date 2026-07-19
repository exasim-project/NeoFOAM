# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Stage one mesh variant's mini-case (dicts + STLs from the base case, the
# variant's mesh-config payloads applied). Runs once per mesh mini-case, keyed
# by MESH_STEM (a composite `{cad}__{mesh}` when a CAD axis is present, else
# `{mesh}`); the concrete dir comes from _mesh_case.
# Consumes header globals: SOLVER, BASE_CASE, MESH_STEM, _mesh_case.
rule setup_mesh:
    input:
        "configs/mesh/{mesh}.json"
    output:
        MESH_STEM + "/.staged.json"
    params:
        solver=lambda wc: SOLVER,
        base=lambda wc: BASE_CASE,
        case=lambda wc: _mesh_case(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner mesh-setup"
        " --solver {params.solver} --base '{params.base}'"
        " --case '{params.case}' --config '{input}' --stamp '{output}'"
