# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Stage one mesh variant's mini-case (dicts + STLs from the base case, the
# variant's mesh-config payloads applied). Runs once per mesh variant.
# Consumes header globals: SOLVER, BASE_CASE.
rule setup_mesh:
    input:
        "configs/mesh/{mesh}.json"
    output:
        "meshes/{mesh}/.staged.json"
    params:
        solver=lambda wc: SOLVER,
        base=lambda wc: BASE_CASE,
    shell:
        "python -m neofoam.workflow.sweep_runner mesh-setup"
        " --solver {params.solver} --base '{params.base}'"
        " --case 'meshes/{wildcards.mesh}'"
        " --config '{input}' --stamp '{output}'"
