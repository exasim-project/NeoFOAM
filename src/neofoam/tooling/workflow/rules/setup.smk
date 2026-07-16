# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case: clone the base case, apply the case's config payloads and copy the
# built mesh in from the case's mesh mini-case. Removes the clone's
# preprocess.yaml so nothing ever re-meshes it. The mesh dir (_mesh_dir_of)
# composes the case's cad + mesh variants, matching the mesh chain's MESH_STEM.
# Consumes header globals: SOLVER, BASE_CASE, _mesh_dir_of, MESH_DONE.
rule setup:
    input:
        cfg="configs/{case}/setup.json",
        mesh=lambda wc: f"{_mesh_dir_of(wc)}/{MESH_DONE}",
    output:
        "cases/{case}/.applied.json"
    params:
        solver=lambda wc: SOLVER,
        base=lambda wc: BASE_CASE,
        mesh_dir=lambda wc: _mesh_dir_of(wc),
    shell:
        "python -m neofoam.tooling.workflow.sweep_runner setup"
        " --solver {params.solver} --base '{params.base}'"
        " --case 'cases/{wildcards.case}' --config '{input.cfg}'"
        " --mesh-src '{params.mesh_dir}' --stamp '{output}'"
