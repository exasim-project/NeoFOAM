# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case: run the solver on the prepared clone. --no-preprocess because the
# mesh was copied in by setup (and the clone's preprocess.yaml was removed).
# Consumes header globals: SOLVER_CMD.
rule solve:
    input:
        "cases/{case}/.applied.json"
    output:
        "cases/{case}/done"
    params:
        solver_cmd=lambda wc: SOLVER_CMD,
    shell:
        "cd 'cases/{wildcards.case}' &&"
        " neofoam solver {params.solver_cmd} --no-preprocess"
        " > log.solver 2>&1 && touch done"
