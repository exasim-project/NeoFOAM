# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case, per solver: stage the tutorial (casebuild) and run its Allrun end to
# end, writing a status JSON. One rule keyed on (id, solver) covers the native
# reference and every candidate backend — the runner resolves the label itself
# (native solver vs a candidate command), so each run lands in a solver-named dir
# (`simpleFoam/`, `incompressiblefluid/`, `incompressiblefluidneon/`) rather than an
# anonymous `native/`/`neo/`. The DAG runs them all independently.
#
# `threads` is the case's numberOfSubdomains: a parallel case shells out to
# mpirun internally, so this keeps `snakemake -jN` from oversubscribing the box.
#
# The `|| mark-failed` backstop: staging runs in-process and casebuild can
# hard-exit the interpreter on a FOAM fatal error (uncatchable in Python), which
# would leave no status file and abort the whole DAG. On any non-zero exit the
# fallback writes a CASE_SETUP_FAILED status so the case is recorded and the run
# continues.
#
# Consumes header globals: CONFIG, WORK, THREADS.

rule verify_run:
    output:
        WORK + "/{id}/{solver}.status.json",
    threads: lambda wc: THREADS[wc.id]
    shell:
        "python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " run --case {wildcards.id} --solver {wildcards.solver}"
        " --work '{WORK}' --status '{output}'"
        " || python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " mark-failed --solver {wildcards.solver} --status '{output}'"
