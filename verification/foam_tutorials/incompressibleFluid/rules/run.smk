# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Rule 3 of 3 (build → swap → run). Plain shell: run the prepared case's ./Allrun.
# No timeout (the solvers do not hang, and a batch scheduler like Slurm caps wall
# time), no interpretation — a broken solver's crash is the finding, so `|| true`
# keeps it from aborting the DAG. All "did it finish, and why not" is decided later
# by the compare rule, which reads this dir. A case that cannot run was neutralised
# by the swap rule (its Allrun is a no-op), so ./Allrun is always safe to invoke.
#
# The wall-clock seconds are recorded in .seconds for the report; .ran marks the
# rule done so compare can depend on it.
#
# `threads` is the case's numberOfSubdomains (a parallel case shells out to mpirun
# internally), so `snakemake -jN` does not oversubscribe the box.
#
# Consumes header globals: CASE_ROOT, THREADS.

rule run:
    input:
        CASE_ROOT + "/{id}/{solver}/.swapped.json",
    output:
        CASE_ROOT + "/{id}/{solver}/.ran",
    threads: lambda wc: THREADS[wc.id]
    shell:
        "d='{CASE_ROOT}/{wildcards.id}/{wildcards.solver}'; "
        "start=$(date +%s); "
        "( cd \"$d\" && ./Allrun > log.allrun 2>&1 ) || true; "
        "echo $(( $(date +%s) - start )) > \"$d/.seconds\"; "
        "touch '{output}'"
