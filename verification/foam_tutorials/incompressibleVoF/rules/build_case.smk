# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Rule 1 of 3 (build → swap → run). Select + copy the tutorial with casebuild into
# a solver-named case dir, cleaned, truncated to STEP_BUDGET steps, with a fallback
# Allrun written if the tutorial ships none. Identical recipe for the native
# reference and every candidate — the solver swap is the next rule.
#
# `threads` is the case's numberOfSubdomains, so `snakemake -jN` does not
# oversubscribe a case that shells out to mpirun internally.
#
# The `|| mark-failed` backstop: casebuild runs in-process and can hard-exit the
# interpreter on a FOAM fatal error (uncatchable in Python). On any non-zero exit
# the fallback writes a stage_failed stamp so the case is recorded and the DAG
# continues instead of one bad case stranding the whole report.
#
# Consumes header globals: CONFIG, CASE_ROOT, THREADS, REPO_ROOT.

rule build_case:
    output:
        CASE_ROOT + "/{id}/{solver}/.built.json",
    threads: lambda wc: THREADS[wc.id]
    shell:
        "PYTHONPATH=\"{REPO_ROOT}:${{PYTHONPATH:-}}\" python -m verification.dropin.runner --config '{CONFIG}'"
        " build --case {wildcards.id} --solver {wildcards.solver}"
        " --cases '{CASE_ROOT}' --stamp '{output}'"
        " || PYTHONPATH=\"{REPO_ROOT}:${{PYTHONPATH:-}}\" python -m verification.dropin.runner --config '{CONFIG}'"
        " mark-failed --solver {wildcards.solver} --status '{output}'"
