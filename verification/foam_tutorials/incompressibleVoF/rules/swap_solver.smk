# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Rule 2 of 3 (build → swap → run). Make the built case run this solver: replace the
# solver token in Allrun with the neofoam command (and apply the study's neo_patch
# deviation). Candidate-only — the native reference is a no-op, its Allrun left
# pristine. A build that failed staging, or an Allrun with no unique solver token,
# is recorded in the stamp and carried forward to the run rule, never raised.
#
# Consumes header globals: CONFIG, CASE_ROOT, REPO_ROOT.

rule swap_solver:
    input:
        CASE_ROOT + "/{id}/{solver}/.built.json",
    output:
        CASE_ROOT + "/{id}/{solver}/.swapped.json",
    shell:
        "PYTHONPATH=\"{REPO_ROOT}:${{PYTHONPATH:-}}\" python -m verification.dropin.runner --config '{CONFIG}'"
        " swap --case {wildcards.id} --solver {wildcards.solver}"
        " --cases '{CASE_ROOT}' --built '{input}' --stamp '{output}'"
