# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case: diff every candidate backend against the native reference and fold the
# results into one results/{id}.json. Depends on the native .ran marker plus every
# candidate's .ran — so it fires as soon as a case's runs are done, regardless of
# the rest of the sweep. This is where the run outcome is decided: the runner reads
# each run dir (solver log + swap stamp), the shell run rule having only executed
# ./Allrun.
#
# Consumes header globals: CONFIG, CASE_ROOT, RESULTS, STUDY, CANDIDATE_LABELS, REPO_ROOT.

rule compare:
    input:
        native=lambda wc: f"{CASE_ROOT}/{wc.id}/{STUDY.by_id(wc.id).native_label}/.ran",
        candidates=lambda wc: [
            f"{CASE_ROOT}/{wc.id}/{label}/.ran" for label in CANDIDATE_LABELS
        ],
    output:
        RESULTS + "/{id}.json",
    shell:
        "PYTHONPATH=\"{REPO_ROOT}:${{PYTHONPATH:-}}\" python -m verification.dropin.runner --config '{CONFIG}'"
        " compare --case {wildcards.id} --cases '{CASE_ROOT}' --out '{output}'"
