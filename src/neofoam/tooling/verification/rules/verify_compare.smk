# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case: diff every candidate backend against the native reference and fold the
# results into one results/<id>.json. Depends on the native reference status plus
# every candidate's status — all named by their solver — so it fires as soon as a
# case's runs are done, regardless of the rest of the sweep.
#
# Consumes header globals: CONFIG, WORK, RESULTS, STUDY, CANDIDATE_LABELS.

rule verify_compare:
    input:
        native=lambda wc: f"{WORK}/{wc.id}/{STUDY.by_id(wc.id).native_label}.status.json",
        candidates=lambda wc: [
            f"{WORK}/{wc.id}/{label}.status.json" for label in CANDIDATE_LABELS
        ],
    output:
        RESULTS + "/{id}.json",
    shell:
        "python -m neofoam.tooling.workflow.study.runner --config '{CONFIG}'"
        " compare --case {wildcards.id} --work '{WORK}' --out '{output}'"
