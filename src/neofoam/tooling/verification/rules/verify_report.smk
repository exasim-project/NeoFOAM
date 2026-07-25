# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# The sink: gather every case's results/<id>.json into one self-contained
# report.html. This is the default target (the `all` rule depends on it).
#
# Consumes header globals: CONFIG, RESULTS, REPORT, IDS.

rule verify_report:
    input:
        expand(RESULTS + "/{id}.json", id=IDS),
    output:
        REPORT,
    shell:
        "python -m neofoam.tooling.workflow.study.runner --config '{CONFIG}'"
        " report --results '{RESULTS}' --out '{output}'"
