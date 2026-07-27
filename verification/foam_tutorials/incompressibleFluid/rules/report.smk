# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# The sink: gather every case's results/{id}.json into one self-contained
# report.html. This is what the `all` rule depends on.
#
# Consumes header globals: CONFIG, RESULTS, REPORT, IDS, REPO_ROOT.

rule report:
    input:
        expand(RESULTS + "/{id}.json", id=IDS),
    output:
        REPORT,
    shell:
        "PYTHONPATH=\"{REPO_ROOT}:${{PYTHONPATH:-}}\" python -m verification.dropin.runner --config '{CONFIG}'"
        " report --results '{RESULTS}' --out '{output}'"
