# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# The sink: gather every case's results/{id}.json into one self-contained
# report.html. This is what the `all` rule depends on.
#
# A harness fault makes the runner exit non-zero after writing the report, and
# Snakemake deletes a failed rule's outputs — the copy keeps the fault section
# readable in report-with-faults.html.
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
        " || {{ cp '{output}' report-with-faults.html; exit 1; }}"
