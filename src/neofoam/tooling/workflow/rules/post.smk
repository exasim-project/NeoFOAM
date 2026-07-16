# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Per case: post-processing seam (opt-in; not part of the default pipeline).
# Replace the shell with your own post-processing command.
rule post:
    input:
        "cases/{case}/done"
    output:
        "cases/{case}/.post.done"
    shell:
        "touch '{output}'"
