# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Transitional shell — the drop-in verification engine now lives elsewhere.

Everything that used to be here is :mod:`neofoam.tooling.workflow.study`. What
remains is :mod:`~neofoam.tooling.verification.rules`, the packaged combined-run
``.smk`` set the ``incompressibleVoF`` study still includes; it is deleted once
that study moves onto the shared ``study.smk`` pipeline.
"""

__all__: list[str] = []
