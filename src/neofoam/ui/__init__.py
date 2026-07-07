# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoFOAM case-wizard UI (trame + JSONForms).

Renders each config's JSON schema as a native Vue form (JSONForms/Vuetify) inside a
trame app. Importing this package must not require trame — :func:`build_app` imports
it lazily.
"""

from __future__ import annotations

from typing import Any

from neofoam.ui.forms import FormEntry, build_forms

__all__ = ["FormEntry", "build_app", "build_forms"]


def build_app(
    server: Any = None,
    *,
    solver_name: str = "incompressibleFluid",
    plugins: Any = None,
) -> Any:
    """Lazy wrapper — imports trame only when the app is actually built."""
    from neofoam.ui.app import build_app as _build_app

    return _build_app(server, solver_name=solver_name, plugins=plugins)
