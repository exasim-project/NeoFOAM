# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trame client module for the bundled JSONForms Vue component.

``server.enable_module(module)`` (where ``module`` is this package) makes trame
serve ``static/`` and load the UMD bundle + CSS on the client, then register the
``<json-forms>`` component via ``vue_use``. The bundle externalises Vue + Vuetify
to trame's client globals, so it binds to trame's single Vue app / Vuetify plugin
instance (see ``vite.config.mjs`` / ``entry.mjs``).

The built assets under ``static/`` are checked in; regenerate with
``bun install && bunx vite build`` in this directory.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["serve", "scripts", "styles", "vue_use", "STATIC_DIR"]

STATIC_DIR = Path(__file__).parent / "static"

_PREFIX = "__neofoam_jsonforms"

serve = {_PREFIX: str(STATIC_DIR)}
scripts = [f"{_PREFIX}/neofoam_jsonforms.umd.js"]
styles = [f"{_PREFIX}/neofoam_jsonforms.css"]
# Install the bundle's Vue plugin (registers the <json-forms> component) against
# trame's client app. The global name matches vite.config.mjs `lib.name`.
vue_use = ["neofoam_jsonforms"]
