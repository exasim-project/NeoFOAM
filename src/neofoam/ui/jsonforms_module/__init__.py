# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Bundled JSONForms trame client module (see :mod:`.module`)."""

from .module import STATIC_DIR, scripts, serve, styles, vue_use

__all__ = ["serve", "scripts", "styles", "vue_use", "STATIC_DIR"]
