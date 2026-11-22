# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""YAML format marker."""

from typing import Optional


class YAMLStrategy:
    """Format marker for YAML config files.

    Reads and writes go through :class:`neofoam.io.DictFile` (which dispatches on
    the ``.yaml``/``.yml`` suffix); this class survives as the format tag stored
    in ``IOMetadata``, carrying the optional ``subdict_path``.
    """

    def __init__(self, subdict_path: Optional[str] = None):
        self.subdict_path = subdict_path
