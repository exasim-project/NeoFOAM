# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A plugin module whose import fails, like one with a missing dependency."""

raise ImportError("missing dependency of the broken plugin")
