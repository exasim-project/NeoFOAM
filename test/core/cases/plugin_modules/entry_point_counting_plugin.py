# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A plugin module that counts how often its body runs; a reload keeps the old count."""

import_count: int = globals().get("import_count", 0) + 1
