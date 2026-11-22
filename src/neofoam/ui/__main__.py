# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``python -m neofoam.ui`` — build the app and start the trame server."""

from neofoam.ui import build_app

if __name__ == "__main__":
    build_app().start()
