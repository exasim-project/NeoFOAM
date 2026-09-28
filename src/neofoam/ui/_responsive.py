# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Phone/desktop switching shared by the wizard's two drawers."""

from __future__ import annotations

from typing import Any

# Below Vuetify's md breakpoint (960 px) both drawers are closed-by-default overlays
# with their own open flags, so a phone never inherits the desktop's open drawers.
_MOBILE = "$vuetify.display.smAndDown"


def _responsive_open(desktop_flag: str, mobile_flag: str) -> dict[str, Any]:
    """Drawer props: permanent + ``desktop_flag`` on desktop, overlay + ``mobile_flag`` below."""
    return {
        "permanent": (f"!{_MOBILE}",),
        "temporary": (_MOBILE,),
        "model_value": (f"{_MOBILE} ? {mobile_flag} : {desktop_flag}",),
        "update_modelValue": f"{_MOBILE} ? ({mobile_flag} = $event) : ({desktop_flag} = $event)",
    }


def _toggle(desktop_flag: str, mobile_flag: str) -> str:
    """JS toggling whichever open flag the current display width uses."""
    return f"{_MOBILE} ? ({mobile_flag} = !{mobile_flag}) : ({desktop_flag} = !{desktop_flag})"
