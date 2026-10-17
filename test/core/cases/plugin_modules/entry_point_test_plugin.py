# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A plugin module: importing it registers a family and one plugin, like a real add-on."""

from typing import Literal

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class entryPointTestFamily(BaseModel):
    """Family the test plugin extends."""


@entryPointTestFamily.register
class addOne(BaseModel):
    """The contributed plugin."""

    model_type: Literal["addOne"] = "addOne"
