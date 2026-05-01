# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for the dummy solver's optional models."""

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummySolverModel(BaseModel):
    """Plugin interface — optional models register with this."""

    pass
