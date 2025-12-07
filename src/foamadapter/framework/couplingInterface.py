# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from pydantic import BaseModel


class CouplingInterface(BaseModel):
    type: str
