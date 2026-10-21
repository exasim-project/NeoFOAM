from typing import Any, Literal

import numpy as np

from neofoam.postprocess import Node, Selector
from neofoam.preprocess import SetFields


@Node.register
class LeftHalf(Selector):
    """Everything left of ``x``."""

    type: Literal["leftHalf"] = "leftHalf"
    x: float

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return positions[:, 0] < self.x


setFields = SetFields(defaults={"alpha.water": 0.0})
