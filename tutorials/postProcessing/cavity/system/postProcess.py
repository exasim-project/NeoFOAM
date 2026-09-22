"""In-situ post-processing tables — the script front door of this case.

The solver executes this file before it resolves ``system/postProcess.yaml``, so
the ``Square`` node registered here is usable from that file as ``type: square``.
Table names must not collide with the declared ones, hence the distinct names.
"""

from typing import Literal

from neofoam.algorithms.field_writer.write_control import IntervalWriteControl
from neofoam.postprocess import InternalDataSet, Mag, Node, Scale, TableSet, VolIntegrate, field


@Node.register
class Square(Node):
    """Square every value."""

    type: Literal["square"] = "square"

    def compute(self, dataset: InternalDataSet) -> InternalDataSet:
        return dataset.with_field(dataset.field**2)


postProcess = TableSet()  # the single module-level TableSet the loader picks up


# The decorated function runs here, at load time, so ``Square`` must already
# be defined above it.
@postProcess.table("kinetic_energy.csv")
def kinetic_energy():
    """Half the volume integral of |U|^2 — the cavity spinning up, built with ``|``."""
    return field("U") | Mag() | Square() | Scale(factor=0.5) | VolIntegrate(name="kinetic_energy")


@postProcess.table("volume_U_script.csv", write_control=IntervalWriteControl(interval=10))
def volume_U_script():
    """A vector table on its own cadence: every tenth step, in mm/s."""
    return field("U") | Scale(factor=1000.0) | VolIntegrate(name="volume_U_script")
