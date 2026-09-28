"""A script that also registers a write policy — a family the loader must leave alone."""

from typing import Literal

from neofoam.algorithms.field_writer.write_control import StepView, WriteControl
from neofoam.postprocess import TableSet, VolIntegrate, field


@WriteControl.register
class NeverWriteControl(WriteControl):
    """Never a write step."""

    write_control_type: Literal["never"] = "never"

    def should_write(self, stepper: StepView) -> bool:
        return False


postProcess = TableSet()


@postProcess.table("volume_p.csv")
def volume_p():
    return field("p") | VolIntegrate(name="volume_p")
