"""Two tables sharing the TableSet's default runTime cadence."""

from neofoam.algorithms.field_writer.write_control import RunTimeWriteControl
from neofoam.postprocess import TableSet, VolIntegrate, field

postProcess = TableSet(RunTimeWriteControl(interval=0.1))


@postProcess.table("first.csv")
def first():
    return field("p") | VolIntegrate(name="first")


@postProcess.table("second.csv")
def second():
    return field("p") | VolIntegrate(name="second")
