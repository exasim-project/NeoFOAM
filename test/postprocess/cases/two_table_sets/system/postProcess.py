"""A script with two TableSets — the loader cannot tell which one to run."""

from neofoam.postprocess import TableSet, VolIntegrate, field

postProcess = TableSet()
alsoPostProcess = TableSet()


@postProcess.table("volume_p.csv")
def volume_p():
    return field("p") | VolIntegrate(name="volume_p")
