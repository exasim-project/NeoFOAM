"""The script twin of cases/one_table: the same table, declared in Python."""

from neofoam.postprocess import TableSet, VolIntegrate, field

postProcess = TableSet()


@postProcess.table("volume_p.csv")
def volume_p():
    return field("p") | VolIntegrate(name="volume_p")
