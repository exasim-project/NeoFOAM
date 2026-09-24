"""The script twin of ../../postprocess_yaml: the same table, declared in Python.

``Print`` is in the pipeline so the run also proves a non-terminal node survives
the solver seam; it passes the dataset through unchanged.
"""

from neofoam.postprocess import Print, TableSet, VolIntegrate, field

postProcess = TableSet()


@postProcess.table("volume_p.csv")
def volume_p():
    return field("p") | Print(label="volume_p") | VolIntegrate(name="volume_p")
