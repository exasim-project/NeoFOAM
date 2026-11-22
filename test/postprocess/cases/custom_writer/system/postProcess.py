"""A case-defined writer: registering it here makes ``writer: {type: text}``
valid in this case's postProcess.yaml, which is resolved after this script has run."""

from pathlib import Path
from typing import Literal

from pydantic import PrivateAttr

from neofoam.postprocess import (
    AggregatedDataSet,
    TableSet,
    TableWriter,
    VolIntegrate,
    field,
    table_rows,
)

postProcess = TableSet()


@TableWriter.register
class TextWriter(TableWriter):
    """Write one space-separated line per row to ``<table>.txt``."""

    type: Literal["text"] = "text"

    _path: Path = PrivateAttr(default=Path())

    def open(self, path_stem: Path, *, append: bool) -> None:
        self._path = path_stem.with_name(path_stem.name + ".txt")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self._path.write_text("")

    def write(self, time: float, result: AggregatedDataSet) -> None:
        with self._path.open("a") as handle:
            for row in table_rows(result):
                handle.write(" ".join(str(value) for value in [time, *row]) + "\n")


@postProcess.table("volume_p_script.txt", writer=TextWriter())
def volume_p_script():
    return field("p") | VolIntegrate(name="volume_p_script")
