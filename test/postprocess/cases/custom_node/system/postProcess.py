"""A case-defined node: registering it here makes ``type: clip`` valid in this
case's postProcess.yaml, which is resolved after this script has run."""

from typing import Literal

from neofoam.postprocess import DataSet, Node, TableSet, VolIntegrate, field

postProcess = TableSet()


@Node.register
class Clip(Node):
    """Raise every value below the threshold up to it."""

    type: Literal["clip"] = "clip"
    threshold: float = 0.0

    def compute(self, dataset: DataSet) -> DataSet:
        return dataset.with_values(dataset.values.clip(min=self.threshold))


@postProcess.table("clipped_p_script.csv")
def clipped_p_script():
    return field("p") | Clip(threshold=1.5) | VolIntegrate(name="clipped_p_script")
