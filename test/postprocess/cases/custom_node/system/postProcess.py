"""A case-defined node: registering it here makes ``type: clip`` valid in this
case's postProcess.yaml, which is resolved after this script has run."""

from typing import Literal

from neofoam.postprocess import InternalDataSet, Node, TableSet, VolIntegrate, field

postProcess = TableSet()


@Node.register
class Clip(Node):
    """Raise every value below the threshold up to it."""

    type: Literal["clip"] = "clip"
    threshold: float = 0.0

    def compute(self, dataset: InternalDataSet) -> InternalDataSet:
        return dataset.with_field(dataset.field.clip(min=self.threshold))


@postProcess.table("clipped_p_script.csv")
def clipped_p_script():
    return field("p") | Clip(threshold=1.5) | VolIntegrate(name="clipped_p_script")
