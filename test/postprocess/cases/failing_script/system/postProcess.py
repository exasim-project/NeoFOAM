"""A script that registers a node and then fails — the registration must not
outlive the failed load, or ``clip`` would name two classes at the next load."""

from typing import Literal

from neofoam.postprocess import InternalDataSet, Node, TableSet, VolIntegrate, field

postProcess = TableSet()


@Node.register
class Clip(Node):
    """The same ``type`` cases/custom_node registers."""

    type: Literal["clip"] = "clip"
    threshold: float = 0.0

    def compute(self, dataset: InternalDataSet) -> InternalDataSet:
        return dataset.with_field(dataset.field.clip(min=self.threshold))


@postProcess.table("clipped_p.csv")
def clipped_p():
    return field("p") | Clip(threshold=1.5) | VolIntegrate(name="clipped_p")


raise RuntimeError("this case's script is broken")
