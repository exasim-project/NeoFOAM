from typing import Literal, Optional

from pydantic import BaseModel

from neofoam.framework.decorator import decorated_member_functions
from neofoam.framework.model import Model, ModelInterface
from neofoam.framework.operations import Operation, OperationCollection


@Model
class MyCustomModel(BaseModel):
    name: Literal["MyCustomModel"] = "MyCustomModel"

    @Model.operation(operation_number=1)  # check if the ops are sorted
    def op_one(self):
        pass

    @Model.operation(operation_number=2, depends_on=["op_one"])
    def op_two(self):
        pass

    @Model.operation(operation_number=3, depends_on=["op_two"])
    def op_three(self):
        pass

    @Model.operation(operation_number=4, depends_on=["op_three"])
    def op_four(self):
        pass

    def operations(self, domain_name: Optional[str] = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def run(self, ctx):
        pass


def test_model_ops_registration():
    assert issubclass(MyCustomModel, ModelInterface)
    # assert MyCustomModel.number_steps() == 4

    my_model = MyCustomModel()
    ops = my_model.operations()
    assert len(ops) == 4

    first_op = ops[0]
    assert first_op.operation_name == "op_one"
    assert first_op.operation_number == 1
    assert first_op.depends_on == []

    second_op = ops[1]
    assert second_op.operation_name == "op_two"
    assert second_op.operation_number == 2
    assert second_op.depends_on == ["op_one"]

    third_op = ops[2]
    assert third_op.operation_name == "op_three"
    assert third_op.operation_number == 3
    assert third_op.depends_on == ["op_two"]

    fourth_op = ops[3]
    assert fourth_op.operation_name == "op_four"
    assert fourth_op.operation_number == 4
    assert fourth_op.depends_on == ["op_three"]

    solver = MyCustomModel()
    assert solver.name == "MyCustomModel"

    schema = MyCustomModel.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
