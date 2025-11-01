
from foamadapter.framework.dag import NodeData
from foamadapter.framework.solver import Step, step, update_dependencies
from foamadapter.framework.context import Context
from typing import Protocol, runtime_checkable, ClassVar


def Model(cls):
    step_functions: list[Step] = []
    
    # Collect all step functions
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if getattr(attr, "_is_step", False):
            step_functions.append(
                Step(
                    func=attr,
                    cls=cls,
                    step_number=attr._step_number,
                    depends_on=attr._depends_on,
                    step_name=attr.__name__,
                )
            )
    # Sort steps by step_number
    step_functions = sorted(step_functions, key=lambda s: s.step_number)
    
    # Automatically set dependencies if not provided
    step_functions = update_dependencies(step_functions)
    
    # Store the step functions
    cls._steps = step_functions
    cls.number_steps = classmethod(lambda c: len(c._steps))
    
    return cls

Model.step = staticmethod(step)


@runtime_checkable
class ModelInterface(Protocol):
    _steps: ClassVar[list[Step]]

    def steps(self) -> list[Step]:
        ...

    def dependencies(self) -> list[NodeData]:
        ...

    def run(self, ctx: Context):
        ...
