import inspect
import functools
from pydantic import BaseModel
from typing import ClassVar, Protocol, runtime_checkable, Callable
from foamadapter.framework.dag import NodeData

from typing import Annotated, get_origin, get_args,Protocol, Any

from .context import Context, FieldUpdates
from .step import Step, step, update_dependencies
from dataclasses import is_dataclass, dataclass





def Solver(cls):
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

Solver.step = staticmethod(step)

@runtime_checkable
class SolverInterface(Protocol):
    _steps: ClassVar[list[Step]]

    @classmethod
    def steps(cls) -> int:
        ...

    def dependencies(self) -> list[NodeData]:
        ...

    def main_loop(self, ctx: Context):
        ...

