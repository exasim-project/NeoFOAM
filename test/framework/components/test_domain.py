from pydantic import BaseModel
from foamadapter.framework.solver import Solver

@Solver
class MyCustomSolver(BaseModel):
    pass
    