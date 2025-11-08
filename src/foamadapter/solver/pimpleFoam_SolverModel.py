import sys
from pydantic import BaseModel, create_model, Field
from foamadapter.framework.solver import Solver
from pybFoam.io.model_base import IOModelBase

from foamadapter.framework.step import Step
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)

from ..inputs_files.system import ControlDictBase, FvSchemesBase, DIVSchemes
from pybFoam import (
    volScalarField,
    volVectorField,
    surfaceScalarField,
    fvScalarMatrix,
    fvVectorMatrix,
    fvMesh,
    Time,
    fvc,
    fvm,
    Word,
    dictionary,
    Info,
    dynamicFvMesh,
    solve,
    adjustPhi,
    constrainPressure,
    createPhi,
    setRefCell,
    constrainHbyA,
    pimpleControl,
    computeCFLNumber,
)
from ..inputs_files.case_inputs import Registry, FileSpec
from ..turbulence.incompressible import TurbulenceModel
from ..modules.pressureVelocityCoupling.incompressible import (
    PimpleAlgorithm,
    PressureVelocityFields,
)
import pybFoam
from pybFoam import Time, dynamicFvMesh, Info, pimpleControl
from ..modules.pressureVelocityCoupling.incompressible import PressureVelocityFields

from ..modules.stability_criteria import StabilityCriteria
from ..modules.transportModels import SinglePhaseTransportModel
from ..modules.fields import Fields
from ..modules.models import Models
from ..modules.setup import initialize_containers
from ..modules.setup import visualize_dag, containers_to_deps
from foamadapter.modules import models

ControlDict = create_model(
    "controlDict",
    maxCo=(float, Field(..., description="Maximum Courant number")),
    __base__=ControlDictBase,
)

divSchemes = create_model(
    "divSchemes",
    __base__=DIVSchemes,
)

FvSchemes = create_model("fvSchemes", __base__=FvSchemesBase)


class TransportProperties(IOModelBase):
    transportModel: str = Field(..., description="Transport model type")
    nu: float = Field(..., description="Kinematic viscosity", gt=0)


def create_fields_models(mesh):
    pU = PimpleAlgorithm(mesh=mesh)

    fields = Fields()
    fields.add_fields(pU.register_fields(mesh=mesh))

    models = Models(models={})
    models.add_model("pU", pU)

    singlePhaseTransportModel = SinglePhaseTransportModel()
    models.add_model("singlePhaseTransportModel", singlePhaseTransportModel)

    turbulence = TurbulenceModel.from_ofdict(
        dictionary.read("constant/turbulenceProperties")
    )
    models.add_model("turbulence", turbulence)

    # visualize_dag(
    #     containers_to_deps(fields, models), "pimpleFoam_fields_models_dag", show=True
    # )
    initialize_containers(fields, models)

    return fields, models


@Solver
class PimpleFoamSolver:

    @classmethod
    def inputs(cls):
        registry = Registry({})
        registry = PimpleAlgorithm.inputs(registry)
        registry = TurbulenceModel.inputs(registry)
        registry = SinglePhaseTransportModel.inputs(registry)
        return registry

    def __init__(self, argv):
        """
        Initialize the PIMPLE solver with command line arguments.
        """
        self._argv = argv

    def create_context(self) -> Context:
        argList = pybFoam.argList(self._argv)
        runTime = Time(argList)
        mesh = dynamicFvMesh.New(argList, runTime)

        field, models = create_fields_models(mesh=mesh)
        ctx = Context(fields={}, models={}, mesh=mesh, runTime=runTime)
        for name, fld in field.entries.items():
            ctx.fields[name] = fld
        for name, mdl in models.entries.items():
            ctx.models[name] = mdl
        return ctx

    @Solver.step(step_number=1, depends_on=[])
    def advance_time(self, runTime: Time, stability_criteria: StabilityCriteria):
        Info(f"Time = {runTime.timeName()}")
        stability_criteria.setDelta(runTime)

    @Solver.step(step_number=2, depends_on=["advance_time"])
    def momentum_predictor(
        self, puData: PressureVelocityFields, pimple: PimpleAlgorithm
    ):
        while pimple.loop():  # <-- this a problem for readability
            self.pU.momentum_equation(pimple, puData)

    @Solver.step(step_number=3, depends_on=["advance_time"])
    def pressure_corrector(
        self, puData: PressureVelocityFields, pimple: PimpleAlgorithm
    ):
        while pimple.correct():
            self.pU.pressure_correction(pimple, puData)

    @Solver.step(step_number=4, depends_on=["momentum_predictor"])
    def turbulence(self, pimple: PimpleAlgorithm):
        if pimple.turbCorr():
            self.models["singlePhaseTransportModel"].correct()
            self.models["turbulence"].correct()

    @Solver.step(step_number=5, depends_on=["turbulence"])
    def write(self):
        self.runTime.write(True)
        self.runTime.printExecutionTime()

    def steps(self, domain_name: str | None = None) -> list[Step]:
        pass
        # builder = StepBuilder()
        # builder.step(self.advance_time)
        # builder.step(self.momentum_predictor)
        # builder.step(self.pressure_corrector)
        # builder.step(self.turbulence)
        # builder.step(self.write)
        # return builder.steps
        # # steps = [*self._steps]
        # # for step in steps:
        # #     step.cls = self
        # #     step.domain = domain_name
        # # return steps

    def configure_steps(self) -> Operations:
        main_loop = StepBuilder()
        ops = self.operations()
        pU_ops = self.models["pU"].operations()

        with main_loop.loop(ops["time_loop"]) as time_loop:

            time_loop.step(ops["advance_time"])

            with time_loop.loop(ops["pimple_loop"]) as pimple_loop:

                pimple_loop.step(pU_ops["momentum_predictor"])
                pimple_loop.step(pU_ops["pressure_corrector"])
                pimple_loop.step(ops["turbulence"])

            time_loop.step(ops["write"])

        return main_loop.operations

    def main_loop(self, ctx: Context) -> None:
        
        ops = self.configure_steps()
        models = self.configure_models()

        ops = update_steps(ops, models)
        ops.run(ctx)



@Model
class TracerEquation:

    @Model.step
    def solve_tracer(self, ...):
        pass
        
    def ...




def run_solver(argv):
    solver = PimpleFoamSolver(argv=argv)
    ctx = solver.create_context()
    for step in solver.steps():
        print(f"Running step: {step.step_name}")
        step.run(ctx)


# def main(argv):
#     solver = PimpleFoamSolver(argv=argv)
#     ctx = solver.create_context()
#     for step in solver.steps():
#         print(f"Running step: {step.step_name}")
#         step(ctx)
#         step(ctx)


# if __name__ == "__main__":
#     main(sys.argv)
