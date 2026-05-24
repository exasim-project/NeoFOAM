"""
Declare per-model fvSchemes / fvSolution slices
================================================

OpenFOAM solvers share two case-level dictionaries: ``system/fvSchemes``
(discretization options) and ``system/fvSolution`` (linear-solver
controls). Each model that runs only needs a *slice* of those files —
PIMPLE reads ``divSchemes.div(phi,U)``; the turbulence model reads
``divSchemes.div(phi,k)``; both share the file but care about different
entries.

The framework exposes ``fvSchemes`` and
``fvSolution`` as ``BaseConfig`` base classes.
``spec.config(fvSchemes)`` returns a *per-spec subclass*; operations on
that spec extend it with typed entries via ``@<Subclass>.add(...)``.
Each ``.add(...)`` call injects typed Pydantic fields whose value types
come from ``neofoam.foam.schemes`` — bad values raise at LOAD time,
missing entries surface as Pydantic ``missing`` errors.

This page works the mechanism end-to-end against dictionary fixtures
shipped next to the example (YAML on disk so it runs without pybFoam).
"""

# %%
# Stage the case directory from shipped dictionary fixtures
# ---------------------------------------------------------
# Real OpenFOAM cases ship ``system/fvSchemes`` and ``system/fvSolution``
# in dictionary format. The fixtures shipped next to this page have the
# same *shape* — sections at the top level, each containing entry keys
# like ``div(phi,U)`` — kept as YAML so the example runs without pybFoam.
# We copy them into a throwaway case dir and load *from those files*
# rather than constructing the dictionaries inline.
#
# .. literalinclude:: ../../examples/how-to/per_model_fvSchemes.yaml
#    :language: yaml
#    :caption: per_model_fvSchemes.yaml
#
# .. literalinclude:: ../../examples/how-to/per_model_fvSolution.yaml
#    :language: yaml
#    :caption: per_model_fvSolution.yaml

import inspect
import shutil
import tempfile
from pathlib import Path

from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.model import Model
from neofoam.io import IOStrategy, YAML


def _here() -> None:
    """Marker used to locate this script on disk via inspect.getfile()."""


HERE = Path(inspect.getfile(_here)).resolve().parent
CASE_DIR = Path(tempfile.mkdtemp(prefix="neofoam_fvschemes_"))
for _name in ("per_model_fvSchemes.yaml", "per_model_fvSolution.yaml"):
    shutil.copy(HERE / _name, CASE_DIR / _name)
print("case_dir =", CASE_DIR)


# %%
# Declare the model and register per-spec subclasses
# --------------------------------------------------
# ``spec.config(fvSchemes)`` returns a fresh subclass scoped to this
# model. The subclass inherits the IOStrategy binding from the base; we
# override it here with ``@IOStrategy(YAML(...))`` so the example doesn't
# need pybFoam.

pimple = Model("Pimple")

PimpleFvSchemes = pimple.config(fvSchemes)
PimpleFvSolution = pimple.config(fvSolution)

# Point the synthesised subclasses at our shipped YAML fixtures.
IOStrategy(YAML("per_model_fvSchemes.yaml"))(PimpleFvSchemes)
IOStrategy(YAML("per_model_fvSolution.yaml"))(PimpleFvSolution)


# %%
# Operations declare which entries they need
# ------------------------------------------
# Every ``@PimpleFvSchemes.add(...)`` call extends the subclass with
# typed Pydantic fields for the listed entries. Section short-names
# (``ddt``, ``div``, ``grad``, ``laplacian``, ``snGrad``,
# ``interpolation``) map to their OpenFOAM long names; unknown names
# pass through with ``str`` typing.
#
# These decorators are no-ops on the function itself — their effect is
# the side effect of extending the subclass. Stack them on top of
# ``@pimple.operation(...)`` and the operation declares both its
# execution slot and the dictionary entries it reads, in one place.


@pimple.operation(operation_number="2.1")
@PimpleFvSchemes.add(div="div(phi,U)", grad="grad(U)", laplacian="laplacian(nuEff,U)")
@PimpleFvSolution.add("U")
def momentum() -> None:
    """Pretend U-momentum operation — body is the point of this page."""


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
@PimpleFvSchemes.add(grad="grad(p)", laplacian="laplacian(rAU,p)")
@PimpleFvSolution.add("p")
def continuity() -> None:
    """Pretend continuity operation."""


# %%
# What the subclass now looks like
# --------------------------------
# After both operations are imported, ``PimpleFvSchemes`` carries
# exactly the entries the pimple model needs. The Pydantic class
# itself is the schema — its ``model_fields`` and section sub-models
# encode both presence (required keys) and value type (DivScheme,
# GradScheme, …).

print("PimpleFvSchemes sections :", sorted(PimpleFvSchemes.model_fields))
print(
    "  divSchemes entries     :",
    sorted(PimpleFvSchemes.model_fields["divSchemes"].annotation.model_fields),
)
print(
    "  gradSchemes entries    :",
    sorted(PimpleFvSchemes.model_fields["gradSchemes"].annotation.model_fields),
)
print(
    "  laplacianSchemes ents  :",
    sorted(PimpleFvSchemes.model_fields["laplacianSchemes"].annotation.model_fields),
)
print(
    "PimpleFvSolution.solvers :",
    sorted(PimpleFvSolution.model_fields["solvers"].annotation.model_fields),
)


# %%
# Load and validate the typed slices
# ----------------------------------
# A single Pydantic validation pass per subclass checks both presence
# (every declared entry exists in the file) and type (every value
# matches its scheme union).

schemes = PimpleFvSchemes.load(case_dir=CASE_DIR)
solution = PimpleFvSolution.load(case_dir=CASE_DIR)

print("schemes.divSchemes.div_phi_U  :", schemes.divSchemes.div_phi_U)
print("schemes.gradSchemes.grad_U    :", schemes.gradSchemes.grad_U)
print("solution.solvers.U            :", solution.solvers.U)


# %%
# Bad input surfaces as a validation error
# ----------------------------------------
# An OpenFOAM file with an unparsable scheme value fails Pydantic
# validation — no separate verification step needed. The malformed
# fixture ships alongside the page:
#
# .. literalinclude:: ../../examples/how-to/per_model_fvSchemes_bad.yaml
#    :language: yaml
#    :caption: per_model_fvSchemes_bad.yaml

shutil.copy(
    HERE / "per_model_fvSchemes_bad.yaml", CASE_DIR / "per_model_fvSchemes_bad.yaml"
)
try:
    PimpleFvSchemes.load(case_dir=CASE_DIR, file="per_model_fvSchemes_bad.yaml")
except Exception as exc:
    print(f"caught {type(exc).__name__} as expected — bad scheme value")


# %%
# Two specs get independent subclasses
# ------------------------------------
# A second model registering ``fvSchemes`` gets a *separate* subclass.
# Adding entries to one does not affect the other — no shared mutable
# state between models.

turbulence = Model("Turbulence")
TurbFvSchemes = turbulence.config(fvSchemes)
TurbFvSchemes.add(div="div(phi,k)")

print("TurbFvSchemes is its own class:", TurbFvSchemes is not PimpleFvSchemes)
print("TurbFvSchemes sections        :", sorted(TurbFvSchemes.model_fields))


# %%
# See also
# --------
#
# - :doc:`example_custom_solver_with_configs` — solver-owned config
#   ownership story; the per-model fvSchemes pattern here is the
#   per-model analog.
# - ``neofoam.foam.schemes`` — the typed scheme unions used as
#   value types when ``.add(...)`` injects fields.
"""Cleanup the temporary case directory."""
shutil.rmtree(CASE_DIR, ignore_errors=True)
