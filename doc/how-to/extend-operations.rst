Extend operations from another model
====================================

An operation sometimes has to do something *extra* only when some other
model is active: take the flux relative to a rotating frame, subtract an
``fvOptions`` source, re-find zone faces after a mesh move. Writing that
as ``if "mrf_zones" in ctx.models`` inside the algorithm couples the
algorithm to every optional model that will ever exist. An **extension
point** inverts that: the operation module declares *where* it can be
extended, models register implementations, and the operation calls
whatever the case activated — without naming a single model.

Extension point or model interface?
-----------------------------------

Two seams exist, and picking the wrong one is the usual mistake:

:class:`~neofoam.framework.model.extension.ExtensionPoint`
    Several **heterogeneous sites** — a term to add here, a constraint to
    apply there, a boundary to correct after the solve. Declared by the
    *operation module*, one point per operation with one method per site,
    no combining rule. The operation receives every active implementation
    behind one aggregated container and calls each site once.

:class:`~neofoam.framework.model.interface.ModelInterface`
    One **single value** combined by one rule (``sum``, ``min``, ``any``).
    Declared with ``@<model>.interface``
    (:meth:`ModelSpec.interface
    <neofoam.framework.model.spec.ModelSpec.interface>`) on the fold
    function, contributed to with ``@<model>.contributes``
    (:meth:`ModelSpec.contributes
    <neofoam.framework.model.spec.ModelSpec.contributes>`), and *called*
    by the consumer to get the folded result — see
    :doc:`/auto_how-to/example_use_an_interface`.

Rule of thumb: if you would have to write a fold per method, you want an
extension point; if one ``sum(...)`` covers it, you want an interface.

The worked example on this page is the pressure-velocity seam of the
``incompressibleFluid`` solver: MRF and ``fvOptions`` hook into
``UEqn.H`` / ``pEqn.H`` at a dozen different sites, spread over the
``momentum``, ``continuity`` and ``mesh_update`` operations — one point
each, which is exactly the multi-method case.

Declare the point next to the operations
----------------------------------------

The declaring module owns two things per extensible operation: an
interface class with one **no-op default method per site**, and one
module-level ``ExtensionPoint`` naming it. Both live next to the
operations they serve — in
``solver/incompressibleFluid/models/pressure_velocity/extension.py``:

.. code-block:: python

    from neofoam.framework.model import ExtensionPoint


    class MomentumExtension:
        """How the momentum operation can be extended."""

        def correct_boundary_velocity(self, U: volVectorField) -> None:
            """Set the boundary velocities the coefficients are built from."""

        def terms(self, U: volVectorField) -> list[Any]:
            """Terms folded into the momentum sum at ``+ ext.terms(U)``."""
            return []


    class PressureExtension:
        """How the continuity operation can be extended."""

        def make_relative(self, phiHbyA: surfaceScalarField) -> None:
            """Take the predicted flux relative to whatever frame this model adds."""

        def constrain_pressure(self, p, U, phiHbyA, rAU) -> bool:
            """Constrain the pressure boundaries; ``True`` if this call handled it."""
            return False


    momentum_extension = ExtensionPoint(
        "momentum_extension", MomentumExtension, folds_into=pyf.tmp_fvVectorMatrix
    )
    pressure_extension = ExtensionPoint("pressure_extension", PressureExtension)

``folds_into`` names the type term sums have at the operation's fold site
— declare it iff the interface has a term site, so ``+ ext.terms(U)`` can
join an equation expression (see below).

Three conventions matter here:

* **One interface per operation, not one per module.** ``momentum``,
  ``continuity`` and ``mesh_update`` each declare their own
  (``MomentumExtension``, ``PressureExtension``,
  ``MeshUpdateExtension``), so an implementation only ever sees the
  sites of the operation it extends. A model that acts in one operation
  — ``fvOptions`` has nothing to say about a mesh move — then carries no
  inherited no-ops from the others, and each operation's parameter type
  names exactly what it may call.
* **Every method has a working default** (``None``, ``[]``, the
  identity, ``False``). An implementation then overrides only the sites
  its model acts at — ``_FvOptionsMomentumExtension`` overrides three of
  the four momentum sites, and inherits the rest.
* **A method that reports back returns a value the operation can act
  on.** ``constrain_pressure`` returns ``True`` when it handled the
  constraint, which lets the operation fall back to the plain call when
  nothing did (see below).

The ``ExtensionPoint`` instance holds only factories and their owning
specs — no ``Context``, no live case objects — so one module-level
instance is safe across runs.

Register an implementation from a model
---------------------------------------

An implementing model subclasses the interface class and registers a
**factory** with ``@<model>.extends`` (:meth:`ModelSpec.extends
<neofoam.framework.model.spec.ModelSpec.extends>`) — one per point it
acts at, so a model that reaches into several operations ships one small
class each rather than one class straddling all of them:

.. code-block:: python

    from neofoam.fv_options import fvOptions
    from neofoam.mrf import mrf


    class _MRFMomentumExtension(MomentumExtension):
        def __init__(self, mrf_zones: pyf.IOMRFZoneList) -> None:
            self._mrf_zones = mrf_zones

        def terms(self, U: volVectorField) -> list[Any]:
            return [self._mrf_zones.DDt(U)]


    class _MRFPressureExtension(PressureExtension):
        def __init__(self, mrf_zones: pyf.IOMRFZoneList) -> None:
            self._mrf_zones = mrf_zones

        def make_relative(self, phiHbyA: surfaceScalarField) -> None:
            self._mrf_zones.makeRelative(phiHbyA)


    @mrf.extends(momentum_extension)
    def make_mrf_momentum_extension(mrf_zones: Annotated[Any, "models"]) -> MomentumExtension:
        """Wrap the case's zone list for the momentum sites."""
        return _MRFMomentumExtension(mrf_zones)


    @mrf.extends(pressure_extension)
    def make_mrf_pressure_extension(mrf_zones: Annotated[Any, "models"]) -> PressureExtension:
        """Wrap the case's zone list for the continuity sites."""
        return _MRFPressureExtension(mrf_zones)

The factory's parameters resolve exactly like an
``@model.operation`` body's, against the **registering model's** own
runtime plus the live ``Context``:

``Annotated[Any, "models"]``
    ``ctx.models[<param name>]`` — here the ``mrf_zones`` object the MRF
    model's ``@build`` step published.

A ``BaseConfig`` subclass
    Pulled by type from the *registering* model's config, not from
    whoever consumes the point.

A bare name
    ``ctx.fields[<param name>]``; ``Depends(...)`` markers work too.

A parameter nothing supplies raises a ``ValueError`` naming the point,
the factory, and the parameter — a misconfigured model can never
silently drop out.

Consume it in an operation
--------------------------

The consuming operation adds one parameter annotated
``Annotated[Extensions[Iface], point]`` — its **own** operation's point —
and calls each site **once** on the container, which acts as one
aggregated implementation (as ``fvModels()`` / ``fvConstraints()`` do in
OpenFOAM). From ``pimpleAlgorithm.momentum``:

.. code-block:: python

    from neofoam.framework.model import Extensions
    from .extension import MomentumExtension, momentum_extension


    @pimple.operation(operation_number="2.1")
    def momentum(
        U: volVectorField,
        phi: surfaceScalarField,
        ...,
        ext: Annotated[Extensions[MomentumExtension], momentum_extension],
    ) -> FieldUpdates:
        ext.correct_boundary_velocity(U)
        UEqn = fvVectorMatrix(
            fvm.ddt(U) + fvm.div(phi, U) + viscousStress.divDevReff(U) + ext.terms(U)
        )
        UEqn.relax()
        ext.constrain(UEqn)

Each ``ext.<site>(...)`` call runs the site on every active
implementation in registration order. ``+ ext.terms(U)`` is the **fold
site**: every returned term joins the running sum in that order — ``+``
by default, ``-`` for terms wrapped in
:func:`~neofoam.framework.model.extension.negated` (how the ``fvOptions``
source keeps native's ``== fvOptions(U)`` arithmetic) — and with no
active model the sum passes through untouched. The *order* stays visible
at the call sites: in ``UEqn.H`` the source joins the sum before
relaxation, the constraints are applied after it, and the correction
runs after the solve.

A site whose per-implementation return value the operation must inspect
is iterated explicitly instead — the "did anyone handle it?" pattern
from the continuity operation:

.. code-block:: python

    handled = False
    for extension in ext:
        handled = extension.constrain_pressure(p, U, phiHbyA, rAU) or handled
    if not handled:
        pyf.constrainPressure(p, U, phiHbyA, rAU)

Activation: there is no wiring step
-----------------------------------

An implementation is active **iff its registering model is active for
the case** — i.e. has a live ``ModelRuntime`` in ``ctx.models``. Nothing
else to switch on:

* ``constant/MRFProperties`` present → the ``mrf`` model is detected →
  ``make_mrf_momentum_extension`` runs and its instance shows up in the
  ``ext`` of ``momentum``.
* No ``MRFProperties`` → no MRF runtime → the factory is skipped, the
  sites dispatch to one fewer implementation, and the operation code is
  identical either way.

Matching is by ``ModelSpec`` **identity**, not by name, so a model
instantiated under an instance id still matches its spec.
:meth:`ExtensionPoint.resolve
<neofoam.framework.model.extension.ExtensionPoint.resolve>` runs on
every injection and hands back a fresh ``Extensions`` in registration
(import) order — so the implementations, which wrap live case objects,
never outlive one operation call, and nothing mesh-bound survives
between runs.

Gotcha: no ``from __future__ import annotations``
-------------------------------------------------

The resolver recognises the seam by inspecting the **live annotation
object** (``isinstance(args[1], ExtensionPoint)``). Under
``from __future__ import annotations`` every annotation becomes a
string, the check fails silently, and the parameter falls through to a
``ctx.fields`` lookup by name — the operation gets ``None`` instead of
the extensions, with no error. The same applies to a factory's
``Annotated[Any, "models"]`` parameters.

So: modules that declare a point, register a factory, or consume one
must **not** use ``from __future__ import annotations``. (Framework
internals like ``extension.py`` may, because nothing inspects *their*
annotations at runtime.)

Where to put the implementations
--------------------------------

The MRF and ``fvOptions`` implementations live in the solver's
``pressure_velocity/extension.py``, not in ``neofoam/mrf.py`` /
``neofoam/fv_options.py``. Those two specs are shared with
``incompressibleVoF``, whose frame and source terms differ
(``DDt(rho, U)``, ``fvOptions(rho, U)``) — so the *call sites* belong to
the algorithm that makes them, and each solver ships its own
implementations of its own points. The model spec stays solver-agnostic;
only the wrapper is per-solver.

See also
--------

* :doc:`/auto_how-to/example_use_an_interface` — the single-value fold
  (``@interface`` / ``@contributes``).
* :doc:`/explanation/parameter-injection` — how factory and operation
  parameters are resolved from the ``Context``.
* :doc:`/explanation/model-structure` — what a model owns and how it is
  detected for a case.
