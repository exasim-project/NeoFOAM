Extend operations from another model
====================================

An operation sometimes has to do something *extra* only when some other
model is active: take the flux relative to a rotating frame, subtract an
``fvOptions`` source, re-find zone faces after a mesh move. Writing that
as ``if "mrf_zones" in ctx.models`` inside the algorithm couples the
algorithm to every optional model that will ever exist. An **extension**
inverts that: the operation module defines *where* it can be extended,
models contribute per hook, and the operation calls whatever the case
activated — without naming a single model.

Extension or model interface?
-----------------------------

One mechanism, two spellings — a declaration whose body combines the
contributions' results is a :class:`~neofoam.framework.model.extension.Hook`
either way; what differs is who owns it:

:class:`~neofoam.framework.model.extension.Extension`
    An *operation* opens seams: several **heterogeneous hooks** — a term
    to add here, a constraint to apply there, a boundary to correct after
    the solve. Defined by the operation module, one extension per
    operation with one ``@defines`` function per hook. The operation
    receives one bound handle and calls each hook once.

``@<model>.interface``
    A *model* gathers values: one hook on a model-private extension
    (:meth:`ModelSpec.interface
    <neofoam.framework.model.spec.ModelSpec.interface>`). The consumer
    annotates a parameter with the handle itself and *calls* the injected
    hook to get the single combined result — see
    :doc:`/auto_how-to/example_use_an_interface`.

Rule of thumb: one gather point combined by one rule wants an interface;
a family of hooks that activate together wants an extension. The
contribution side is the same decorator either way —
``@<model>.contributes(<target>)``.

The worked example on this page is the pressure-velocity seam of the
``incompressibleFluid`` solver: MRF and ``fvOptions`` hook into
``UEqn.H`` / ``pEqn.H`` at a dozen different points, spread over the
``momentum``, ``continuity`` and ``mesh_update`` operations — one
extension each, which is exactly the multi-hook case.

Define the extension next to the operations
-------------------------------------------

The defining module owns one ``Extension`` per extensible operation, and
one ``@defines`` function per hook. The function's leading parameters are
what the operation passes at the call site; a **trailing parameter the
call does not supply** receives the list of active contributions'
results, and the body combines them — it owns the rule (``fold``,
``any``, ``min``, …), exactly like a model interface's body. A
declaration whose every parameter is a call argument is a **broadcast
hook**: the call returns the raw results and the body is never invoked.
Both live next to the operations they serve — in
``solver/incompressibleFluid/models/pressure_velocity/extension.py``:

.. code-block:: python

    from neofoam.framework.model import Extension, fold

    momentum_extension = Extension("momentum")
    pressure_extension = Extension("pressure")


    @momentum_extension.defines
    def correct_boundary_velocity(U: volVectorField) -> None:
        """Set the boundary velocities the coefficients are built from."""


    @momentum_extension.defines
    def terms(U: volVectorField, contributions: list) -> Any:
        """Terms folded into the momentum sum at ``+ ext.terms(U)``."""
        zero_rate = pyf.dimensionedScalar("0", pyf.dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0)
        return fold(fvm.Sp(zero_rate, U), contributions)


    @pressure_extension.defines
    def make_relative(phiHbyA: surfaceScalarField) -> None:
        """Take the predicted flux relative to whatever frame this model adds."""


    @pressure_extension.defines
    def constrain_pressure(p, U, phiHbyA, rAU, handled: list) -> bool:
        """Constrain the pressure boundaries; a contribution returns ``True``
        if it handled them (the operation falls back when none did)."""
        return any(handled)

Three conventions matter here:

* **One extension per operation, not one per module.** ``momentum``,
  ``continuity`` and ``mesh_update`` each define their own, so a
  contribution only ever sees the hooks of the operation it extends — a
  model that acts in one operation (``fvOptions`` has nothing to say
  about a mesh move) simply contributes nowhere else.
* **The declaration body owns the combine rule.** ``terms`` folds every
  contribution onto its seed — the empty momentum source native's
  ``fvOptions(U)`` starts from — with
  :func:`~neofoam.framework.model.extension.fold`, so ``+ ext.terms(U)``
  is well-formed with any number of active contributions, including
  none. ``constrain_pressure`` applies ``any``. The rule is written
  once, in the declaration, not in every consumer.
* **Side-effect hooks stay broadcast.** ``correct_boundary_velocity`` /
  ``make_relative`` list only call arguments: there is nothing to
  combine, the operation ignores the returned results.

The ``Extension`` instance holds only the hook declarations and their
contributions — no ``Context``, no live case objects — so one
module-level instance is safe across runs.

Contribute from a model
-----------------------

A model contributes one plain function per hook it acts at, with
``@<model>.contributes`` (:meth:`ModelSpec.contributes
<neofoam.framework.model.spec.ModelSpec.contributes>`) — the same
decorator used for interfaces. Hooks are reached as attributes of the
extension (``momentum_extension.terms``), so two extensions can share a
hook name. The contributions live next to the model spec they belong to —
in ``neofoam/mrf.py`` / ``neofoam/fv_options.py``:

.. code-block:: python

    from neofoam.solver.incompressibleFluid.models.pressure_velocity.extension import (
        momentum_extension,
        pressure_extension,
    )


    @mrf.contributes(momentum_extension.terms)
    def mrf_frame_acceleration(U: volVectorField, mrf_zones: Annotated[Any, "models"]) -> Any:
        return mrf_zones.DDt(U)


    @mrf.contributes(pressure_extension.make_relative)
    def mrf_make_relative(phiHbyA: surfaceScalarField, mrf_zones: Annotated[Any, "models"]) -> None:
        mrf_zones.makeRelative(phiHbyA)


    @fvOptions.contributes(momentum_extension.terms)
    def fv_options_momentum_source(U: volVectorField, fv_options: Annotated[Any, "models"]) -> Any:
        # Native's ``== fvOptions(U)``: the source joins the sum subtracted.
        return negated(fv_options(U))

A contribution's parameters resolve exactly like an
``@model.operation`` body's, against the **contributing model's** own
runtime plus the live ``Context`` — with one addition:

A parameter named in the hook declaration (``U``, ``phiHbyA``, …)
    Taken from the operation's call — ``ext.terms(U)`` hands ``U``
    to every contribution that names it.

``Annotated[Any, "models"]``
    ``ctx.models[<param name>]`` — here the ``mrf_zones`` object the MRF
    model's ``@build`` step published.

A ``BaseConfig`` subclass
    Pulled by type from the *contributing* model's config, not from
    whoever consumes the extension.

A bare name
    ``ctx.fields[<param name>]``; ``Depends(...)`` markers work too.

A parameter nothing supplies raises a ``ValueError`` naming the hook,
the contribution, and the parameter — a misconfigured model can never
silently drop out.

Consume it in an operation
--------------------------

The consuming operation adds one parameter annotated
``Annotated[BoundExtension, <extension>]`` — its **own** operation's
extension — and calls each hook **once** on the handle (as
``fvModels()`` / ``fvConstraints()`` are called in OpenFOAM). From
``pimpleAlgorithm.momentum``:

.. code-block:: python

    from neofoam.framework.model import BoundExtension
    from .extension import momentum_extension


    @pimple.operation(operation_number="2.1")
    def momentum(
        U: volVectorField,
        phi: surfaceScalarField,
        ...,
        ext: Annotated[BoundExtension, momentum_extension],
    ) -> FieldUpdates:
        ext.correct_boundary_velocity(U)
        UEqn = fvVectorMatrix(
            fvm.ddt(U) + fvm.div(phi, U) + viscousStress.divDevReff(U) + ext.terms(U)
        )
        UEqn.relax()
        ext.constrain(UEqn)

Each ``ext.<hook>(...)`` call runs every active contribution in
registration order and hands the results to the declaration body.
``+ ext.terms(U)`` joins the body's fold — ``+`` by default, ``-`` for
terms wrapped in :func:`~neofoam.framework.model.extension.negated` (how
the ``fvOptions`` source keeps native's ``== fvOptions(U)`` arithmetic).
The *order* stays visible at the call sites: in ``UEqn.H`` the source
joins the sum before relaxation, the constraints are applied after it,
and the correction runs after the solve.

Because the declaration owns the combine rule, the consumer just acts on
the combined value — from the continuity operation:

.. code-block:: python

    if not ext.constrain_pressure(p, U, phiHbyA, rAU):   # declaration: any(handled)
        pyf.constrainPressure(p, U, phiHbyA, rAU)

Activation: there is no wiring step
-----------------------------------

A contribution is active **iff its contributing model is active for
the case** — i.e. has a live ``ModelRuntime`` in ``ctx.models``. Nothing
else to switch on:

* ``constant/MRFProperties`` present → the ``mrf`` model is detected →
  ``mrf_frame_acceleration`` folds into the ``ext.terms(U)`` of
  ``momentum``.
* No ``MRFProperties`` → no MRF runtime → the contribution is skipped,
  the hooks dispatch to one fewer contribution, and the operation code
  is identical either way.

Matching is by ``ModelSpec`` **identity**, not by name, so a model
instantiated under an instance id still matches its spec.
:meth:`Extension.resolve
<neofoam.framework.model.extension.Extension.resolve>` runs on every
injection and hands back a fresh
:class:`~neofoam.framework.model.extension.BoundExtension` — so nothing
mesh-bound survives between runs.

Gotcha: no ``from __future__ import annotations``
-------------------------------------------------

The resolver recognises the seam by inspecting the **live annotation
object** (``isinstance(args[1], Extension)``). Under
``from __future__ import annotations`` every annotation becomes a
string, the check fails silently, and the parameter falls through to a
``ctx.fields`` lookup by name — the operation gets ``None`` instead of
the extension, with no error. The same applies to a contribution's
``Annotated[Any, "models"]`` parameters.

So: modules that define an extension, contribute to one, or consume one
must **not** use ``from __future__ import annotations``. (Framework
internals like ``extension.py`` may, because nothing inspects *their*
annotations at runtime.)

Where to put the contributions
------------------------------

The extension *definitions* live with the operations they extend
(``pressure_velocity/extension.py``); the *contributions* live with the
model spec they belong to (``neofoam/mrf.py``, ``neofoam/fv_options.py``)
— everything the MRF model does to a case reads in one module. Two
consequences of that placement:

* The extension import in ``mrf.py`` sits **below** the spec definition.
  The solver package imports ``neofoam.mrf`` back to register the spec,
  so a top-of-file import of the solver's extension module would re-enter
  ``mrf.py`` before ``mrf`` exists — a circular import.
* The MRF and ``fvOptions`` specs are shared with ``incompressibleVoF``,
  whose frame and source terms differ (``DDt(rho, U)``,
  ``fvOptions(rho, U)``). When VoF's operations define their own
  extensions, those contributions join the same modules — one set per
  solver's extension, side by side under the one spec.

See also
--------

* :doc:`/auto_how-to/example_use_an_interface` — the single-value fold
  (``@interface`` / ``@contributes``).
* :doc:`/explanation/parameter-injection` — how contribution and
  operation parameters are resolved from the ``Context``.
* :doc:`/explanation/model-structure` — what a model owns and how it is
  detected for a case.
