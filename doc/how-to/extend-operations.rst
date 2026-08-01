Extend operations from another model
====================================

An operation sometimes has to do something *extra* only when some other
model is active: take the flux relative to a rotating frame, subtract an
``fvOptions`` source, re-find zone faces after a mesh move. Writing that
as ``if "mrf_zones" in ctx.models`` inside the algorithm couples the
algorithm to every optional model that will ever exist. An **extension**
inverts that: the operation module defines *where* it can be extended,
models contribute per site, and the operation calls whatever the case
activated — without naming a single model.

Extension or model interface?
-----------------------------

Two seams exist, and picking the wrong one is the usual mistake:

:class:`~neofoam.framework.model.extension.Extension`
    Several **heterogeneous sites** — a term to add here, a constraint to
    apply there, a boundary to correct after the solve. Defined by the
    *operation module*, one extension per operation with one
    ``@defines`` function per site. The operation receives one bound
    handle and calls each site once.

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

Rule of thumb: one gather point folded by one rule wants an interface; a
family of hook points that activate together wants an extension. The
contribution side is the same decorator either way —
``@<model>.contributes(<target>)``.

The worked example on this page is the pressure-velocity seam of the
``incompressibleFluid`` solver: MRF and ``fvOptions`` hook into
``UEqn.H`` / ``pEqn.H`` at a dozen different sites, spread over the
``momentum``, ``continuity`` and ``mesh_update`` operations — one
extension each, which is exactly the multi-site case.

Define the extension next to the operations
-------------------------------------------

The defining module owns one ``Extension`` per extensible operation, and
one ``@defines`` function per site. The function's parameters are what
the operation passes at the call site; its body produces the **site
default** — ``None`` for a broadcast site, or the seed value every
contribution folds onto for a term site. Both live next to the
operations they serve — in
``solver/incompressibleFluid/models/pressure_velocity/extension.py``:

.. code-block:: python

    from neofoam.framework.model import Extension

    momentum_extension = Extension("momentum")
    pressure_extension = Extension("pressure")


    @momentum_extension.defines
    def correct_boundary_velocity(U: volVectorField) -> None:
        """Set the boundary velocities the coefficients are built from."""


    @momentum_extension.defines
    def terms(U: volVectorField) -> Any:
        """Terms folded into the momentum sum at ``+ ext.terms(U)``."""
        zero_rate = pyf.dimensionedScalar("0", pyf.dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0)
        return fvm.Sp(zero_rate, U)


    @pressure_extension.defines
    def make_relative(phiHbyA: surfaceScalarField) -> None:
        """Take the predicted flux relative to whatever frame this model adds."""


    @pressure_extension.defines
    def constrain_pressure(p, U, phiHbyA, rAU) -> None:
        """Constrain the pressure boundaries; a contribution returns ``True``
        if it handled them."""

Three conventions matter here:

* **One extension per operation, not one per module.** ``momentum``,
  ``continuity`` and ``mesh_update`` each define their own, so a
  contribution only ever sees the sites of the operation it extends — a
  model that acts in one operation (``fvOptions`` has nothing to say
  about a mesh move) simply contributes nowhere else.
* **The declaration body is the site default.** A term site returns its
  seed — here the empty momentum source native's ``fvOptions(U)`` starts
  from — so ``+ ext.terms(U)`` is well-formed with any number of active
  contributions, including none. A broadcast site's body is empty
  (``None``): the call returns the raw per-contribution results.
* **A site that reports back returns a value the operation can act
  on.** ``constrain_pressure`` contributions return ``True`` when they
  handled the constraint, which lets the operation fall back to the
  plain call when nothing did (see below).

The ``Extension`` instance holds only the site declarations and their
contributions — no ``Context``, no live case objects — so one
module-level instance is safe across runs.

Contribute from a model
-----------------------

A model contributes one plain function per site it acts at, with
``@<model>.contributes`` (:meth:`ModelSpec.contributes
<neofoam.framework.model.spec.ModelSpec.contributes>`) — the same
decorator used for interfaces. Sites are reached as attributes of the
extension (``momentum_extension.terms``), so two extensions can share a
site name:

.. code-block:: python

    from neofoam.fv_options import fvOptions
    from neofoam.mrf import mrf


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

A parameter named in the site declaration (``U``, ``phiHbyA``, …)
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

A parameter nothing supplies raises a ``ValueError`` naming the site,
the contribution, and the parameter — a misconfigured model can never
silently drop out.

Consume it in an operation
--------------------------

The consuming operation adds one parameter annotated
``Annotated[BoundExtension, <extension>]`` — its **own** operation's
extension — and calls each site **once** on the handle (as
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

Each ``ext.<site>(...)`` call runs every active contribution in
registration order. ``+ ext.terms(U)`` is the **fold site**: the site
returns its seed with every contribution's term folded on in that order
— ``+`` by default, ``-`` for terms wrapped in
:func:`~neofoam.framework.model.extension.negated` (how the ``fvOptions``
source keeps native's ``== fvOptions(U)`` arithmetic). The *order* stays
visible at the call sites: in ``UEqn.H`` the source joins the sum before
relaxation, the constraints are applied after it, and the correction
runs after the solve.

A broadcast site returns the raw per-contribution results, so a
"did anyone handle it?" site is one ``any``  — from the continuity
operation:

.. code-block:: python

    if not any(ext.constrain_pressure(p, U, phiHbyA, rAU)):
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
  the sites dispatch to one fewer contribution, and the operation code
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

The MRF and ``fvOptions`` contributions live in the solver's
``pressure_velocity/extension.py``, not in ``neofoam/mrf.py`` /
``neofoam/fv_options.py``. Those two specs are shared with
``incompressibleVoF``, whose frame and source terms differ
(``DDt(rho, U)``, ``fvOptions(rho, U)``) — so the *call sites* belong to
the algorithm that makes them, and each solver ships its own
contributions to its own extensions. The model spec stays
solver-agnostic; only the contribution is per-solver.

See also
--------

* :doc:`/auto_how-to/example_use_an_interface` — the single-value fold
  (``@interface`` / ``@contributes``).
* :doc:`/explanation/parameter-injection` — how contribution and
  operation parameters are resolved from the ``Context``.
* :doc:`/explanation/model-structure` — what a model owns and how it is
  detected for a case.
