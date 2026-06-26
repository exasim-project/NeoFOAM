# Specification: the `Interface` spec — a framework mechanism for model-gathered contributions

Status: draft · Scope: a new framework spec type **`Interface`** in
`neofoam.framework`, parallel to `Model`/`ModelSpec`. Decision context:
`plans/context-injected-interfaces.md` (approach B). First consumers:
`plans/timestep-constraints.md` (time-step constraints + loop conditions) and,
later, equation source terms (fvOptions-style).

## 1. Motivation

Several places in the framework follow the same shape: **many models each contribute
a piece, and one consumer reads the combined result** — time-step constraints (fold
`min`), loop stop-conditions (fold `all`), equation source terms (sum into a matrix),
post-processing hooks, etc. Today each is wired ad-hoc (e.g. the CFL `set_time_step`
hardcodes `min`, name-matches `measurement_provider.*`, and needs
`install_constraints_step` + hand-written `depends_on`).

`Interface` makes that shape a **first-class spec, like `Model`**: declare a gather
point once; any model contributes to it with an operation-style function; a consumer
**injects** it (like a field) and **calls** it to read the fold. It reuses the existing
operations dependency-injection + init-graph, so contributions get build-time
dependency checking for free.

## 2. Requirements

Priority: **MUST** (essential), **SHOULD** (important), **COULD** (optional). Each
statement is atomic and verifiable.

### Defining an interface
- **IF1 (MUST)** `Interface(name: str)` returns a new `InterfaceSpec`, the way
  `Model(name)` returns a `ModelSpec`.
- **IF2 (MUST)** An `InterfaceSpec` is generic in its contribution type — `Interface[T]`.
- **IF3 (MUST)** Exactly one aggregation is registered per interface, via an
  `@<iface>.combine`-decorated function of signature `(Iterable[T]) -> T`.
- **IF4 (MUST)** The factory accepts no `combine=`/`empty=` keyword arguments; the fold
  lives only in the `@combine` function.
- **IF5 (MUST)** `@combine` is invoked with the active contributions' results; when none
  are active it is invoked with an empty iterable and its return value is the interface
  result.
- **IF6 (SHOULD)** Registering a second `@<iface>.combine` raises at registration time.

### Contributing
- **IF7 (MUST)** A contribution is registered via an `@<iface>.contribute`-decorated
  function.
- **IF8 (MUST)** A contribution's parameters are dependency-injected by the **same
  resolver used by `@model.operation`** (fields, models, configs, registered
  injectables).
- **IF9 (MUST)** A contribution function does not declare or receive a `Context`
  parameter.
- **IF10 (MUST)** A contribution returns a value of type `T`.
- **IF11 (MUST)** A contribution's build dependencies are inferred from its parameter
  list; the author writes no `depends_on`.
- **IF12 (MUST)** If a contribution declares a parameter no graph node produces, the
  build fails with an explicit error (never a silent skip or default).
- **IF13 (MUST)** A contribution is folded iff its owning model is active (existing
  config-presence / `detect` selection); the interface adds no separate activation.

### Consuming — an interface is like a field in the `Context`
- **IF14 (MUST)** During init the active interface is placed in the `Context` as an
  entry analogous to a field (`ctx.interfaces[name]`), created by the init-graph like
  fields/models.
- **IF15 (MUST)** A consumer obtains the interface by the **same injection mechanism
  used for fields** — declaring an operation parameter **typed with the interface spec
  itself** (the spec is the injection key, not the generic `Interface[T]`).
- **IF16 (MUST)** **Calling the injected interface** (`iface()`) returns the `@combine`
  over the currently active contributions.
- **IF17 (SHOULD)** `spec.collect(ctx)` provides the same fold for code that is not an
  injected operation (e.g. inside an `@build`).

### Cross-cutting constraints
- **IF18 (MUST)** `Interface` is implemented on the existing `PluginSystem` / staged
  init-graph / config-injection resolver; it adds no parallel registry.
- **IF19 (MUST)** Adding or removing a contribution requires no edit to the interface
  definition or to any consumer (OCP).
- **IF20 (MUST)** The `Interface` spec module is pure-Python; only contributions import
  backend (pybFoam) types (DIP).
- **IF21 (SHOULD)** One unchanged `Interface` mechanism supports at least two distinct
  domains — a `min`-folded `Interface[float]` and a `sum`-folded `Interface[Matrix]`.
- **IF22 (COULD)** `Interface[T]` lets mypy check that contributions return `T` and that
  calling the interface / `collect(ctx)` yields `T`.

## 3. API (the contract)

```python
from neofoam.framework import Interface

# define like a ModelSpec — Interface(name) -> InterfaceSpec (generic in T):
timeStepConstraint = Interface("timeStepConstraint")     # Interface[float]

# register the aggregation (the fold, incl. empty case) with @<iface>.combine:
@timeStepConstraint.combine
def combine(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)

# contribute (operation-style; deps inferred from params; no ctx, no depends_on):
@timeStepConstraint.contribute
def courant(phi: surfaceScalarField, deltaT: DeltaT, cfg: CourantConfig) -> float: ...

# consume — inject the interface by typing the parameter with the spec; call it for the fold:
@someModel.operation
def use(self, constraints: timeStepConstraint) -> None:   # annotated with the spec (the key)
    self.dt = constraints()                                # calling the injected interface -> fold

# outside an operation (e.g. in @build) — explicit read:
#   dt = timeStepConstraint.collect(ctx)
```

Surface:
- `Interface(name) -> Interface[T]` — define a gather point (parallel to `Model(name)`).
- `@iface.combine` over `(Iterable[T]) -> T` — register the fold (incl. the empty case).
- `@iface.contribute` — register an operation-style contribution (returns the fn).
- consumer injection — the active interface is injected into an operation **like a
  field**, by typing the parameter with the **interface spec** (the key).
- calling the injected interface (`constraints()`) — folded result over active contributions.
- `spec.collect(ctx)` — the same fold for non-operation code (e.g. `@build`).

**Toy / cross-domain example (proves reuse):**

```python
momentumSource = Interface("momentumSource")

@momentumSource.combine
def combine(terms: Iterable[Matrix]) -> Matrix:
    return sum(terms, ZERO)

@momentumSource.contribute
def porous_drag(U: volVectorField, cfg: PorosityConfig): return -cfg.D * U

@pimple.operation
def assemble_UEqn(self, ueqn, sources: momentumSource):  # typed with the spec; injected like a field
    ueqn += sources()                                    # calling -> sum of active source terms
```

## 4. Acceptance criteria

Each maps to a requirement and is independently testable.
- **IF1–IF5:** `x = Interface("x")` + an `@x.combine` returning `min(limits, default=VGREAT)`
  yields an `Interface[float]`; `x.collect(ctx)` with two active contributions returns
  the min; with none returns `VGREAT` (IF5).
- **IF6:** a second `@x.combine` raises at registration.
- **IF8/IF9/IF11:** a contribution injects `phi: surfaceScalarField` + a config (no
  `ctx`, no `depends_on`) and is called with them resolved — parity with a
  `@model.operation` (shared-resolver test).
- **IF12:** a contribution declaring an unproduced parameter fails the build with an
  explicit error (not a silent skip).
- **IF13:** a contribution whose owning model is inactive is excluded from the fold.
- **IF14/IF15/IF16:** a consumer operation declaring a parameter typed with the spec `x`
  receives the active interface (from `ctx.interfaces["x"]`); **calling it** (`x()`)
  equals the fold.
- **IF19/IF21:** a second domain — a `sum`-folded `Interface[Matrix]` with a contributor
  — works with no change to the mechanism, and adding it touches no consumer.
- **IF22:** mypy flags a contribution that returns the wrong type for `Interface[T]`.

## 5. Out of scope / open questions

- **Injection key (IF15).** Decided: bind by the **interface spec object** used as the
  parameter annotation (resolver matches `param.annotation is <spec>`, like the
  existing `is Context` match) — unambiguous even when interfaces share a return type.
  Open: how mypy treats a spec-instance-as-annotation (a typing shim may be needed).
- **Context placement (IF14).** Exact `Context` surface for interfaces
  (`ctx.interfaces[name]`) and where `Interface` specs are declared (module convention).
- **`collect(ctx)` signature (IF17).** Confirm a pure fold over active contributions
  suffices for all consumers (incl. source terms); no consumer-supplied args needed.
- **Ordering.** Whether contribution order matters for any fold (it must not for
  `min`/`all`/`sum`); if a future fold is order-sensitive, define the order source.
- Marker types for non-field injectables (e.g. a loop scalar) are an application
  concern — specified by each consumer, see `plans/timestep-constraints.md`.
