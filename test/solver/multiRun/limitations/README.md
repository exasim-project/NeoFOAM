# Limitations of the region coupling

Tiny multiRun cases of the test-only solver `probe` (`probe_solver.py`): every
region solves one number, `field = offset + sum(gain * received)`, and couples
the patches named in its `gains` with `@coupled` boundary conditions (it sends its
field as the pydantic model `X` or `Y` and receives the partner's). Every region
runs its own graph on its own thread; its `solve_<field>` runs inside its coupling op
`<region>.coupling`, whose loop test hands the region's sides the partner's data,
mapped. `system/regionCoupling.yaml` lists the `exchanges`; exchanges that share a
region form one group, led by its last region. The cases couple parallel (the
default scheme). No mesh, no physics.

Run a case like any multiRun case (the runner only hands multiRun the `probe`
solver, which is not a package under `neofoam.solver`):

    python test/solver/multiRun/limitations/run.py test/solver/multiRun/limitations/cases/pair

In the neofoam_ui DAG stepper: `scripts/probe.sh <case>` of neofoam_ui, through
`probe_ui.py` (`--solver probe_ui:probe`).

`test_limitations.py` pins what each case does; an `xfail(strict=True)` test
states the behaviour we want and the error it fails with now.

| case | math | probes | now |
|---|---|---|---|
| `pair` | a: x = 1 + x_b, b: x = 0.5 x_a → (2, 1) | the baseline | works, 39 passes |
| `chain` | a: x = 1 + x_b, b: x = 0.5 x_a + 0.25 x_c, c: x = 1 + 0.5 x_b → (3, 2, 2) | a ↔ b ↔ c, two exchanges | works, one group led by c, 57 passes (nested loops took a and b 122 solves, c 11) |
| `two_interfaces` | a: x = 1 + 0.5 x_b(p1) + 0.5 x_b(p2), b: x = 0.25 x_a(q1) + 0.25 x_a(q2) → (2, 1) | two exchanges between one pair | works, one loop, 39 passes (nested: 113) |
| `no_coordinates` | as `pair`, `coordinates: false` | a scalar exchange without points | `ValueError` at the first coupling test: `nearest-neighbor` needs coordinates; `mapping: identity` works |
| `renamed_fields` | a: x = 1 + y_b, b: y = 0.5 x_a → (2, 1) | x vs y, one quantity each way | works: a sends `X` and receives `Y`, b the other way round |
| `convergence_typo` | as `pair`, convergence `data: T` | a measure on data nobody sends | `ValueError` when the leader builds its controller |
