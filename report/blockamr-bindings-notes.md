# blockAmr binding notes

Rationale extracted from the nanobind binding sources under `src/blockAmr/bindings/` so the
sources can stay readable. Each heading is referenced from a one-line pointer in the code.

## mlmg-smoothing

`linop.cpp` binds MLMG's own V-cycle smoothing counts — `nu1`/`nu2`/`nuf`/`nub`, via
`set_pre_smooth` / `set_post_smooth` / `set_final_smooth` / `set_bottom_smooth`. They are the
direct counterparts of `gmg_pre_sweeps` / `gmg_post_sweeps` / `gmg_coarsest_sweeps` on the native
side.

They are bound so MLMG can be tuned on the same axes we tune ourselves on. Without them, any
"faster than MLMG" number is a tuned solver measured against a stock one.

AMReX defaults: `nu1 = nu2 = 2` (the same 2+2 we default to), `nuf = 8`, `nub = 0`. `nuf` applies
only when the SMOOTHER is the bottom solver, so it does nothing under `set_bottom_solver("cg")`.

## rtol-norm

`set_always_use_bnorm` measures the relative tolerance against `||b||` always, rather than against
`max(||b||, ||r0||)`. Which of the two is used changes what `rtol` MEANS, and therefore any
iteration count compared against another solver's.
