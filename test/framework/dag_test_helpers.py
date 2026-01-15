# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Common test helpers and operations for DAG resolver tests."""

from foamadapter.framework.context import Context


# Common test operations
def noop(ctx: Context) -> None:
    """No-op operation."""
    pass


def increment_count(ctx: Context) -> None:
    """Increment count field."""
    ctx.fields["count"] = ctx.fields.get("count", 0) + 1


def set_value(ctx: Context) -> None:
    """Set a value field."""
    ctx.fields["value"] = 42


def double_value(ctx: Context) -> None:
    """Double the value field."""
    ctx.fields["value"] = ctx.fields.get("value", 0) * 2


def add_ten(ctx: Context) -> None:
    """Add 10 to value."""
    ctx.fields["value"] = ctx.fields.get("value", 0) + 10


def multiply_by_three(ctx: Context) -> None:
    """Multiply value by 3."""
    ctx.fields["value"] = ctx.fields.get("value", 0) * 3


# Time loop operations
def set_time_step(ctx: Context) -> None:
    """Set time step."""
    ctx.fields["dt"] = 0.1


def increment_time(ctx: Context) -> None:
    """Increment simulation time."""
    ctx.fields["time"] = ctx.fields.get("time", 0) + ctx.fields.get("dt", 0.1)


def write_output(ctx: Context) -> None:
    """Write output (no-op for tests)."""
    pass


# Iteration loop operations
def initialize_residual(ctx: Context) -> None:
    """Initialize residual."""
    ctx.fields["residual"] = 1.0


def solve_iteration(ctx: Context) -> None:
    """Solve one iteration."""
    ctx.fields["residual"] = ctx.fields.get("residual", 1.0) * 0.5


def check_convergence(ctx: Context) -> None:
    """Check convergence."""
    pass


# Algorithm operations
def momentum(ctx: Context) -> None:
    """Momentum equation."""
    ctx.fields["velocity"] = ctx.fields.get("velocity", 0.0) + 1.0


def continuity(ctx: Context) -> None:
    """Continuity equation. Depends on velocity from momentum."""
    velocity = ctx.fields.get("velocity", 0.0)
    ctx.fields["pressure"] = velocity * 2.0


def pressure_correction(ctx: Context) -> None:
    """Pressure correction."""
    pressure = ctx.fields.get("pressure", 0.0)
    ctx.fields["pressure"] = pressure * 1.1


# Model operations
def update_buoyancy(ctx: Context) -> None:
    """Update buoyancy forces."""
    ctx.fields["buoyancy"] = 9.81


def turbulence_correction(ctx: Context) -> None:
    """Apply turbulence correction. Depends on pressure from continuity."""
    pressure = ctx.fields.get("pressure", 0.0)
    ctx.fields["turbulence"] = pressure * 0.5


def update_properties(ctx: Context) -> None:
    """Update material properties."""
    ctx.fields["density"] = 1000.0


# Loop condition functions
def time_loop_condition(ctx: Context) -> bool:
    """Time loop condition."""
    return ctx.fields.get("time", 0) < ctx.fields.get("end_time", 1)


def inner_loop_condition(ctx: Context) -> bool:
    """Inner iteration loop condition."""
    return ctx.fields.get("iteration", 0) < ctx.fields.get("max_iter", 3)


def convergence_condition(ctx: Context) -> bool:
    """Convergence loop condition."""
    return ctx.fields.get("residual", 1.0) > ctx.fields.get("tolerance", 0.001)
