# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent NeoN worker of the two-phase parity tests, for one staged case and executor."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Callable

import neon._neon as nn
import numpy as np
import pybFoam as pyf
from _neon_parity_worker import Op, _serve, _to_numpy, _with_boundary

from neofoam import neofoam_bindings as nfb

_SCALAR_FIELDS = ("T", "rAU", "alpha.water", "p_rgh")


def serve(case_dir: Path, executor: str) -> None:
    """Read the case once, then serve requests."""
    os.chdir(case_dir)
    # pybFoam holds raw references, and every NeoN object must die with this frame, before
    # main finalizes Kokkos
    arg_list = pyf.argList(["neonTwoPhaseParityWorker"])
    run_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(run_time, executor)

    fields: dict[str, Any] = {
        "U": nfb.read_vector_volume_field(rt, "U"),
        "phi": nfb.create_phi(rt, "U"),
    }
    for name in _SCALAR_FIELDS:
        fields[name] = nfb.read_scalar_volume_field(rt, name)

    def reload_scalar(name: str) -> None:
        fields[name] = nfb.read_scalar_volume_field(rt, name)

    def reload_velocity() -> None:
        fields["U"] = nfb.read_vector_volume_field(rt, "U")

    def update_flux() -> None:
        fields["phi"] = nfb.create_phi(rt, "U")

    def stale_buffer() -> Any:
        """A face field of stale non-zero values for an ``out=`` variant to overwrite."""
        return fields["phi"] * 7.0

    props = nfb.read_two_phase_transport_properties(rt)
    gh = nfb.create_gh(rt)
    ghf = nfb.create_ghf(rt)

    def into_or_new(data: Any, compute: Callable[..., Any]) -> Any:
        """compute() as a new field, or with ``data[0]`` set into a stale buffer via out=."""
        if data is None or not bool(data[0]):
            return compute()
        out = stale_buffer()
        compute(out=out)
        return out

    def rho() -> Any:
        result = nfb.create_uniform_volume_field(rt, "rho", 7.0)
        nfb.update_mixture_density(fields["alpha.water"], props["rho1"], props["rho2"], result)
        return result

    def nu() -> Any:
        result = nfb.create_uniform_volume_field(rt, "nuMixture", 7.0)
        nfb.update_mixture_viscosity(
            fields["alpha.water"],
            props["rho1"],
            props["nu1"],
            props["rho2"],
            props["nu2"],
            result,
        )
        return result

    def product() -> Any:
        result = nfb.create_uniform_volume_field(rt, "TrAU", 7.0)
        nfb.multiply(fields["T"], fields["rAU"], result)
        return result

    def rho_phi() -> Any:
        result = stale_buffer()
        nfb.multiply(nfb.linear_interpolate(rt, rho()), fields["phi"], result)
        return result

    def rauf() -> Any:
        return nfb.linear_interpolate(rt, fields["rAU"])

    def gravity_face_force(data: Any = None) -> Any:
        density = rho()
        return into_or_new(data, lambda **out: nfb.gravity_face_force(rt, ghf, density, **out))

    def phig(data: Any = None) -> Any:
        force = gravity_face_force()
        return into_or_new(data, lambda **out: nfb.buoyancy_flux(force, rauf(), **out))

    def constrain_pressure() -> Any:
        nfb.constrain_pressure(fields["p_rgh"], fields["U"], fields["phi"] + phig(), rauf())
        return fields["p_rgh"]

    def momentum_flux(data: Any) -> Any:
        constrain_pressure()
        force = gravity_face_force()
        return into_or_new(
            data, lambda **out: nfb.buoyant_momentum_flux(rt, force, fields["p_rgh"], **out)
        )

    def static_pressure() -> Any:
        result = nfb.create_uniform_volume_field(rt, "p", 7.0)
        nfb.update_static_pressure(fields["p_rgh"], rho(), gh, result)
        return result

    def reconstructed() -> np.ndarray:
        result = nfb.read_vector_volume_field(rt, "U")
        nfb.reconstruct(fields["phi"], result)
        return np.concatenate(
            [_to_numpy(result.internal_vector()), _to_numpy(nfb.vector_boundary_values(result))]
        )

    def buoyant_velocity() -> np.ndarray:
        result = nfb.read_vector_volume_field(rt, "U")
        nfb.update_velocity_buoyant(
            fields["U"], fields["rAU"], phig(), fields["phi"], rauf(), result
        )
        return _to_numpy(result.internal_vector())

    alpha_water = ("alpha.water",)
    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("twoPhase.properties", ()): lambda d: np.array(
            [props[k] for k in ("rho1", "nu1", "rho2", "nu2", "sigma")]
        ),
        ("twoPhase.phases", ()): lambda d: np.array([props["phase1"], props["phase2"]]),
        ("twoPhase.density", alpha_water): lambda d: _with_boundary(rho()),
        ("twoPhase.viscosity", alpha_water): lambda d: _with_boundary(nu()),
        ("twoPhase.rhoPhi", alpha_water): lambda d: _with_boundary(rho_phi()),
        ("twoPhase.interpolate", ("rAU",)): lambda d: _with_boundary(
            into_or_new(d, lambda **out: nfb.linear_interpolate(rt, fields["rAU"], **out))
        ),
        ("twoPhase.multiply", ("T", "rAU")): lambda d: _with_boundary(product()),
        ("twoPhase.gh", ()): lambda d: _with_boundary(gh),
        ("twoPhase.ghf", ()): lambda d: _with_boundary(ghf),
        ("twoPhase.staticPressure", alpha_water): lambda d: _with_boundary(static_pressure()),
        ("buoyancy.gravityFaceForce", alpha_water): lambda d: _with_boundary(gravity_face_force(d)),
        ("buoyancy.flux", alpha_water): lambda d: _with_boundary(phig(d)),
        ("buoyancy.constrainPressure", alpha_water): lambda d: _to_numpy(
            constrain_pressure().boundary_data_value()
        ),
        ("buoyancy.momentumFlux", alpha_water): lambda d: _with_boundary(momentum_flux(d)),
        ("buoyancy.reconstruct", ("phi",)): lambda d: reconstructed(),
        ("buoyancy.velocity", alpha_water): lambda d: buoyant_velocity(),
        ("field.reload", ("U",)): lambda d: reload_velocity(),
        ("flux.update", ("U",)): lambda d: update_flux(),
    }
    for name in _SCALAR_FIELDS:
        ops[("field.reload", (name,))] = lambda d, name=name: reload_scalar(name)
    _serve(ops)


def main() -> None:
    nn.initialize(["neonTwoPhaseParityWorker"])
    serve(Path(sys.argv[1]).resolve(), sys.argv[2])
    gc.collect()  # drop any cyclic refs holding Kokkos views before finalize
    nn.finalize()


if __name__ == "__main__":
    main()
