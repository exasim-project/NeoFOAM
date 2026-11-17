# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent pybFoam worker of the two-phase parity tests, for one staged case."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf
from _of_parity_worker import Op, _generate_mesh, _serve, _set_field
from pybFoam import fvc, multiphase, surfaceScalarField, volScalarField, volVectorField

# The cases' patches that carry faces, in their mesh (and NeoN face) order; frontAndBack is empty.
_PATCHES = ("walls", "atmosphere")


def _with_patches(mesh: Any, field: Any) -> np.ndarray:
    """Internal then boundary values of every non-empty patch, in patch order."""
    names = [name for name in _PATCHES if mesh.boundary().findPatchID(name) >= 0]
    return np.concatenate(
        [np.asarray(field.internalField())] + [np.asarray(field[name]) for name in names]
    )


def _corrected_patches(mesh: Any, field: Any) -> np.ndarray:
    """The boundary values after ``correctBoundaryConditions`` with the current ``phi``."""
    field.correctBoundaryConditions()
    return _with_patches(mesh, field)[mesh.nCells() :]


def _gravity_head(mesh: Any) -> tuple[Any, Any]:
    """gh.H's (gh, ghf), with ghRef = g & (cmptMag(g)/mag(g))*hRef in OpenFOAM's order."""
    g = pyf.uniformDimensionedVectorField(mesh, "g")
    h_ref = pyf.uniformDimensionedScalarField(mesh, "hRef").value()
    gv = [g.value()[i] for i in range(3)]
    mag_g = np.sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2])
    scaled = [(abs(c) / mag_g) * h_ref for c in gv]
    gh_ref = pyf.dimensionedScalar(
        "ghRef",
        g.dimensions() * pyf.dimLength,
        gv[0] * scaled[0] + gv[1] * scaled[1] + gv[2] * scaled[2],
    )
    return (
        volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref),
        surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref),
    )


class _TwoPhase:
    """interFoam's mixture and buoyancy terms on the case's alpha.water and p_rgh."""

    def __init__(self, mesh: Any, fields: dict[str, Any]):
        self.mesh = mesh
        self.t, self.u, self.phi, self.rau = (fields[n] for n in ("T", "U", "phi", "rAU"))
        u, phi = self.u, self.phi
        self.mixture = multiphase.immiscibleIncompressibleTwoPhaseMixture(u, phi)
        self.alpha1 = self.mixture.alpha1()
        self.p_rgh = volScalarField.read_field(mesh, "p_rgh")
        self.gh, self.ghf = _gravity_head(mesh)

    def rho(self) -> Any:
        alpha2 = volScalarField(pyf.Word("alpha2"), -self.alpha1 + 1.0)
        return volScalarField(
            pyf.Word("rho"),
            self.alpha1 * self.mixture.rho1() + alpha2 * self.mixture.rho2(),
        )

    def nu(self) -> Any:
        self.mixture.correct()
        return volScalarField(pyf.Word("nuMixture"), self.mixture.nu())

    def rho_phi(self) -> Any:
        return surfaceScalarField(pyf.Word("rhoPhi"), fvc.interpolate(self.rho()) * self.phi)

    def n_hat_f(self) -> Any:
        self.mixture.correct()
        return self.mixture.nHatf()

    def curvature(self) -> Any:
        return volScalarField(pyf.Word("K"), -fvc.div(self.n_hat_f()))

    def surface_tension_force(self) -> Any:
        self.mixture.correct()
        return surfaceScalarField(pyf.Word("stf"), self.mixture.surfaceTensionForce())

    def rauf(self) -> Any:
        return surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(self.rau))

    def gravity_face_force(self) -> Any:
        """surfaceTensionForce() - ghf*snGrad(rho) without surface tension."""
        rho = self.rho()
        ghf_sn_grad = surfaceScalarField(pyf.Word("ghfSnGradRho"), self.ghf * fvc.snGrad(rho))
        return surfaceScalarField(pyf.Word("gravityForce"), -ghf_sn_grad)

    def phig(self) -> Any:
        return surfaceScalarField(
            pyf.Word("phig"), self.gravity_face_force() * self.rauf() * self.mesh.magSf()
        )

    def constrain_pressure(self) -> Any:
        """pEqn.H's constrainPressure for phiHbyA = phi + phig, then the patch evaluation."""
        # the cases' rAU is in seconds; a unit density matches the dimensions, exactly
        unit_density = pyf.dimensionedScalar("unitDensity", pyf.dimDensity, 1.0)
        phi_hbya = surfaceScalarField(pyf.Word("phiHbyA"), self.phi * unit_density + self.phig())
        pyf.constrainPressure(self.p_rgh, self.u, phi_hbya, self.rauf())
        self.p_rgh.correctBoundaryConditions()
        return self.p_rgh

    def momentum_flux(self) -> Any:
        """UEqn.H's reconstruct argument, after constrainPressure set the wall gradient."""
        self.constrain_pressure()
        return surfaceScalarField(
            pyf.Word("momentumFlux"),
            (self.gravity_face_force() - fvc.snGrad(self.p_rgh)) * self.mesh.magSf(),
        )

    def buoyant_velocity(self) -> np.ndarray:
        """U + rAU*reconstruct((phig - phi)/rAUf): HbyA = U and phi as the pressure flux."""
        # The cases' rAU is in seconds, so the dimensions do not close: the face flux gets a unit
        # density (exact) and the cell update U + rAU*r is formed in numpy, the same two roundings.
        unit_density = pyf.dimensionedScalar("unitDensity", pyf.dimDensity, 1.0)
        reconstructed = volVectorField(
            pyf.Word("reconstructed"),
            fvc.reconstruct((self.phig() - self.phi * unit_density) / self.rauf()),
        )
        rau = np.asarray(self.rau.internalField())
        u = np.asarray(self.u.internalField())
        return u + rau[:, None] * np.asarray(reconstructed.internalField())

    def static_pressure(self) -> Any:
        return volScalarField(pyf.Word("p"), self.p_rgh + self.rho() * self.gh)

    def set_p_rgh(self, values: np.ndarray | None) -> None:
        """Set p_rgh; its fixedFluxPressure walls need a gradient before they can be evaluated."""
        assert values is not None
        np.asarray(self.p_rgh.internalField())[:] = values
        zero_wall_flux = surfaceScalarField(pyf.Word("zeroWallFlux"), fvc.flux(self.u))
        pyf.constrainPressure(self.p_rgh, self.u, zero_wall_flux, self.rauf())
        self.p_rgh.correctBoundaryConditions()
        pyf.write(self.p_rgh)

    def ops(self) -> dict[tuple[str, tuple[str, ...]], Op]:
        mesh = self.mesh
        alpha = ("alpha.water",)
        return {
            ("field.get", alpha): lambda d: np.array(self.alpha1.internalField()),
            ("field.set", alpha): lambda d: _set_field(self.alpha1, d),
            ("field.get", ("p_rgh",)): lambda d: np.array(self.p_rgh.internalField()),
            ("field.set", ("p_rgh",)): lambda d: self.set_p_rgh(d),
            ("twoPhase.density", alpha): lambda d: _with_patches(mesh, self.rho()),
            ("twoPhase.viscosity", alpha): lambda d: _with_patches(mesh, self.nu()),
            ("twoPhase.rhoPhi", alpha): lambda d: _with_patches(mesh, self.rho_phi()),
            ("twoPhase.interpolate", ("rAU",)): lambda d: _with_patches(mesh, self.rauf()),
            ("twoPhase.multiply", ("T", "rAU")): lambda d: _with_patches(
                mesh, volScalarField(pyf.Word("TrAU"), self.t * self.rau)
            ),
            ("twoPhase.gh", ()): lambda d: _with_patches(mesh, self.gh),
            ("twoPhase.ghf", ()): lambda d: _with_patches(mesh, self.ghf),
            ("twoPhase.staticPressure", alpha): lambda d: _with_patches(
                mesh, self.static_pressure()
            ),
            ("buoyancy.gravityFaceForce", alpha): lambda d: _with_patches(
                mesh, self.gravity_face_force()
            ),
            ("buoyancy.flux", alpha): lambda d: _with_patches(mesh, self.phig()),
            ("interface.nHatf", alpha): lambda d: _with_patches(mesh, self.n_hat_f()),
            ("interface.curvature", alpha): lambda d: _with_patches(mesh, self.curvature()),
            ("interface.surfaceTensionForce", alpha): lambda d: _with_patches(
                mesh, self.surface_tension_force()
            ),
            ("buoyancy.constrainPressure", alpha): lambda d: _with_patches(
                mesh, self.constrain_pressure()
            )[mesh.nCells() :],
            ("buoyancy.momentumFlux", alpha): lambda d: _with_patches(mesh, self.momentum_flux()),
            ("buoyancy.reconstruct", ("phi",)): lambda d: _with_patches(
                mesh, volVectorField(pyf.Word("reconstructPhi"), fvc.reconstruct(self.phi))
            ),
            ("buoyancy.velocity", alpha): lambda d: self.buoyant_velocity(),
        }


def run(case_dir: Path) -> None:
    """Generate the mesh, read the fields, then serve requests."""
    os.chdir(case_dir)
    # pybFoam holds raw references across these objects: they stay in this frame while serving
    arg_list = pyf.argList(["ofTwoPhaseParityWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    mesh = pyf.fvMesh(runtime)

    t = volScalarField.read_field(mesh, "T")
    u = volVectorField.read_field(mesh, "U")
    rau = volScalarField.read_field(mesh, "rAU")
    inlet_outlet = volScalarField.read_field(mesh, "alphaInletOutlet")
    phi = pyf.createPhi(u)
    centres = np.array(mesh.C().internalField())
    volumes = np.array(mesh.V())

    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("mesh.C", ()): lambda d: centres,
        ("mesh.V", ()): lambda d: volumes,
        ("field.get", ("T",)): lambda d: np.array(t.internalField()),
        ("field.get", ("U",)): lambda d: np.array(u.internalField()),
        ("field.get", ("rAU",)): lambda d: np.array(rau.internalField()),
        ("field.set", ("T",)): lambda d: _set_field(t, d),
        ("field.set", ("U",)): lambda d: _set_field(u, d),
        ("field.set", ("rAU",)): lambda d: _set_field(rau, d),
        ("flux.update", ("U",)): lambda d: phi.assign(fvc.flux(u)),
        ("field.get", ("alphaInletOutlet",)): lambda d: np.array(inlet_outlet.internalField()),
        ("field.set", ("alphaInletOutlet",)): lambda d: _set_field(inlet_outlet, d),
        ("bc.correctWithFlux", ("alphaInletOutlet",)): lambda d: _corrected_patches(
            mesh, inlet_outlet
        ),
    }
    two_phase = _TwoPhase(mesh, {"T": t, "U": u, "phi": phi, "rAU": rau})
    ops |= two_phase.ops()
    print("#READY", flush=True)
    _serve(ops)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
