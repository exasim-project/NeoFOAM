// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/algorithms/pressureVelocityCoupling.hpp"
#include "NeoFOAM/datastructures/pde.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerPressureVelocityCoupling(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Pressure-velocity coupling helpers (free functions)
    // -------------------------------------------------------------------
    m.def(
        "compute_rau_and_hbya",
        [](const nf::PDE<NeoN::Vec3>& UEqn) { return nf::computeRAUandHByA(UEqn); },
        "UEqn"_a,
        "Compute rAU and HbyA from the assembled momentum equation"
    );

    m.def(
        "constrain_hbya",
        [](const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<NeoN::scalar>& p,
           fvcc::VolumeField<NeoN::Vec3>& hByA) { nf::constrainHbyA(U, p, hByA); },
        "U"_a,
        "p"_a,
        "hByA"_a,
        "Constrain HbyA at boundaries with assigned velocity BCs"
    );

    m.def(
        "flux",
        [](const fvcc::VolumeField<NeoN::Vec3>& volField) { return nf::flux(volField); },
        "vol_field"_a,
        "Compute face flux from a volume vector field"
    );

    m.def(
        "constrain_pressure",
        [](fvcc::VolumeField<NeoN::scalar>& p_rgh,
           const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::SurfaceField<NeoN::scalar>& phiHbyA,
           const fvcc::SurfaceField<NeoN::scalar>& rAUf)
        { nf::constrainPressure(p_rgh, U, phiHbyA, rAUf); },
        "p_rgh"_a,
        "U"_a,
        "phiHbyA"_a,
        "rAUf"_a,
        "Set wall fixedFluxPressure refGrad so the projection cancels the buoyancy face flux."
    );

    m.def(
        "update_face_velocity",
        [](const fvcc::SurfaceField<NeoN::scalar>& phiHbyA,
           const nf::PDE<NeoN::scalar>& pEqn,
           fvcc::SurfaceField<NeoN::scalar>& phi) { nf::updateFaceVelocity(phiHbyA, pEqn, phi); },
        "phi_hbya"_a,
        "pEqn"_a,
        "phi"_a,
        "Update face velocity (phi) after pressure correction"
    );

    m.def(
        "update_velocity",
        [](const fvcc::VolumeField<NeoN::Vec3>& hByA,
           const fvcc::VolumeField<NeoN::scalar>& rAU,
           const fvcc::VolumeField<NeoN::scalar>& p,
           fvcc::VolumeField<NeoN::Vec3>& U) { nf::updateVelocity(hByA, rAU, p, U); },
        "hByA"_a,
        "rAU"_a,
        "p"_a,
        "U"_a,
        "Update cell velocity: U = HbyA - rAU * grad(p)"
    );

    m.def(
        "update_velocity_buoyant",
        [](const fvcc::VolumeField<NeoN::Vec3>& hByA,
           const fvcc::VolumeField<NeoN::scalar>& rAU,
           const fvcc::SurfaceField<NeoN::scalar>& numeratorFlux,
           const fvcc::SurfaceField<NeoN::scalar>& rAUf,
           fvcc::VolumeField<NeoN::Vec3>& U)
        { nf::updateVelocityBuoyant(hByA, rAU, numeratorFlux, rAUf, U); },
        "hByA"_a,
        "rAU"_a,
        "numerator_flux"_a,
        "rAUf"_a,
        "U"_a,
        "Buoyant velocity: U = HbyA + rAU * reconstruct(numerator_flux / rAUf), where "
        "numerator_flux = phig - pEqn.flux() (interFoam velocity correction)."
    );

    m.def(
        "ddt_flux_corr",
        [](const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double dt,
           fvcc::DdtScheme scheme)
        { return fvcc::ddtFluxCorr(U, phi, static_cast<NeoN::scalar>(dt), scheme); },
        "U"_a,
        "phi"_a,
        "dt"_a,
        "scheme"_a,
        "Compute ddt flux correction for PISO loop (BDF1/BDF2)"
    );
}

} // namespace NeoFOAM::bindings
