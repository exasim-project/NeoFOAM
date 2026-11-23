// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/algorithms/vof/courant.hpp"
#include "NeoFOAM/algorithms/vof/fluxes.hpp"
#include "NeoFOAM/algorithms/vof/mules.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"
#include "bindings.hpp"

#include "cutCellIso.H"
#include "cutFaceIso.H"
#include "fvMesh.H"
#include "volFields.H"

namespace nb = nanobind;
using namespace nb::literals;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

namespace
{

using SurfaceScalar = fvcc::SurfaceField<NeoN::scalar>;

// The return-new / write-into-out convention of the per-step VoF bindings: without @p out the
// result is a new field, with it the result goes into @p out (reused buffers) and None is returned.
template<typename Into>
nb::object intoOrNew(SurfaceScalar* out, const SurfaceScalar& like, const char* name, Into into)
{
    if (out != nullptr)
    {
        into(*out);
        return nb::none();
    }
    SurfaceScalar result(
        like.exec(),
        name,
        like.mesh(),
        fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(like.mesh())
    );
    into(result);
    return nb::cast(std::move(result), nb::rv_policy::move);
}

using HostScalars = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NumpyScalars = nb::ndarray<nb::numpy, double, nb::c_contig>;

NumpyScalars ownedArray(const Foam::UList<Foam::scalar>& values)
{
    auto* buffer = new double[values.size()];
    std::copy(values.begin(), values.end(), buffer);
    nb::capsule owner(buffer, [](void* data) noexcept { delete[] static_cast<double*>(data); });
    const size_t shape[1] = {static_cast<size_t>(values.size())};
    return NumpyScalars(buffer, 1, shape, owner);
}

void requireSize(const HostScalars& values, Foam::label expected, const std::string& what)
{
    if (values.size() != static_cast<size_t>(expected))
    {
        throw nb::value_error((what + ": expected " + std::to_string(expected) + " values, got "
                               + std::to_string(values.size()))
                                  .c_str());
    }
}

// The cell and boundary loops of setAlpha in OpenFOAM's setAlphaField.C.
nb::tuple isoVolumeFractions(const Foam::fvMesh& mesh, HostScalars pointValues, double isoValue)
{
    requireSize(pointValues, mesh.nPoints(), "iso_volume_fractions: point_values");
    Foam::scalarField f(mesh.nPoints());
    std::copy_n(pointValues.data(), mesh.nPoints(), f.begin());
    Foam::cutCellIso cutCell(mesh, f);
    Foam::cutFaceIso cutFace(mesh, f);

    Foam::scalarField cells(mesh.nCells());
    forAll(cells, celli)
    {
        cutCell.calcSubCell(celli, isoValue);
        cells[celli] = Foam::clamp(cutCell.VolumeOfFluid(), Foam::zero_one {});
    }

    nb::dict patches;
    forAll(mesh.boundary(), patchi)
    {
        const Foam::fvPatch& patch = mesh.boundary()[patchi];
        const Foam::scalarField& magSf = mesh.magSf().boundaryField()[patchi];
        Foam::scalarField wetted(patch.size());
        forAll(wetted, patchFacei)
        {
            cutFace.calcSubFace(patch.start() + patchFacei, isoValue);
            wetted[patchFacei] = Foam::mag(cutFace.subFaceArea()) / magSf[patchFacei];
        }
        patches[patch.name().c_str()] = ownedArray(wetted);
    }
    return nb::make_tuple(ownedArray(cells), patches);
}

// Assigns the patch values directly, as setAlpha does: evaluating the boundary conditions would
// replace the wetted area fractions. Written with setAlphaField's 18 digits.
void writeVolumeFractions(Foam::volScalarField& field, HostScalars cells, const nb::dict& patches)
{
    requireSize(cells, field.size(), "write_volume_fractions: cells");
    std::copy_n(cells.data(), field.size(), field.primitiveFieldRef().begin());
    for (const auto& [name, values] : patches)
    {
        const auto patchName = nb::cast<std::string>(name);
        const Foam::label patchi = field.mesh().boundary().findPatchID(patchName);
        if (patchi < 0)
        {
            throw nb::value_error(("write_volume_fractions: no patch " + patchName).c_str());
        }
        const auto patchValues = nb::cast<HostScalars>(values);
        Foam::scalarField& target = field.boundaryFieldRef()[patchi];
        requireSize(patchValues, target.size(), "write_volume_fractions: patch " + patchName);
        std::copy_n(patchValues.data(), target.size(), target.begin());
    }
    const auto precision = Foam::IOstream::defaultPrecision(18);
    field.write();
    Foam::IOstream::defaultPrecision(precision);
}

void registerIsoVolumeFractions(nb::module_& m)
{
    // pybFoam binds points() on polyMesh only, and the tool's mesh is an fvMesh.
    nb::handle fvMeshType = nb::type<Foam::fvMesh>();
    fvMeshType.attr("points") = nb::cpp_function(
        [](const Foam::fvMesh& self) -> const Foam::pointField& { return self.points(); },
        nb::scope(fvMeshType),
        nb::name("points"),
        nb::is_method(),
        nb::rv_policy::reference_internal
    );

    m.def(
        "iso_volume_fractions",
        &isoVolumeFractions,
        "mesh"_a,
        "point_values"_a,
        "iso_value"_a = 0.0,
        "(cell volume fractions, {patch: wetted area fractions}) of the region where "
        "point_values > iso_value, from OpenFOAM's cutCellIso and cutFaceIso"
    );
    m.def(
        "write_volume_fractions",
        &writeVolumeFractions,
        "field"_a,
        "cells"_a,
        "patches"_a,
        "Assign cell and patch values to a volScalarField and write it"
    );
}

} // namespace

void registerVofOperators(nb::module_& m)
{
    registerIsoVolumeFractions(m);

    m.def(
        "upwind_flux",
        [](const fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi) { return nf::upwindAlphaFlux(alpha, phi); },
        "alpha"_a,
        "phi"_a,
        "Upwind convective flux phi_f * alpha_upwind(f)"
    );

    m.def(
        "vanleer_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha,
           const SurfaceScalar& phi,
           SurfaceScalar* out)
        {
            return intoOrNew(
                out,
                phi,
                "alphaPhiVanLeer",
                [&](SurfaceScalar& flux) { nf::vanLeerAlphaFlux(rt, alpha, phi, flux); }
            );
        },
        "runtime"_a,
        "alpha"_a,
        "phi"_a,
        "out"_a.none() = nb::none(),
        "vanLeer-limited convective flux fvc::flux(phi, alpha, 'Gauss vanLeer'); with out, "
        "written into out and None returned"
    );

    m.def(
        "alpha_phase_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double cAlpha) { return nf::alphaPhaseFlux(rt, alpha1, phi, cAlpha); },
        "runtime"_a,
        "alpha1"_a,
        "phi"_a,
        "c_alpha"_a,
        "interFoam alphaPhiUn = vanLeer flux + cAlpha interface compression"
    );

    m.def(
        "interface_compression_velocity",
        [](const SurfaceScalar& phi, double cAlpha, SurfaceScalar* out)
        {
            return intoOrNew(
                out,
                phi,
                "phic",
                [&](SurfaceScalar& phic) { nf::interfaceCompressionVelocity(phi, cAlpha, phic); }
            );
        },
        "phi"_a,
        "c_alpha"_a,
        "out"_a.none() = nb::none(),
        "Interface compression velocity phic = cAlpha*|phi/magSf| (alphaEqn.H), 0 on "
        "boundaries; with out, written into out and None returned"
    );

    m.def(
        "linear_compression_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const SurfaceScalar& phir,
           SurfaceScalar* out)
        {
            return intoOrNew(
                out,
                phir,
                "alphaPhiCompression",
                [&](SurfaceScalar& flux) { nf::linearCompressionFlux(rt, alpha1, phir, flux); }
            );
        },
        "runtime"_a,
        "alpha1"_a,
        "phir"_a,
        "out"_a.none() = nb::none(),
        "Compression flux fvc::flux(-fvc::flux(-phir, 1 - alpha1, linear), alpha1, linear) "
        "with OpenFOAM's linear weights; with out, written into out and None returned"
    );

    m.def(
        "add_interface_compression_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const SurfaceScalar& phic,
           const SurfaceScalar& nHatf,
           SurfaceScalar& alphaPhi)
        { nf::addInterfaceCompressionFlux(rt, alpha1, phic, nHatf, alphaPhi); },
        "runtime"_a,
        "alpha1"_a,
        "phic"_a,
        "nhatf"_a,
        "alpha_phi"_a,
        "alpha_phi += linear_compression_flux(alpha1, phic*nhatf), without the intermediate "
        "fields; after vanleer_flux into alpha_phi this is alpha_phase_flux"
    );

    m.def(
        "interface_nhatf",
        [](nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& alpha1, SurfaceScalar* out
        ) -> nb::object
        {
            if (out != nullptr)
            {
                nf::computeNHatf(rt, alpha1, *out);
                return nb::none();
            }
            return nb::cast(nf::computeNHatf(rt, alpha1), nb::rv_policy::move);
        },
        "runtime"_a,
        "alpha1"_a,
        "out"_a.none() = nb::none(),
        "Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK); "
        "with out, written into out and None returned"
    );

    m.def(
        "mules_explicit_solve",
        [](fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::VolumeField<NeoN::scalar>& alphaOld,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           fvcc::SurfaceField<NeoN::scalar>& alphaPhi,
           double deltaT,
           double psiMax,
           double psiMin,
           int nLimiterIter) {
            nf::mulesExplicitSolve(
                alpha,
                alphaOld,
                phi,
                alphaPhi,
                deltaT,
                psiMax,
                psiMin,
                nLimiterIter
            );
        },
        "alpha"_a,
        "alpha_old"_a,
        "phi"_a,
        "alpha_phi"_a,
        "delta_t"_a,
        "psi_max"_a = 1.0,
        "psi_min"_a = 0.0,
        "n_limiter_iter"_a = 3,
        "Bounded explicit MULES solve (MULES::explicitSolve; rho=1, Sp=Su=0, static mesh). "
        "alpha is the current corrector's field (upwind flux, local extrema), alpha_old the "
        "field at the start of the time step (psi.oldTime(): bounds and update; may be alpha "
        "itself). Limits alpha_phi in place and advances alpha conservatively."
    );

    m.def(
        "mules_correct",
        [](fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           fvcc::SurfaceField<NeoN::scalar>& phiCorr,
           double deltaT,
           int nLimiterIter,
           double relaxOld)
        { nf::mulesCorrect(alpha, phi, phiCorr, deltaT, nLimiterIter, relaxOld); },
        "alpha"_a,
        "phi"_a,
        "phi_corr"_a,
        "delta_t"_a,
        "n_limiter_iter"_a = 3,
        "relax_old"_a = 0.0,
        "MULES::correct: limit phi_corr in place and apply it to alpha. phi is the flux whose "
        "boundary value marks outlet faces (interFoam passes alphaPhiUn). relax_old > 0 blends "
        "with the pre-correction field (interFoam aCorr > 0). Boundary values of alpha are not "
        "corrected; call correct_boundary_conditions_with_flux afterwards."
    );

    m.def(
        "courant_numbers",
        [](const fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double deltaT) { return nf::courantNumbers(alpha, phi, deltaT); },
        "alpha"_a,
        "phi"_a,
        "delta_t"_a,
        "(CoNum, alphaCoNum) of advectorVoF's setDeltaT.H: 0.5*max(surfaceSum(|phi|)/V)*delta_t, "
        "the second over the cells with 0.01 < alpha < 0.99"
    );

    m.def(
        "correct_boundary_conditions_with_flux",
        [](fvcc::VolumeField<NeoN::scalar>& alpha, const fvcc::SurfaceField<NeoN::scalar>& phi)
        { nf::correctBoundaryConditionsWithFlux(alpha, phi); },
        "alpha"_a,
        "phi"_a,
        "Correct alpha's boundary values with phi as the BoundaryContext flux (inletOutlet)"
    );
}

} // namespace NeoFOAM::bindings
