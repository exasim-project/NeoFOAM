// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "fvCFD.H"
#include "uniformDimensionedFields.H"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace
{
// Read constant/transportProperties: phases list + per-phase nu/rho + sigma.
struct PhaseProps
{
    double rho1, rho2, nu1, nu2, sigma;
};

PhaseProps readPhaseProps(nf::RunTime& rt)
{
    Foam::IOdictionary tp(Foam::IOobject(
        "transportProperties",
        rt.mesh.time().constant(),
        rt.mesh,
        Foam::IOobject::MUST_READ_IF_MODIFIED,
        Foam::IOobject::NO_WRITE
    ));
    Foam::wordList phases(tp.lookup("phases"));
    const Foam::dictionary& d1 = tp.subDict(phases[0]);
    const Foam::dictionary& d2 = tp.subDict(phases[1]);
    Foam::dimensionedScalar rho1("rho", Foam::dimDensity, d1);
    Foam::dimensionedScalar rho2("rho", Foam::dimDensity, d2);
    Foam::dimensionedScalar nu1("nu", Foam::dimViscosity, d1);
    Foam::dimensionedScalar nu2("nu", Foam::dimViscosity, d2);
    Foam::dimensionedScalar sigma("sigma", Foam::dimensionSet(1, 0, -2, 0, 0), tp);
    return {rho1.value(), rho2.value(), nu1.value(), nu2.value(), sigma.value()};
}

Foam::vector readG(nf::RunTime& rt)
{
    // constant/g is stored as a uniformDimensionedVectorField, not a plain dict.
    Foam::uniformDimensionedVectorField gFile(Foam::IOobject(
        "g",
        rt.mesh.time().constant(),
        rt.mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));
    return gFile.value();
}

Foam::volScalarField readAlpha1(nf::RunTime& rt, const std::string& alphaName = "alpha.water")
{
    return Foam::volScalarField(
        Foam::IOobject(
            alphaName,
            rt.mesh.time().timeName(),
            rt.mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        rt.mesh
    );
}

template<typename FoamField>
auto& registerVol(nf::RunTime& rt, const FoamField& f, const std::string& name)
{
    fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
    return vc.registerVector<fvcc::VolumeField<NeoN::scalar>>(nf::CreateFromFoamField<FoamField> {
        .exec = rt.exec,
        .nfMesh = rt.nfMesh,
        .foamField = f,
        .name = name
    });
}

// Initial rhoPhi = fvc::interpolate(rho) * phi, read OpenFOAM-side (setup only).
Foam::surfaceScalarField readRhoPhi(nf::RunTime& rt, const std::string& alphaName = "alpha.water")
{
    PhaseProps p = readPhaseProps(rt);
    Foam::volScalarField a = readAlpha1(rt, alphaName);
    Foam::volScalarField rho("rho", a * p.rho1 + (1.0 - a) * p.rho2);
    Foam::volVectorField U(
        Foam::IOobject(
            "U",
            rt.mesh.time().timeName(),
            rt.mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        rt.mesh
    );
    return Foam::surfaceScalarField("rhoPhi", Foam::fvc::interpolate(rho) * Foam::fvc::flux(U));
}
} // namespace

namespace NeoFOAM::bindings
{
void registerTwoPhaseProperties(nb::module_& m)
{
    m.def(
        "read_two_phase_transport_properties",
        [](nf::RunTime& rt) -> std::map<std::string, double>
        {
            PhaseProps p = readPhaseProps(rt);
            return {
                {"rho1", p.rho1},
                {"rho2", p.rho2},
                {"nu1", p.nu1},
                {"nu2", p.nu2},
                {"sigma", p.sigma}
            };
        },
        "runtime"_a,
        "Read rho1/rho2/nu1/nu2/sigma from constant/transportProperties"
    );

    m.def(
        "read_gravity",
        [](nf::RunTime& rt) -> std::tuple<double, double, double>
        {
            Foam::vector g = readG(rt);
            return {g.x(), g.y(), g.z()};
        },
        "runtime"_a,
        "Read gravity vector from constant/g"
    );

    m.def(
        "create_mixture_density",
        [](nf::RunTime& rt, const std::string& alphaName) -> fvcc::VolumeField<NeoN::scalar>&
        {
            PhaseProps p = readPhaseProps(rt);
            Foam::volScalarField a = readAlpha1(rt, alphaName);
            Foam::volScalarField rho("rho", a * p.rho1 + (1.0 - a) * p.rho2);
            return registerVol(rt, rho, "rho");
        },
        "runtime"_a,
        "alpha_name"_a = "alpha.water",
        nb::rv_policy::reference,
        "Create the mixture density rho = a*rho1 + (1-a)*rho2"
    );

    m.def(
        "create_mixture_viscosity",
        [](nf::RunTime& rt, const std::string& alphaName) -> fvcc::VolumeField<NeoN::scalar>&
        {
            PhaseProps p = readPhaseProps(rt);
            Foam::volScalarField a = readAlpha1(rt, alphaName);
            Foam::volScalarField mu("mu", a * (p.rho1 * p.nu1) + (1.0 - a) * (p.rho2 * p.nu2));
            return registerVol(rt, mu, "mu");
        },
        "runtime"_a,
        "alpha_name"_a = "alpha.water",
        nb::rv_policy::reference,
        "Create the mixture dynamic viscosity mu"
    );

    m.def(
        "create_gh",
        [](nf::RunTime& rt) -> fvcc::VolumeField<NeoN::scalar>&
        {
            Foam::vector gv = readG(rt);
            Foam::dimensionedVector g("g", Foam::dimAcceleration, gv);
            Foam::volScalarField gh("gh", g & rt.mesh.C());
            return registerVol(rt, gh, "gh");
        },
        "runtime"_a,
        nb::rv_policy::reference,
        "Create gh = g & C"
    );

    m.def(
        "create_ghf",
        [](nf::RunTime& rt) -> fvcc::SurfaceField<NeoN::scalar>&
        {
            Foam::vector gv = readG(rt);
            Foam::dimensionedVector g("g", Foam::dimAcceleration, gv);
            Foam::surfaceScalarField ghf("ghf", g & rt.mesh.Cf());
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            return vc.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::surfaceScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ghf,
                    .name = "ghf"
                }
            );
        },
        "runtime"_a,
        nb::rv_policy::reference,
        "Create ghf = g & Cf"
    );

    m.def(
        "create_rho_phi",
        [](nf::RunTime& rt, const std::string& alphaName) -> fvcc::SurfaceField<NeoN::scalar>&
        {
            Foam::surfaceScalarField rhoPhi = readRhoPhi(rt, alphaName);
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            return vc.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::surfaceScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = rhoPhi,
                    .name = "rhoPhi"
                }
            );
        },
        "runtime"_a,
        "alpha_name"_a = "alpha.water",
        nb::rv_policy::reference,
        "Create the initial density-weighted face flux rhoPhi = interpolate(rho)*phi"
    );

    m.def(
        "bound_scalar_field",
        [](fvcc::VolumeField<NeoN::scalar>& f, double lo, double hi)
        {
            auto v = f.internalVector().view();
            NeoN::parallelFor(
                f.exec(),
                {0, f.internalVector().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) {
                    v[i] = Kokkos::min(hi, Kokkos::max(lo, v[i]));
                }
            );
        },
        "field"_a,
        "lo"_a,
        "hi"_a,
        "Clamp scalar cell values to [lo, hi] in place"
    );

    m.def(
        "update_mixture_density",
        [](fvcc::VolumeField<NeoN::scalar>& rho,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           double rho1,
           double rho2)
        {
            auto r = rho.internalVector().view();
            auto a = alpha1.internalVector().view();
            NeoN::parallelFor(
                rho.exec(),
                {0, rho.internalVector().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) { r[i] = a[i] * rho1 + (1.0 - a[i]) * rho2; }
            );
        },
        "rho"_a,
        "alpha1"_a,
        "rho1"_a,
        "rho2"_a,
        "In-place rho = alpha1*rho1 + (1-alpha1)*rho2 from the live alpha1"
    );

    m.def(
        "update_mixture_viscosity",
        [](fvcc::VolumeField<NeoN::scalar>& mu,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           double rho1,
           double rho2,
           double nu1,
           double nu2)
        {
            auto m_ = mu.internalVector().view();
            auto a = alpha1.internalVector().view();
            NeoN::parallelFor(
                mu.exec(),
                {0, mu.internalVector().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) {
                    m_[i] = a[i] * (rho1 * nu1) + (1.0 - a[i]) * (rho2 * nu2);
                }
            );
        },
        "mu"_a,
        "alpha1"_a,
        "rho1"_a,
        "rho2"_a,
        "nu1"_a,
        "nu2"_a,
        "In-place dynamic-viscosity blend from the live alpha1"
    );

    m.def(
        "update_rho_phi",
        [](fvcc::SurfaceField<NeoN::scalar>& rhoPhi,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double rho1,
           double rho2)
        {
            // Upwind alpha face VALUE consistent with the implicit upwind div solve.
            // computeUpwindInterpolation writes the upwind cell value at each face
            // (not the flux), so multiply by phi to form the alpha face flux
            // alphaPhi = phi*alpha_upwind, then rhoPhi = alphaPhi*(rho1-rho2)+phi*rho2.
            fvcc::Upwind<NeoN::scalar> upw(
                alpha1.exec(),
                alpha1.mesh(),
                NeoN::Input {NeoN::TokenList {}}
            );
            upw.interpolate(phi, alpha1, rhoPhi); // rhoPhi := upwind alpha face value
            auto rp = rhoPhi.internalVector().view();
            auto ph = phi.internalVector().view();
            NeoN::parallelFor(
                rhoPhi.exec(),
                {0, rhoPhi.internalVector().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) {
                    rp[i] = ph[i] * (rp[i] * (rho1 - rho2) + rho2);
                }
            );
            // Boundary faces carry the upwind alpha value (weight 1) after interpolate;
            // apply the same phi-scaled density weight so boundary rhoPhi is a real
            // density flux in [rho2,rho1]*phi (design F2), consumed by div(rhoPhi,U).
            auto rpb = rhoPhi.boundaryData().value().view();
            auto phb = phi.boundaryData().value().view();
            NeoN::parallelFor(
                rhoPhi.exec(),
                {0, rhoPhi.boundaryData().value().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) {
                    rpb[i] = phb[i] * (rpb[i] * (rho1 - rho2) + rho2);
                }
            );
        },
        "rhoPhi"_a,
        "alpha1"_a,
        "phi"_a,
        "rho1"_a,
        "rho2"_a,
        "rhoPhi = upwind(phi,alpha1)*(rho1-rho2) + phi*rho2 from the live alpha1"
    );

    m.def(
        "update_static_pressure",
        [](fvcc::VolumeField<NeoN::scalar>& p,
           const fvcc::VolumeField<NeoN::scalar>& p_rgh,
           const fvcc::VolumeField<NeoN::scalar>& rho,
           const fvcc::VolumeField<NeoN::scalar>& gh)
        {
            auto pv = p.internalVector().view();
            auto prgh = p_rgh.internalVector().view();
            auto rhov = rho.internalVector().view();
            auto ghv = gh.internalVector().view();
            NeoN::parallelFor(
                p.exec(),
                {0, p.internalVector().size()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) { pv[i] = prgh[i] + rhov[i] * ghv[i]; }
            );
        },
        "p"_a,
        "p_rgh"_a,
        "rho"_a,
        "gh"_a,
        "In-place p = p_rgh + rho*gh over the internal field"
    );
}
} // namespace NeoFOAM::bindings
