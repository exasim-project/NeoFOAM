// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>
#include <string>
#include <utility>

#include "NeoN/NeoN.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vof/twoPhaseMixture.hpp"
#include "detail.hpp"
#include "foamGeometry.hpp"

#include "IOdictionary.H"
#include "dimensionedScalar.H"
#include "wordList.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace
{

using NeoN::localIdx;
using NeoN::scalar;

// Scalar.H's clamp(val, zero_one): NaN and the sign of zero pass through as in OpenFOAM.
KOKKOS_INLINE_FUNCTION scalar clampZeroOne(scalar value)
{
    return (value < 0) ? 0 : (1 < value) ? 1 : value;
}

// incompressibleTwoPhaseMixture::calcNu and mu() on one value, in their evaluation order.
KOKKOS_INLINE_FUNCTION scalar
mixtureNu(scalar alpha1, scalar rho1, scalar nu1, scalar rho2, scalar nu2)
{
    const scalar limited = clampZeroOne(alpha1);
    const scalar mu = limited * rho1 * nu1 + (scalar(1) - limited) * rho2 * nu2;
    return mu / (limited * rho1 + (scalar(1) - limited) * rho2);
}

void multiplyValues(
    const NeoN::Vector<scalar>& x,
    const NeoN::Vector<scalar>& y,
    NeoN::Vector<scalar>& out
)
{
    const auto xv = x.view();
    const auto yv = y.view();
    auto ov = out.view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) { ov[i] = xv[i] * yv[i]; },
        "twoPhase::multiply"
    );
}

void mixtureDensityValues(
    const NeoN::Vector<scalar>& alpha1,
    scalar rho1,
    scalar rho2,
    NeoN::Vector<scalar>& rho
)
{
    const auto a = alpha1.view();
    auto r = rho.view();
    NeoN::parallelFor(
        rho.exec(),
        {0, rho.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar alpha2 = 1.0 - a[i];
            r[i] = a[i] * rho1 + alpha2 * rho2;
        },
        "twoPhase::mixtureDensity"
    );
}

void mixtureViscosityValues(
    const NeoN::Vector<scalar>& alpha1,
    const TwoPhaseTransportProperties& p,
    NeoN::Vector<scalar>& nu
)
{
    const auto a = alpha1.view();
    auto n = nu.view();
    const scalar rho1 = p.rho1;
    const scalar nu1 = p.nu1;
    const scalar rho2 = p.rho2;
    const scalar nu2 = p.nu2;
    NeoN::parallelFor(
        nu.exec(),
        {0, nu.size()},
        NEON_LAMBDA(const localIdx i) { n[i] = mixtureNu(a[i], rho1, nu1, rho2, nu2); },
        "twoPhase::mixtureViscosity"
    );
}

// The phase sub-dictionary's Newtonian nu and rho, as viscosityModels::Newtonian and
// incompressibleTwoPhaseMixture read them.
std::pair<scalar, scalar>
readNewtonianPhase(const Foam::dictionary& phaseDict, const Foam::word& phase)
{
    const Foam::word model = phaseDict.get<Foam::word>("transportModel");
    if (model != "Newtonian")
    {
        throw std::runtime_error(
            "two-phase mixture: phase '" + phase + "' uses transportModel " + model
            + "; only Newtonian is supported"
        );
    }
    const Foam::dimensionedScalar rho("rho", Foam::dimDensity, phaseDict);
    const Foam::dimensionedScalar nu("nu", Foam::dimViscosity, phaseDict);
    return {rho.value(), nu.value()};
}

} // namespace

TwoPhaseTransportProperties readTwoPhaseTransportProperties(RunTime& rt)
{
    const Foam::IOdictionary transportProperties(Foam::IOobject(
        "transportProperties",
        rt.mesh.time().constant(),
        rt.mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));
    const Foam::wordList phases(transportProperties.get<Foam::wordList>("phases"));
    if (phases.size() != 2)
    {
        throw std::runtime_error("transportProperties: 'phases' must list exactly two phases");
    }
    if (transportProperties.isDict("sigma"))
    {
        throw std::runtime_error(
            "transportProperties: sigma must be a constant; surface tension models are not "
            "supported"
        );
    }
    const auto [rho1, nu1] = readNewtonianPhase(transportProperties.subDict(phases[0]), phases[0]);
    const auto [rho2, nu2] = readNewtonianPhase(transportProperties.subDict(phases[1]), phases[1]);
    const Foam::dimensionedScalar sigma(
        "sigma",
        Foam::dimensionSet(1, 0, -2, 0, 0),
        transportProperties
    );
    return {phases[0], phases[1], rho1, nu1, rho2, nu2, sigma.value()};
}

void updateMixtureDensity(
    const fvcc::VolumeField<scalar>& alpha1,
    scalar rho1,
    scalar rho2,
    fvcc::VolumeField<scalar>& rho
)
{
    mixtureDensityValues(alpha1.internalVector(), rho1, rho2, rho.internalVector());
    mixtureDensityValues(alpha1.boundaryData().value(), rho1, rho2, rho.boundaryData().value());
}

void updateMixtureViscosity(
    const fvcc::VolumeField<scalar>& alpha1,
    const TwoPhaseTransportProperties& properties,
    fvcc::VolumeField<scalar>& nu
)
{
    mixtureViscosityValues(alpha1.internalVector(), properties, nu.internalVector());
    mixtureViscosityValues(alpha1.boundaryData().value(), properties, nu.boundaryData().value());
}

void multiply(
    const fvcc::VolumeField<scalar>& x,
    const fvcc::VolumeField<scalar>& y,
    fvcc::VolumeField<scalar>& out
)
{
    multiplyValues(x.internalVector(), y.internalVector(), out.internalVector());
    multiplyValues(x.boundaryData().value(), y.boundaryData().value(), out.boundaryData().value());
}

void multiply(
    const fvcc::SurfaceField<scalar>& x,
    const fvcc::SurfaceField<scalar>& y,
    fvcc::SurfaceField<scalar>& out
)
{
    multiplyValues(x.internalVector(), y.internalVector(), out.internalVector());
    multiplyValues(x.boundaryData().value(), y.boundaryData().value(), out.boundaryData().value());
}

void linearInterpolate(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& vf,
    fvcc::SurfaceField<scalar>& out
)
{
    const auto& mesh = vf.mesh();
    if (mesh.nProcBoundaryFaces() > 0)
    {
        throw std::runtime_error("linearInterpolate: coupled (processor) patches are not supported"
        );
    }
    const auto weights = foamGeometry(rt).weights.view();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto vI = vf.internalVector().view();
    auto oI = out.internalVector().view();
    NeoN::parallelFor(
        vf.exec(),
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { oI[f] = faceValue(weights[f], vI[own[f]], vI[nei[f]]); },
        "twoPhase::linearInterpolate"
    );
    const auto vB = vf.boundaryData().value().view();
    auto oB = out.boundaryData().value().view();
    NeoN::parallelFor(
        vf.exec(),
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { oB[bf] = vB[bf]; },
        "twoPhase::linearInterpolateBoundary"
    );
}

} // namespace NeoFOAM
