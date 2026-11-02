// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

/*---------------------------------------------------------------------------*\
perfectGas equation of state with constant Cv-or-Cp and constant transport.

Matches classic OpenFOAM's hePsiThermo/pureMixture/perfectGas/[h|e]Const/const
thermoType, read from constant/thermophysicalProperties. C++ port of
src/neofoam/thermo/perfectGas.py's PerfectGasConstTransport -- see there for the
alphaHe-vs-alpha derivation (he, not T, is being diffused here, so the
diffusivity needed is kappa/Cv, matching OpenFOAM's heThermo::alphahe()).
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;

struct PerfectGasConstTransport
{
    static constexpr NeoN::scalar universalGasConstant = 8314.46261815324; // J/(kmol K)

    NeoN::scalar R;
    NeoN::scalar Cv;
    NeoN::scalar Cp;
    NeoN::scalar gamma;
    NeoN::scalar mu;
    NeoN::scalar Pr;
    NeoN::scalar alpha;
    NeoN::scalar alphaHe;

    explicit PerfectGasConstTransport(NeoFOAM::RunTime& rt)
    {
        Foam::IOdictionary thermoDict(Foam::IOobject(
            "thermophysicalProperties",
            rt.mesh.time().constant(),
            rt.mesh,
            Foam::IOobject::MUST_READ_IF_MODIFIED,
            Foam::IOobject::NO_WRITE
        ));
        const Foam::dictionary& mixture = thermoDict.subDict("mixture");
        const Foam::dictionary& specie = mixture.subDict("specie");
        const Foam::dictionary& thermodynamics = mixture.subDict("thermodynamics");
        const Foam::dictionary& transport = mixture.subDict("transport");

        const double molWeight = specie.get<double>("molWeight");
        R = static_cast<NeoN::scalar>(universalGasConstant / molWeight);
        if (thermodynamics.found("Cv"))
        {
            Cv = static_cast<NeoN::scalar>(thermodynamics.get<double>("Cv"));
            Cp = Cv + R;
        }
        else
        {
            Cp = static_cast<NeoN::scalar>(thermodynamics.get<double>("Cp"));
            Cv = Cp - R;
        }
        gamma = Cp / Cv;
        mu = static_cast<NeoN::scalar>(transport.get<double>("mu"));
        Pr = static_cast<NeoN::scalar>(transport.get<double>("Pr"));
        // alpha = mu/Pr = kappa/Cp: the diffusivity for *temperature* diffusion. Diffusing
        // he (internal energy) instead needs kappa/Cv = gamma*alpha (OpenFOAM's
        // heThermo::alphahe() = CpByCpv()*alpha_).
        alpha = mu / Pr;
        alphaHe = gamma * alpha;
    }
};

// psi = 1/(R*T): perfectGas compressibility, internal and boundary values alike.
inline void computePsi(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& T,
    NeoN::scalar R,
    fvcc::VolumeField<NeoN::scalar>& psi
)
{
    {
        const auto TV = T.internalVector().view();
        auto psiV = psi.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, T.internalVector().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { psiV[i] = 1.0 / (R * TV[i]); },
            "computePsi::internal"
        );
    }
    {
        const auto TV = T.boundaryData().value().view();
        auto psiV = psi.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, T.boundaryData().value().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { psiV[i] = 1.0 / (R * TV[i]); },
            "computePsi::boundary"
        );
    }
}

// rho = psi*p, internal and boundary values alike.
inline void computeRho(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& psi,
    const fvcc::VolumeField<NeoN::scalar>& p,
    fvcc::VolumeField<NeoN::scalar>& rho
)
{
    {
        const auto [psiV, pV] = NeoN::views(psi.internalVector(), p.internalVector());
        auto rhoV = rho.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, psi.internalVector().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { rhoV[i] = psiV[i] * pV[i]; },
            "computeRho::internal"
        );
    }
    {
        const auto [psiV, pV] = NeoN::views(psi.boundaryData().value(), p.boundaryData().value());
        auto rhoV = rho.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, psi.boundaryData().value().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { rhoV[i] = psiV[i] * pV[i]; },
            "computeRho::boundary"
        );
    }
}
