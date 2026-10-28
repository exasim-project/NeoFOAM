// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*---------------------------------------------------------------------------*\
Application
    neorhoSimpleFoam

Description
    Steady-state compressible SIMPLE solver using NeoN + NeoFOAM, the C++
    twin of src/neofoam/solver/neorhoSimpleFoam.py -- see that file's module
    docstring for the full rationale (he-space energy equation, why K and the
    pressure-work term stay separate rather than folded into a single Ekp
    term, the pEqn sign convention). Same minimal scope as
    neorhoPimpleFoam.cpp: laminar only, subsonic (no transonic pEqn branch),
    no SIMPLEC/consistent variant, no closed-volume handling. No ddt terms
    anywhere (steady state), so unlike neorhoPimpleFoam.cpp there is no inner
    PISO loop and no psip0/p*ddt_psi bookkeeping.

    Unlike the Python port, NeoN's C++ API has no high-level elementwise
    VolumeField/SurfaceField arithmetic (see neorhoPimpleFoam.cpp's docstring
    for the exact gaps), so every EOS/bookkeeping update here (psi, rho, K,
    he<->T, phiv, rho*rAU) is an explicit NeoN::parallelFor kernel instead of
    a field expression.
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"
#include "simpleControl.H"

#include "perfectGasConstTransport.hpp"

#include <string>
#include <vector>

using Foam::Info;
using Foam::endl;
using Foam::nl;

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace
{

// Same he = Cv*(T - Tstd) rationale as the Python port's module docstring.
constexpr NeoN::scalar TStd = 298.15;

// Mirrors create_uniform_volume_field (src/bindings/fieldFactories.cpp).
fvcc::VolumeField<NeoN::scalar>
createUniformVolumeScalar(nf::RunTime& rt, const std::string& name, NeoN::scalar value)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::VolumeField<NeoN::scalar> field(rt.exec, name, rt.nfMesh, bcs);
    NeoN::fill(field.internalVector(), value);
    NeoN::fill(field.boundaryData().value(), value);
    return field;
}

// Mirrors create_uniform_surface_field.
fvcc::SurfaceField<NeoN::scalar>
createUniformSurfaceScalar(nf::RunTime& rt, const std::string& name, NeoN::scalar value)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::SurfaceField<NeoN::scalar> field(rt.exec, name, rt.nfMesh, bcs);
    NeoN::fill(field.internalVector(), value);
    NeoN::fill(field.boundaryData().value(), value);
    return field;
}

// Mirrors register_computed_scalar_field: add a freshly-built field to the database under its
// own name so fvSolution relaxation lookups / boundary handling can find it. (Unlike
// neorhoPimpleFoam.cpp's use of this same helper, SIMPLE never needs the old-time storage that
// registration also happens to enable -- there are no ddt terms anywhere in this solver.)
fvcc::VolumeField<NeoN::scalar>& registerComputedScalar(
    fvcc::VectorCollection& vectorCollection,
    const fvcc::VolumeField<NeoN::scalar>& field,
    const std::string& name
)
{
    return vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
        fvcc::CreateFromExistingVector<fvcc::VolumeField<NeoN::scalar>> {
            .name = name,
            .field = field,
            .timeIndex = 0,
            .iterationIndex = 0,
            .subCycleIndex = 0
        }
    );
}

fvcc::SurfaceField<NeoN::scalar>& registerComputedSurface(
    fvcc::VectorCollection& vectorCollection,
    const fvcc::SurfaceField<NeoN::scalar>& field,
    const std::string& name
)
{
    return vectorCollection.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
        fvcc::CreateFromExistingVector<fvcc::SurfaceField<NeoN::scalar>> {
            .name = name,
            .field = field,
            .timeIndex = 0,
            .iterationIndex = 0,
            .subCycleIndex = 0
        }
    );
}

// out = Cv*(T - TStd), internal values only -- he's boundary comes from
// mapEnergyBoundaryFromTemperature + correctBoundaryConditions() instead.
void heFromT(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& T,
    NeoN::scalar Cv,
    fvcc::VolumeField<NeoN::scalar>& he
)
{
    const auto TV = T.internalVector().view();
    auto heV = he.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, T.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { heV[i] = Cv * (TV[i] - TStd); },
        "heFromT"
    );
}

// T = (1/Cv)*he + TStd, internal and boundary alike (mirrors he's own update scope).
void temperatureFromHe(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& he,
    NeoN::scalar Cv,
    fvcc::VolumeField<NeoN::scalar>& T
)
{
    {
        const auto heV = he.internalVector().view();
        auto TV = T.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, he.internalVector().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { TV[i] = heV[i] / Cv + TStd; },
            "temperatureFromHe::internal"
        );
    }
    {
        const auto heV = he.boundaryData().value().view();
        auto TV = T.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, he.boundaryData().value().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { TV[i] = heV[i] / Cv + TStd; },
            "temperatureFromHe::boundary"
        );
    }
}

// out = 0.5*magSqr(U), internal values only (K only ever feeds EEqn's explicit terms, which
// read internal values; matches the Python port's own scope).
void kineticEnergy(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::Vec3>& U,
    fvcc::VolumeField<NeoN::scalar>& K
)
{
    const auto UV = U.internalVector().view();
    auto KV = K.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, U.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const auto& u = UV[i];
            KV[i] = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        },
        "kineticEnergy"
    );
}

// out = a*b, internal AND boundary values -- for rho*rAU, which goes on to be
// face-interpolated (see neorhoPimpleFoam.cpp's identical helper for why boundary values
// matter here, unlike the he<->T/EOS kernels above).
void multiplyFieldsWithBoundary(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& a,
    const fvcc::VolumeField<NeoN::scalar>& b,
    fvcc::VolumeField<NeoN::scalar>& out
)
{
    {
        const auto [aV, bV] = NeoN::views(a.internalVector(), b.internalVector());
        auto outV = out.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, a.internalVector().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] * bV[i]; },
            "multiplyFieldsWithBoundary::internal"
        );
    }
    {
        const auto [aV, bV] = NeoN::views(a.boundaryData().value(), b.boundaryData().value());
        auto outV = out.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, a.boundaryData().value().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] * bV[i]; },
            "multiplyFieldsWithBoundary::boundary"
        );
    }
}

// out = a/b, internal AND boundary values -- NeoN's SurfaceField<scalar> has no native
// operator/ (only operator*,+,-), so phi/rhoFace (phiv, the volumetric flux) needs its own
// explicit kernel.
void divideSurfaceFields(
    const NeoN::Executor& exec,
    const fvcc::SurfaceField<NeoN::scalar>& a,
    const fvcc::SurfaceField<NeoN::scalar>& b,
    fvcc::SurfaceField<NeoN::scalar>& out
)
{
    {
        const auto [aV, bV] = NeoN::views(a.internalVector(), b.internalVector());
        auto outV = out.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, a.internalVector().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] / bV[i]; },
            "divideSurfaceFields::internal"
        );
    }
    {
        const auto [aV, bV] = NeoN::views(a.boundaryData().value(), b.boundaryData().value());
        auto outV = out.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, a.boundaryData().value().size()},
            NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] / bV[i]; },
            "divideSurfaceFields::boundary"
        );
    }
}

} // namespace

int main(int argc, char* argv[])
{
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime, args);
        auto& mesh = rt.mesh;

        Foam::simpleControl simple(mesh);

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        for (const std::string& name : {"p", "U", "rho", "T", "e"})
        {
            if (solverDict.contains(name))
            {
                solverDict.subDict(name) = nf::mapFvSolution(solverDict.subDict(name));
            }
        }
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        PerfectGasConstTransport gas(rt);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& T = nf::constructAndRegister(vectorCollection, rt, ofT, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        // he's patches are real FixedEnergy/GradientEnergy BC objects, refreshed from T's
        // current boundary every outer iteration via mapEnergyBoundaryFromTemperature below
        // -- not a value captured once at setup. Patch order (inlet, outlet, walls,
        // frontAndBack) is this case's own blockMeshDict order; a different mesh needs this
        // list reworked, not just run through unchanged (same caveat as the Python port and
        // neorhoPimpleFoam.cpp).
        NF_ASSERT(
            rt.nfMesh.nBoundaries() == 4,
            "he's patch-type list (fixedEnergy/gradientEnergy/gradientEnergy/empty) assumes "
            "this case's exact 4-patch (inlet, outlet, walls, frontAndBack) blockMeshDict "
            "order; a different mesh needs this list reworked, not just run through unchanged"
        );
        const std::vector<std::string> hePatchTypes =
            {"fixedEnergy", "gradientEnergy", "gradientEnergy", "empty"};
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> heBcs;
        heBcs.reserve(hePatchTypes.size());
        for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(hePatchTypes.size());
             ++patchID)
        {
            NeoN::Dictionary patchDict({{"type", hePatchTypes[static_cast<std::size_t>(patchID)]}});
            heBcs.emplace_back(rt.nfMesh, patchDict, patchID);
        }
        fvcc::VolumeField<NeoN::scalar> heSeed(rt.exec, "e", rt.nfMesh, heBcs);
        heFromT(rt.exec, T, gas.Cv, heSeed);
        auto& he = registerComputedScalar(vectorCollection, heSeed, "e");
        fvcc::mapEnergyBoundaryFromTemperature(T.boundaryData(), he.boundaryData(), gas.Cv, TStd);
        he.correctBoundaryConditions();

        auto psiSeed = createUniformVolumeScalar(rt, "psi", 0.0);
        computePsi(rt.exec, T, gas.R, psiSeed);
        auto& psi = registerComputedScalar(vectorCollection, psiSeed, "psi");

        auto rhoSeed = createUniformVolumeScalar(rt, "rho", 0.0);
        computeRho(rt.exec, psi, p, rhoSeed);
        auto& rho = registerComputedScalar(vectorCollection, rhoSeed, "rho");

        auto KSeed = createUniformVolumeScalar(rt, "K", 0.0);
        kineticEnergy(rt.exec, U, KSeed);
        auto& K = registerComputedScalar(vectorCollection, KSeed, "K");

        // he (internal energy) is being diffused here, not T -- kappa/Cv (gas.alphaHe), not
        // kappa/Cp; see PerfectGasConstTransport's own derivation.
        auto muVol = createUniformVolumeScalar(rt, "mu", gas.mu);
        auto zeroNut = createUniformVolumeScalar(rt, "nut0", 0.0);
        auto alphaEff = createUniformSurfaceScalar(rt, "alphaEff", gas.alphaHe);
        // Scratch buffer for rho*rAU (VolumeField<scalar> has no native operator* either --
        // only SurfaceField does), interpolated to faces as rhorAUf below.
        auto rhoCrAU = createUniformVolumeScalar(rt, "rho_crAU", 0.0);

        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );

        auto rhoFaceInit = surfInterpol.interpolate(rho);
        auto phiSeed = rhoFaceInit * nf::flux(U);
        phiSeed.name = "phi";
        auto& phi = registerComputedSurface(vectorCollection, phiSeed, "phi");

        NeoN::scalar cumulativeContErr = 0.0;

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            rt.t = runTime.time().value();
            NeoN::Logging::info("Time = {}", rt.t);

            // --- Momentum (no ddt term: steady state) ---
            fvcc::VolumeField<NeoN::Tensor> gradU(
                rt.exec,
                "gradU",
                rt.nfMesh,
                fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(rt.nfMesh)
            );
            nf::gradScheme(rt, "grad(U)").gradTensor(U, gradU, dsl::Coeff {});
            gradU.correctBoundaryConditions();

            // Same mu-not-rho*mu collapse rationale as neorhoPimpleFoam.cpp's UEqn (laminar
            // Stokes model: nuEff()==nu()==mu/rho).
            auto muFace = surfInterpol.interpolate(muVol);
            muFace.name = "mu";

            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::div(phi, U) - dsl::imp::laplacian(muFace, U)
                    + dsl::exp::viscousStress(muVol, zeroNut, gradU, true),
                U,
                rt
            );
            UEqn.relax();

            if (simple.momentumPredictor())
            {
                UEqn.solve(-1.0 * dsl::exp::grad(p));
                kineticEnergy(rt.exec, U, K);
            }
            else
            {
                UEqn.assembleAndRelax();
            }

            // --- Energy (he-space; no ddt(rho,he)/ddt(rho,K): steady state) ---
            T.correctBoundaryConditions();
            fvcc::mapEnergyBoundaryFromTemperature(
                T.boundaryData(),
                he.boundaryData(),
                gas.Cv,
                TStd
            );
            he.correctBoundaryConditions();

            auto rhoFaceForPhiv = surfInterpol.interpolate(rho);
            auto phiv = createUniformSurfaceScalar(rt, "phiv", 0.0);
            divideSurfaceFields(rt.exec, phi, rhoFaceForPhiv, phiv);

            nf::PDESolver<NeoN::scalar> EEqn(
                dsl::imp::div(phi, he) - dsl::imp::laplacian(alphaEff, he) + dsl::exp::div(phi, K)
                    + dsl::exp::div(phiv, p),
                he,
                rt
            );
            EEqn.relax();
            EEqn.solve();
            he.correctBoundaryConditions();

            // Convert he back to T for the EOS/reporting.
            temperatureFromHe(rt.exec, he, gas.Cv, T);
            computePsi(rt.exec, T, gas.R, psi);

            // --- Pressure-velocity coupling (single pass: no inner PISO loop) ---
            auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
            nf::constrainHbyA(U, p, hByA);

            auto rhoFace = surfInterpol.interpolate(rho);
            multiplyFieldsWithBoundary(rt.exec, rho, crAU, rhoCrAU);
            auto rhorAUf = surfInterpol.interpolate(rhoCrAU);
            rhorAUf.name = "rhorAUf";
            auto phiHbyA = rhoFace * nf::flux(hByA);

            auto prevP = NeoN::dsl::fieldRelaxationSnapshot(p);

            while (simple.correctNonOrthogonal())
            {
                // pEqn built as +div(phiHbyA) - laplacian(rhorAUf,p), exactly mirroring
                // neorhoPimpleFoam.cpp's pEqn with the ddt(psi,p)/source(pDdtPsi) terms
                // dropped (steady state has no d(psi*p)/dt) -- same opposite-of-incompressible
                // sign convention, so update_face_velocity still needs the explicit override.
                nf::PDESolver<NeoN::scalar> pEqn(
                    dsl::exp::div(phiHbyA) - dsl::imp::laplacian(rhorAUf, p),
                    p,
                    rt
                );

                if (ofP.needReference() && pRefCell >= 0)
                {
                    pEqn.setReference(pRefCell, pRefValue);
                }

                pEqn.solve();
                p.correctBoundaryConditions();

                if (simple.finalNonOrthogonalIter())
                {
                    nf::updateFaceVelocity(
                        phiHbyA,
                        pEqn,
                        phi,
                        nf::PEqnConvention::DivMinusLaplacian
                    );
                }
            }

            nf::reportContinuityError(phi, rt, cumulativeContErr);

            NeoN::dsl::applyFieldRelaxation(
                p,
                prevP,
                nf::lookupFieldRelaxation(rt.fvSolutionDict, p.name, false).value_or(1.0)
            );
            p.correctBoundaryConditions();

            nf::updateVelocity(hByA, crAU, p, U, rt);
            U.correctBoundaryConditions();
            kineticEnergy(rt.exec, U, K);

            auto prevRho = NeoN::dsl::fieldRelaxationSnapshot(rho);
            computeRho(rt.exec, psi, p, rho);
            NeoN::dsl::applyFieldRelaxation(
                rho,
                prevRho,
                nf::lookupFieldRelaxation(rt.fvSolutionDict, rho.name, false).value_or(1.0)
            );
            rho.correctBoundaryConditions();

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing T");
                write(T, mesh);
                NeoN::Logging::info("Writing rho");
                write(rho, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
