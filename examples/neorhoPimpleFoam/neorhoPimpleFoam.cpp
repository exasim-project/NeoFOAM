// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*---------------------------------------------------------------------------*\
Application
    neorhoPimpleFoam

Description
    Transient compressible PIMPLE solver using NeoN + NeoFOAM, the C++ twin of
    src/neofoam/solver/neorhoPimpleFoam.py -- see that file's module docstring
    for the full rationale (why he = Cv*(T - Tstd) rather than raw T, the
    psi*p-weighted imp.ddt(psi,p) pressure equation, phiHbyA's density-weighted
    ddtFluxCorr term). Same minimal scope: laminar only, subsonic (no transonic
    pEqn branch), no SIMPLEC/consistent variant, no closed-volume handling.

    Unlike the Python port, NeoN's C++ API has no high-level elementwise
    VolumeField arithmetic (VolumeField<scalar> has no multiply/divide operators
    at all; SurfaceField<scalar> has multiply/add/subtract but still no divide),
    so every EOS/bookkeeping update here (psi, rho, K, he<->T, psip0, p*ddt_psi,
    phiv, rho*rAU) is an explicit NeoN::parallelFor kernel instead of a field
    expression.
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"

#include "fvCFD.H"
#include "pisoControl.H"

#include "perfectGasConstTransport.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using Foam::Info;
using Foam::endl;
using Foam::nl;

namespace fvc = Foam::fvc;
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

// Mirrors register_computed_scalar_field: give a freshly-built field old-time storage by
// registering it in the VectorCollection under its own name.
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

// out = 0.5*magSqr(U), internal values only (K only ever feeds EEqn's explicit terms,
// which read internal values; matches the Python port's own scope).
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

// out = a*b, internal values only (both of our uses -- psi*p for psip0/rho, p*ddt_psi for
// pEqn's explicit source -- only ever need the internal field).
void multiplyFields(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& a,
    const fvcc::VolumeField<NeoN::scalar>& b,
    fvcc::VolumeField<NeoN::scalar>& out
)
{
    const auto [aV, bV] = NeoN::views(a.internalVector(), b.internalVector());
    auto outV = out.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, a.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] * bV[i]; },
        "multiplyFields"
    );
}

// out = a*b, internal AND boundary values -- for products that go on to be face-interpolated
// (rho*rAU below), unlike multiplyFields's internal-only scope.
void multiplyFieldsWithBoundary(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& a,
    const fvcc::VolumeField<NeoN::scalar>& b,
    fvcc::VolumeField<NeoN::scalar>& out
)
{
    multiplyFields(exec, a, b, out);
    const auto [aV, bV] = NeoN::views(a.boundaryData().value(), b.boundaryData().value());
    auto outV = out.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, a.boundaryData().value().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { outV[i] = aV[i] * bV[i]; },
        "multiplyFieldsWithBoundary::boundary"
    );
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

// rho += psi*p - psip0, internal values only -- the incremental EOS update inside the PISO
// loop (mirrors literal rhoPimpleFoam's psip0 = psi*p / thermo.correctRho(...) pattern, see
// the Python port's module docstring).
void incrementRho(
    const NeoN::Executor& exec,
    const fvcc::VolumeField<NeoN::scalar>& psi,
    const fvcc::VolumeField<NeoN::scalar>& p,
    const fvcc::VolumeField<NeoN::scalar>& psip0,
    fvcc::VolumeField<NeoN::scalar>& rho
)
{
    const auto [psiV, pV, psip0V] =
        NeoN::views(psi.internalVector(), p.internalVector(), psip0.internalVector());
    auto rhoV = rho.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, psi.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rhoV[i] += psiV[i] * pV[i] - psip0V[i]; },
        "incrementRho"
    );
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

        // Inner-loop counting (nCorrectors, nNonOrthogonalCorrectors) lives in the "PIMPLE"
        // subdict; read from there so a stock rhoPimpleFoam case with no "PISO" block works.
        Foam::pisoControl piso(mesh, "PIMPLE");

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        for (const std::string& name :
             {"p", "U", "rho", "T", "he", "pFinal", "UFinal", "rhoFinal", "TFinal", "heFinal"})
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
        // list reworked, not just run through unchanged (same caveat as the Python port).
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
        fvcc::VolumeField<NeoN::scalar> heSeed(rt.exec, "he", rt.nfMesh, heBcs);
        heFromT(rt.exec, T, gas.Cv, heSeed);
        auto& he = registerComputedScalar(vectorCollection, heSeed, "he");
        fvcc::mapEnergyBoundaryFromTemperature(T.boundaryData(), he.boundaryData(), gas.Cv, TStd);
        he.correctBoundaryConditions();
        fvcc::oldTime(he);

        auto psiSeed = createUniformVolumeScalar(rt, "psi", 0.0);
        computePsi(rt.exec, T, gas.R, psiSeed);
        auto& psi = registerComputedScalar(vectorCollection, psiSeed, "psi");

        auto rhoSeed = createUniformVolumeScalar(rt, "rho", 0.0);
        computeRho(rt.exec, psi, p, rhoSeed);
        auto& rho = registerComputedScalar(vectorCollection, rhoSeed, "rho");

        auto KSeed = createUniformVolumeScalar(rt, "K", 0.0);
        kineticEnergy(rt.exec, U, KSeed);
        auto& K = registerComputedScalar(vectorCollection, KSeed, "K");

        fvcc::oldTime(psi);
        fvcc::oldTime(rho);
        fvcc::oldTime(K);

        // he (internal energy) is being diffused here, not T -- kappa/Cv (gas.alphaHe), not
        // kappa/Cp; see PerfectGasConstTransport's own derivation.
        auto muVol = createUniformVolumeScalar(rt, "mu", gas.mu);
        auto zeroNut = createUniformVolumeScalar(rt, "nut0", 0.0);
        auto alphaEff = createUniformSurfaceScalar(rt, "alphaEff", gas.alphaHe);
        // Feature-for-feature port of literal rhoPimpleFoam's psip0 = psi*p / incremental
        // thermo.correctRho() update -- captured once per PISO pass.
        auto psip0 = createUniformVolumeScalar(rt, "psip0", 0.0);
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
        fvcc::oldTime(phi);

        nf::PimpleControl pimpleLoop(rt.fvSolutionDict);

        // Max-component reduction: reduce Ux/Uy/Uz entries to a single {init, final} pair.
        auto reduceU = [](const NeoN::la::SolverStats& s) -> std::pair<NeoN::scalar, NeoN::scalar>
        {
            NeoN::scalar mi = 0.0;
            NeoN::scalar mf = 0.0;
            for (const auto& e : s.entries)
            {
                mi = std::max(mi, e.initResNorm);
                mf = std::max(mf, e.finalResNorm);
            }
            return {mi, mf};
        };

        NeoN::scalar cumulativeContErr = 0.0;

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            NeoN::Logging::info("Time = {}", rt.t);

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(T);
            fvcc::rotateOldTimes(he);
            fvcc::rotateOldTimes(p);
            fvcc::rotateOldTimes(rho);
            fvcc::rotateOldTimes(psi);
            fvcc::rotateOldTimes(K);
            fvcc::rotateOldTimes(phi);

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            nf::ResidualMap residuals;
            while (pimpleLoop.loop(residuals))
            {
                const bool finalIter = pimpleLoop.finalIter();

                // Feature-for-feature port of literal rhoPimpleFoam.C's
                // `if (pimple.firstIter() && !pimple.SIMPLErho())`: transient PIMPLE is never
                // SIMPLErho, so this always runs on the first outer iteration of every
                // timestep, advancing rho via mass conservation using the PREVIOUS timestep's
                // converged phi -- distinct from, and run before, the EOS-based rho update
                // later in the loop.
                if (pimpleLoop.firstIter())
                {
                    nf::PDESolver<NeoN::scalar> rhoEqn(
                        dsl::imp::ddt(rho) + dsl::exp::div(phi),
                        rho,
                        rt
                    );
                    rhoEqn.solve();
                    rho.correctBoundaryConditions();
                }

                auto prevP = NeoN::dsl::fieldRelaxationSnapshot(p);

                fvcc::VolumeField<NeoN::Tensor> gradU(
                    rt.exec,
                    "gradU",
                    rt.nfMesh,
                    fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(rt.nfMesh)
                );
                nf::gradScheme(rt, "grad(U)").gradTensor(U, gradU, dsl::Coeff {});
                gradU.correctBoundaryConditions();

                // Literal's divDevRhoReff coefficient is alpha*rho*nuEff(), and for this
                // laminar (Stokes) model nuEff()==nu()==mu/rho, so the coefficient collapses
                // to mu -- NOT rho*mu -- for both the implicit laplacian and the explicit
                // dev2 correction below.
                auto muFace = surfInterpol.interpolate(muVol);
                muFace.name = "mu";

                nf::PDESolver<NeoN::Vec3> UEqn(
                    dsl::imp::ddt(rho, U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(muFace, U)
                        + dsl::exp::viscousStress(muVol, zeroNut, gradU, true),
                    U,
                    rt
                );

                const auto ddtScheme = UEqn.ddtScheme();
                NF_ASSERT(
                    ddtScheme != fvcc::DdtScheme::None,
                    "neorhoPimpleFoam: steadyState ddt unsupported (BDF1/BDF2 only)"
                );

                UEqn.setFinalIter(finalIter);
                UEqn.relax();

                if (piso.momentumPredictor())
                {
                    auto statsU = UEqn.solve(-1.0 * dsl::exp::grad(p));
                    residuals["U"] = reduceU(statsU);
                    kineticEnergy(rt.exec, U, K);
                }
                else
                {
                    UEqn.assembleAndRelax();
                }

                auto prevHe = NeoN::dsl::fieldRelaxationSnapshot(he);

                // Live per-outer-iteration refresh, matching literal's actual
                // fixedEnergy/gradientEnergy mechanism (recomputed every call, not captured
                // once) -- push T's current boundary state onto he's refValue/refGrad.
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

                // The two TemporalOperator terms (imp::ddt, exp::ddt) must be combined with
                // each other FIRST: NeoN's public dsl headers only overload Expression+/-
                // SpatialOperator and the generic Operator+Operator template (both operands
                // genuine Operator types with a VectorValueType member) -- there is no native
                // Expression+TemporalOperator overload outside the Python-bindings file, so
                // `dsl::imp::ddt(...) + dsl::exp::div(...)` directly would not compile here.
                nf::PDESolver<NeoN::scalar> EEqn(
                    (dsl::imp::ddt(rho, he) + dsl::exp::ddt(rho, K)) + dsl::imp::div(phi, he)
                        - dsl::imp::laplacian(alphaEff, he) + dsl::exp::div(phi, K)
                        + dsl::exp::div(phiv, p),
                    he,
                    rt
                );
                EEqn.setFinalIter(finalIter);
                EEqn.relax();

                auto statsT = EEqn.solve();
                residuals["T"] = {statsT.entries[0].initResNorm, statsT.entries[0].finalResNorm};
                NeoN::dsl::applyFieldRelaxation(
                    he,
                    prevHe,
                    nf::lookupFieldRelaxation(rt.fvSolutionDict, he.name, finalIter).value_or(1.0)
                );
                he.correctBoundaryConditions();
                // Convert he back to T for the EOS/reporting.
                temperatureFromHe(rt.exec, he, gas.Cv, T);
                computePsi(rt.exec, T, gas.R, psi);
                computeRho(rt.exec, psi, p, rho);

                std::pair<NeoN::scalar, NeoN::scalar> pRes {};
                bool havePRes = false;

                while (piso.correct())
                {
                    multiplyFields(rt.exec, psi, p, psip0);
                    auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                    nf::constrainHbyA(U, p, hByA);

                    auto rhoFace = surfInterpol.interpolate(rho);
                    multiplyFieldsWithBoundary(rt.exec, rho, crAU, rhoCrAU);
                    auto rhorAUf = surfInterpol.interpolate(rhoCrAU);
                    rhorAUf.name = "rhorAUf";
                    auto phiHbyA = rhoFace * nf::flux(hByA);

                    while (piso.correctNonOrthogonal())
                    {
                        // pEqn built as +ddt(psi,p) + div(phiHbyA) - laplacian(rhorAUf,p)
                        // (literal rhoPimpleFoam's own pDDtEqn - fvm::laplacian ordering,
                        // needed to keep the matrix positive-definite for p's DIC/IC
                        // preconditioner) -- the opposite overall sign from NeoFOAM's
                        // incompressible solvers' +laplacian(...) - div(...) convention.
                        // Must be told explicitly, or update_face_velocity's default
                        // (calibrated for the incompressible convention) would reconstruct
                        // phi with the wrong sign.
                        nf::PDESolver<NeoN::scalar> pEqn(
                            dsl::imp::ddt(psi, p) + dsl::exp::div(phiHbyA)
                                - dsl::imp::laplacian(rhorAUf, p),
                            p,
                            rt
                        );
                        pEqn.setFinalIter(finalIter);

                        if (ofP.needReference() && pRefCell >= 0)
                        {
                            pEqn.setReference(pRefCell, pRefValue);
                        }

                        auto statsP = pEqn.solve();
                        if (!havePRes)
                        {
                            pRes = {statsP.entries[0].initResNorm, statsP.entries[0].finalResNorm};
                            havePRes = true;
                        }
                        p.correctBoundaryConditions();

                        if (piso.finalNonOrthogonalIter())
                        {
                            nf::updateFaceVelocity(
                                phiHbyA,
                                pEqn,
                                phi,
                                nf::PEqnConvention::DivMinusLaplacian
                            );
                        }
                    }

                    NeoN::dsl::applyFieldRelaxation(
                        p,
                        prevP,
                        nf::lookupFieldRelaxation(rt.fvSolutionDict, p.name, finalIter)
                            .value_or(1.0)
                    );
                    p.correctBoundaryConditions();

                    nf::reportContinuityError(phi, rt, cumulativeContErr);

                    nf::updateVelocity(hByA, crAU, p, U, rt);
                    U.correctBoundaryConditions();
                    // Feature-for-feature port: literal pEqn.H also refreshes K here, from
                    // the just-pressure-corrected U -- not just once after the momentum
                    // predictor.
                    kineticEnergy(rt.exec, U, K);
                    incrementRho(rt.exec, psi, p, psip0, rho);
                }

                if (havePRes)
                {
                    residuals["p"] = pRes;
                }
            }

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
