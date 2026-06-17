// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// End-to-end CPU-serial OpenFOAM parity test (the acceptance bar).
//
// This test runs the REAL OpenFOAM pimpleFoam binary live on the shared, in-envelope
// lid-driven-cavity case (test/setup_pimple, extended to convergence), preserves the
// converged INTERNAL p/U reference in a sidecar BEFORE the in-process neoPimpleFoam loop runs
// (the in-process loop would otherwise overwrite the endTime time dir), runs the
// neoPimpleFoam outer-PIMPLE / inner-PISO loop in-process on the same cavity, and asserts the
// converged INTERNAL p/U fields agree at ~1e-10 via Catch2 EqualsInternal +
// ApproxScalar/ApproxVector. Linear-solver tolerances in fvSolution are set tighter than the
// parity epsilon and L1Stop is activated config-only so NeoN's residual matches OF's.
//
// CPU-serial only (validates CPU-serial parity; the assertion is executor-
// independent; a single MeshAdapter avoids the documented second-MeshAdapter teardown SIGSEGV).
// This is a NeoFOAM/case-config-only test.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "findRefCell.H"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"
#include "pisoControl.H"

#include <cstdlib>

namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // the single Foam::Time from catch2/test_main.cpp

TEST_CASE("neoPimpleFoam converges to OpenFOAM pimpleFoam on the cavity", "[pimpleParity]")
{
    Foam::Time& runTime = *timePtr;

    // ---------------------------------------------------------------------------------------------
    // STEP 1 — Run the REAL pimpleFoam binary live on the shared case dir. The ctest
    // WORKING_DIRECTORY is ${CMAKE_SOURCE_DIR}/test/setup_pimple, so the binary sees the byte-
    // identical system/+constant/+0/ that the in-process loop reads. The catch2 main already
    // ran blockMesh when constant/polyMesh/points was absent. Mirrors test_main.cpp:75.
    // ---------------------------------------------------------------------------------------------
    int rc = std::system("pimpleFoam > log.pimpleFoam 2>&1");
    REQUIRE(
        rc == 0
    ); // ensure OpenFOAM 2412 is sourced and pimpleFoam is on PATH; see log.pimpleFoam

    // ---------------------------------------------------------------------------------------------
    // STEP 2 — Preserve the converged OF reference in a SIDECAR dir BEFORE the in-process loop runs
    // STEP 3 writes its OWN fields into the SAME endTime dir, which would
    // corrupt the EqualsInternal expected reference. Copy the latest converged OF time dir's p/U
    // into endTime_OF/ (fixed-literal shell-out, mirroring STEP 1; in the accepted threat model),
    // then deep-copy into NO_REGISTER OF field objects that survive every later time-dir write.
    // ---------------------------------------------------------------------------------------------
    int cpRc = std::system("rm -rf endTime_OF && mkdir -p endTime_OF && "
                           "cp -r $(foamListTimes -latestTime 2>/dev/null | tail -1)/p "
                           "$(foamListTimes -latestTime 2>/dev/null | tail -1)/U endTime_OF/");
    REQUIRE(cpRc == 0); // sidecar holds the untouchable converged OF p/U

    auto exec = NeoN::Executor(NeoN::SerialExecutor {});
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    // Reconstruct OF reference fields from the sidecar; deep-copy the primitive data so the
    // assertion in STEP 4 is immune to the in-process loop overwriting the case time dir.
    // Select the converged endTime so the IOobject time-instance lookup resolves the field class.
    runTime.setTime(runTime.times().last(), runTime.times().size() - 1);
    Foam::volScalarField ofPRefRaw(
        Foam::IOobject(
            "p",
            "endTime_OF",
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh
    );
    Foam::volVectorField ofURefRaw(
        Foam::IOobject(
            "U",
            "endTime_OF",
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh
    );
    // Deep-copy into stable reference objects (own their own primitiveField copy).
    Foam::volScalarField ofPRef(ofPRefRaw);
    Foam::volVectorField ofURef(ofURefRaw);

    // ---------------------------------------------------------------------------------------------
    // STEP 3 — Run the neoPimpleFoam loop IN-PROCESS on the same case. Rewind the time to startTime
    // so the loop reads 0/{p,U}, and GUARD the rewind. Then mirror
    // examples/neoPimpleFoam/neoPimpleFoam.cpp (lines 56-291) + createFields.H verbatim.
    // ---------------------------------------------------------------------------------------------
    runTime.setTime(0.0, 0);
    REQUIRE(
        runTime.timeName() == Foam::word("0")
    ); // OF 2412 Foam::Time rewind landed at startTime "0"

    // Inner-corrector counts (nCorrectors / nNonOrthogonalCorrectors / momentumPredictor) live in
    // the "PIMPLE" subdict for a stock pimpleFoam case (there is no "PISO" block). pisoControl
    // defaults to reading "PISO": with it absent, nCorrectors silently degrades to 1, giving a
    // single under-converged pressure correction per step and a geometric continuity blow-up on a
    // case where pimpleFoam is stable. Read from "PIMPLE" to match OpenFOAM pimpleFoam (mirrors the
    // neoPimpleFoam app fix).
    Foam::pisoControl piso(mesh, "PIMPLE");

    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

    // Map the base solver subdicts AND the *Final subdicts (so the decisive final outer-corrector
    // pass selects a converted, L1-carrying dict). isDict-guard the *Final mapping.
    {
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        if (solverDict.isDict("pFinal"))
        {
            solverDict.subDict("pFinal") = nf::mapFvSolution(solverDict.subDict("pFinal"));
        }
        if (solverDict.isDict("UFinal"))
        {
            solverDict.subDict("UFinal") = nf::mapFvSolution(solverDict.subDict("UFinal"));
        }
    }

    // --- createFields.H (read fields from 0/, set the reference cell from the PIMPLE subdict) ---
    Foam::IOdictionary transportProperties(Foam::IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        Foam::IOobject::MUST_READ_IF_MODIFIED,
        Foam::IOobject::NO_WRITE
    ));
    Foam::dimensionedScalar viscosity("nu", Foam::dimViscosity, transportProperties);

    Foam::volScalarField ofP(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh
    );
    Foam::volVectorField ofU(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh
    );
    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        fvc::flux(ofU)
    );

    Foam::label pRefCell = 0;
    Foam::scalar pRefValue = 0.0;
    Foam::setRefCell(ofP, mesh.solutionDict().subDict("PIMPLE"), pRefCell, pRefValue);

    fvcc::VectorCollection& vectorCollection =
        fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
    auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

    auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::SurfaceField<NeoN::scalar> nu(rt.exec, "nu", rt.nfMesh, nuBCs);
    NeoN::fill(nu.internalVector(), viscosity.value());
    NeoN::fill(nu.boundaryData().value(), viscosity.value());

    // Volume nu / nut(=0) + gradient operator for the laminar dev2 viscous stress (matches the
    // neoPimpleFoam app): pimpleFoam's divDevReff(U) = -laplacian(nuEff,U) - div(nuEff*dev2(T(grad
    // U))), so the implicit laplacian alone misses the explicit dev2 stress — required for OF
    // parity.
    auto nuVolBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::VolumeField<NeoN::scalar> nuVol(rt.exec, "nu", rt.nfMesh, nuVolBCs);
    NeoN::fill(nuVol.internalVector(), viscosity.value());
    NeoN::fill(nuVol.boundaryData().value(), viscosity.value());
    fvcc::VolumeField<NeoN::scalar> nutVol(rt.exec, "nut", rt.nfMesh, nuVolBCs);
    NeoN::fill(nutVol.internalVector(), 0.0);
    NeoN::fill(nutVol.boundaryData().value(), 0.0);
    fvcc::GaussGreenGrad gradOp(rt.exec, rt.nfMesh);

    auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

    auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
        rt.exec,
        rt.nfMesh,
        NeoN::TokenList({std::string("linear")})
    );

    NeoN::scalar cumulativeContErr = 0.0;

    nf::PimpleControl pimpleLoop(rt.fvSolutionDict);

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

    // --- Time loop (verbatim mirror of neoPimpleFoam.cpp lines 137-291) ---
    while (runTime.loop())
    {
        fvcc::rotateOldTimes(U);
        fvcc::rotateOldTimes(phi);

        auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
        nf::syncRunTimes(runTime, rt, maxCoNum);

        nf::ResidualMap residuals;
        while (pimpleLoop.loop(residuals))
        {
            const bool finalIter = pimpleLoop.finalIter();

            auto prevP = NeoN::dsl::fieldRelaxationSnapshot(p);

            auto gradU = gradOp.gradTensor(U);
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U)
                    + dsl::exp::viscousStress(nuVol, nutVol, gradU),
                U,
                rt
            );

            const auto ddtScheme = UEqn.ddtScheme();
            NF_ASSERT(
                ddtScheme != fvcc::DdtScheme::None,
                "neoPimpleFoam: steadyState ddt unsupported in v1 (BDF1/BDF2 only)"
            );

            UEqn.setFinalIter(finalIter);

            if (piso.momentumPredictor())
            {
                auto statsU = UEqn.solve(-1.0 * dsl::exp::grad(p));
                residuals["U"] = reduceU(statsU);
            }
            else
            {
                UEqn.assembleAndRelax();
            }

            std::pair<NeoN::scalar, NeoN::scalar> pRes {};
            bool havePRes = false;

            while (piso.correct())
            {
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAU);
                rAU.name = "rAUf";

                auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);

                while (piso.correctNonOrthogonal())
                {
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
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
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                    }
                }

                NeoN::dsl::applyFieldRelaxation(
                    p,
                    prevP,
                    nf::lookupFieldRelaxation(rt.fvSolutionDict, p.name, finalIter).value_or(1.0)
                );
                p.correctBoundaryConditions();

                nf::reportContinuityError(phi, rt, cumulativeContErr);

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
            }

            if (havePRes)
            {
                residuals["p"] = pRes;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // STEP 4 — comparison: assert the converged NeoFOAM INTERNAL fields equal the PRESERVED
    // live-pimpleFoam reference (the deep-copied STEP 2 ofPRef/ofURef, NOT a re-read of the case
    // dir that STEP 3 overwrote) at ~1e-10. Boundary patches follow the zeroGradient-skip
    // convention (assert INTERNAL only, not boundary, at the tight margin).
    // ---------------------------------------------------------------------------------------------
    REQUIRE_THAT(p, EqualsInternal(ofPRef, ApproxScalar {1e-10}));
    REQUIRE_THAT(U, EqualsInternal(ofURef, ApproxVector({1e-10, 1e-10, 1e-10})));
}
