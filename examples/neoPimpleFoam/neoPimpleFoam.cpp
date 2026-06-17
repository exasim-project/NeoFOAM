// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*---------------------------------------------------------------------------*\
Application
    neoPimpleFoam

Description
    Transient incompressible PIMPLE solver (PISO inner correctors inside an outer
    under-relaxed PIMPLE loop), the NeoFOAM/NeoN port of OpenFOAM's pimpleFoam.

    v1 validity envelope:

    IN (supported / validated):
      - Closed-domain setReference cases (e.g. lid-driven cavity)
      - Fixed-pressure-outlet cases
      - Transient BDF1 / BDF2 ddt schemes
      - CPU-serial execution

    OUT (not supported in v1; deferred to v2):
      - adjustPhi (fixed-flux inlet/outlet continuity)
      - constrainPressure (non-orthogonal / rotating pressure BCs)
      - steadyState / Crank-Nicolson ddt schemes
      - GPU + MPI distributed parity
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"

#include "fvCFD.H"
#include "pisoControl.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"
#include "LESModel.H"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>

using Foam::Info;
using Foam::endl;
using Foam::nl;

// NOTE about namespace usage
// here the namespaces are used deliberately verbose to
// demonstrate where things are implemented
namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
// Bring up OpenFOAM (and MPI) before NeoN, matching neoIcoFoam/neoPisoFoam, so the rank
// is known when NeoN configures logging and the rank-0 muting engages immediately.
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime);
        auto& mesh = rt.mesh;

        // Inner pressure-corrector / non-orthogonal corrector counting stays on
        // Foam::pisoControl: the native-loop rationale is specific to the
        // residual-fed OUTER loop; the inner loop is pure counting with no residual round-trip.
        //
        // MUST read the inner-corrector counts from the "PIMPLE" subdict, NOT the default
        // "PISO" subdict: OpenFOAM's pimpleFoam puts nCorrectors / nNonOrthogonalCorrectors /
        // momentumPredictor in PIMPLE (via pimpleControl), and a stock pimpleFoam case (like the
        // parity fixtures) has no PISO block. With the default "PISO" name pisoControl reads an
        // absent dict and nCorrectors silently defaults to 1 -> only ONE pressure correction per
        // step -> the velocity projection is incomplete, continuity grows geometrically, and the
        // run diverges on a case where pimpleFoam is stable. pisoControl forwards the dictName to
        // pimpleControl, whose correct() uses nCorrPISO_ = PIMPLE.nCorrectors and resets cleanly
        // per inner loop, so it composes correctly with the nf::PimpleControl outer loop below.
        Foam::pisoControl piso(mesh, "PIMPLE");

#include "createFields.H"

        // Map the base solver subdicts AND any *Final subdicts so the final-pass
        // <field>Final selection in PDESolver reads a converted (NeoN-format) dict. isDict-guard
        // the *Final mapping — a missing pFinal/UFinal degrades to the base entry.
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        // The nuTilda solve inside turb.correct() needs its solver subdict converted too.
        solverDict.subDict("nuTilda") = nf::mapFvSolution(solverDict.subDict("nuTilda"));
        if (solverDict.isDict("pFinal"))
        {
            solverDict.subDict("pFinal") = nf::mapFvSolution(solverDict.subDict("pFinal"));
        }
        if (solverDict.isDict("UFinal"))
        {
            solverDict.subDict("UFinal") = nf::mapFvSolution(solverDict.subDict("UFinal"));
        }
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);
        auto& nuTilda = nf::constructAndRegister(vectorCollection, rt, ofNuTilda, false);
        auto nut = nf::constructAndRegister(vectorCollection, rt, ofNut, false);

        // Gauss-Green gradient operator for the explicit deviatoric viscous-stress term.
        // OpenFOAM's pimpleFoam momentum equation is
        //   ddt(U) + div(phi,U) + divDevReff(U)
        //     = ddt(U) + div(phi,U) - laplacian(nuEff,U) - div(nuEff*dev2(T(grad U))),
        // so the implicit laplacian alone is only PART of the viscous term. The explicit dev2
        // stress div((nuEff*dev2(T(grad(U))))) must be added for OF parity — it is non-zero
        // wherever the reconstructed cell velocity has non-zero divergence. gradU is recomputed
        // LOCALLY each outer corrector so the explicit stress tracks the current U; this
        // is NOT turb.gradU(), which is frozen per step.
        fvcc::GaussGreenGrad gradOp(rt.exec, rt.nfMesh);

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        // Turbulence model setup (SA-DDES). nu is the laminar viscosity from the OF transport
        // model; wallDist / nearWallDist / delta feed the DDES shielding + wall functions.
        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());
        auto wallDist = nf::constructFrom(rt.exec, rt.nfMesh, y.y());
        auto nearWallDist = nf::constructFrom(rt.exec, rt.nfMesh, ofNearWallDist);
        auto delta = nf::constructFrom(rt.exec, rt.nfMesh, lesModel.delta());
        nf::SpalartAllmarasDDES turb(rt.exec, rt.nfMesh, nu, wallDist, nearWallDist, delta);

        // Calculate prerequisite variables for the turbulence model (gradU, diffusion coeff,
        // and an initial correctNut). Mirrors OpenFOAM's turbulence->validate().
        turb.validate(U, nuTilda, nut);

        // Hoist the surface interpolation once (constructed per inner corrector otherwise),
        // mirroring neoPisoFoam.cpp:77-81.
        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );

        NeoN::scalar cumulativeContErr = 0.0;

        // Construct the native outer-loop control ONCE before the time loop. It reads the
        // PIMPLE subdict (nOuterCorrectors, residualControl) from fvSolution.
        nf::PimpleControl pimpleLoop(rt.fvSolutionDict);

        // reduce the segregated Vec3 U solver stats (Ux/Uy/Uz entries) to a single
        // {init, final} residual pair via MAX-COMPONENT, mirroring OpenFOAM's maxResidual.
        // Degenerates to the single entry when stats.entries.size() == 1.
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

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            // Logging supports string formatting
            NeoN::Logging::info("Time = {}", rt.t);

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(phi);
            fvcc::rotateOldTimes(nuTilda);

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            // --- Outer PIMPLE corrector loop. `residuals` carries the
            // {initResNorm, finalResNorm} per field from the PREVIOUS outer pass; the
            // control checks them (skipping the first pass) to decide an early exit.
            nf::ResidualMap residuals;
            while (pimpleLoop.loop(residuals))
            {
                NeoN::Logging::info("PIMPLE outer corrector");
                const bool finalIter = pimpleLoop.finalIter();

                // Snapshot p at the TOP of each outer corrector (deep copy of
                // the internal vector) so we can blend against it after the pressure solve.
                auto prevP = NeoN::dsl::fieldRelaxationSnapshot(p);

                // Momentum predictor. gradU (velocity-gradient tensor) is recomputed LOCALLY from
                // the current U each outer corrector (the OF-parity point); the dev2 viscous
                // stress is explicit (evaluated at assembly, like OpenFOAM's divDevReff). The
                // implicit laplacian uses the SA-DDES effective surface viscosity turb.nuEff()
                // (nu + nut, nut no longer zero), while the explicit stress uses the turbulence
                // volume nu/nut coefficients with this LOCAL gradU — deliberately NOT turb.gradU(),
                // which is frozen per step.
                auto gradU = gradOp.gradTensor(U);
                nf::PDESolver<NeoN::Vec3> UEqn(
                    dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(turb.nuEff(), U)
                        + dsl::exp::viscousStress(nu, nut, gradU),
                    U,
                    rt
                );

                const auto ddtScheme = UEqn.ddtScheme();
                // v1 is BDF1/BDF2 only — fail loud on steadyState ddt
                // (it would silently produce garbage). adjustPhi/constrainPressure are v2 TODOs.
                // NOTE: NF_ASSERT (abort + stack trace) is used instead of NF_ASSERT_THROW because
                // the NeoN NF_ASSERT_THROW macro miss-composes its message (it wraps the assertion
                // text in std::string("..." << message), shifting two const char* operands, which
                // does not compile) — and src/NeoN is treated as frozen here. NF_ASSERT gives the
                // same fail-loud guarantee.
                NF_ASSERT(
                    ddtScheme != fvcc::DdtScheme::None,
                    "neoPimpleFoam: steadyState ddt unsupported in v1 (BDF1/BDF2 only)"
                );

                // Drive the final-pass *Final relaxation + <field>Final solver subdict.
                UEqn.setFinalIter(finalIter);

                if (piso.momentumPredictor())
                {
                    // this solve(rhs) overload applies momentum eqn-URF +
                    // UFinal solver-subdict selection.
                    auto statsU = UEqn.solve(-1.0 * dsl::exp::grad(p));
                    residuals["U"] = reduceU(statsU); // max-component reduction
                }
                else
                {
                    // OF-parity: OpenFOAM's pimpleFoam applies UEqn.relax()
                    // UNCONDITIONALLY, including the `momentumPredictor no` path.
                    // assembleAndRelax() assembles AND relaxes the owned ls_ in place (the same
                    // lookup+relax solve(rhs) runs internally) so computeRAUandHByA reads the
                    // RELAXED augmented diagonal in BOTH configs (the rAU/HbyA read invariant), not
                    // just when the predictor solves. With no U eqn-URF configured, alpha==1 makes
                    // this a bitwise no-op (identical to a bare assemble()), so
                    // neoIcoFoam/neoPisoFoam semantics are unchanged.
                    UEqn.assembleAndRelax();
                }

                std::pair<NeoN::scalar, NeoN::scalar> pRes {};
                bool havePRes = false;

                // --- PISO inner-corrector loop (Foam::pisoControl)
                while (piso.correct())
                {
                    NeoN::Logging::info("PISO loop");
                    auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                    nf::constrainHbyA(U, p, hByA);

                    nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAU);
                    rAU.name = "rAUf";

                    auto phiHbyA =
                        nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);

                    // TODO additionally missing
                    // TODO adjustPhi(phiHbyA, U, p);            // v2 — not yet in NeoN
                    // TODO constrainPressure(p, U, phiHbyA, rAU); // v2 — not yet in NeoN

                    // Non-orthogonal pressure corrector loop
                    while (piso.correctNonOrthogonal())
                    {
                        // Pressure corrector
                        nf::PDESolver<NeoN::scalar> pEqn(
                            NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                            p,
                            rt
                        );

                        pEqn.setFinalIter(finalIter);

                        // re-apply the setReference pin on EVERY pressure solve. The
                        // rank-0 guard lives in PDESolver::setReference (preserve, do NOT
                        // refactor).
                        if (ofP.needReference() && pRefCell >= 0)
                        {
                            pEqn.setReference(pRefCell, pRefValue);
                        }

                        auto statsP = pEqn.solve();
                        if (!havePRes)
                        {
                            // feed the FIRST pressure solve's residual into the NEXT loop().
                            pRes = {statsP.entries[0].initResNorm, statsP.entries[0].finalResNorm};
                            havePRes = true;
                        }
                        p.correctBoundaryConditions();

                        if (piso.finalNonOrthogonalIter())
                        {
                            nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                        }
                    }

                    // relax p AFTER the pressure solve +
                    // correctBoundaryConditions, internal-only blend. The final pass falls back to
                    // alpha=1 via the *Final key (a bitwise no-op).
                    NeoN::dsl::applyFieldRelaxation(
                        p,
                        prevP,
                        nf::lookupFieldRelaxation(rt.fvSolutionDict, p.name, finalIter)
                            .value_or(1.0)
                    );
                    p.correctBoundaryConditions();

                    nf::reportContinuityError(phi, rt, cumulativeContErr);

                    nf::updateVelocity(hByA, crAU, p, U);
                    U.correctBoundaryConditions();
                }

                if (havePRes)
                {
                    residuals["p"] = pRes; // fed into the NEXT loop() call
                }
            }

            // Turbulence update: solve the nuTilda PDE + refresh nut ONCE PER STEP, after
            // the outer PIMPLE loop has closed (turbOnFinalIterOnly=true semantics).
            turb.correct(U, phi, nuTilda, nut, rt);

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
                NeoN::Logging::info("Writing turbulence variables");
                write(nuTilda, mesh);
                write(nut, mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
