// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

// Kokkos profiling regions (Kokkos::Profiling::ScopedRegion / push/popRegion). These hooks are part
// of Kokkos core and are a NO-OP unless a kokkos-tools connector is loaded at runtime via
// KOKKOS_TOOLS_LIBS -- so the annotations below are independent of the NEOFOAM_ENABLE_KOKKOS_TOOLS
// CMake option (which only controls whether the connector libraries are built). The solver compiles
// and runs identically whether that option is ON or OFF.
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "fvCFD.H"
#include "simpleControl.H"
#include "singlePhaseTransportModel.H"

#if NF_WITH_UMPIRE
#include "NeoN/core/memory/umpire.hpp"
#endif

#include <memory>

using Foam::Info;
using Foam::endl;
using Foam::nl;

// NOTE about namespace usage
// here the namespaces are used deliberately verbose to
// demonstrate where things are implemented
namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace
{

// Resolve the field under-relaxation factor (alpha) for the given field name
// from fvSolution.relaxationFactors.fields.<name>. Returns 1.0 (a bitwise
// no-op inside applyFieldRelaxation) if the block, the fields subdict, or the
// entry is absent. Tolerates int / scalar typing the same way the equation
// factor lookup in PDESolver does.
NeoN::scalar
readFieldRelaxationFactor(const NeoN::Dictionary& fvSolutionDict, const std::string& fieldName)
{
    if (!fvSolutionDict.contains("relaxationFactors")) return NeoN::scalar(1);
    const auto& rf = fvSolutionDict.subDict("relaxationFactors");
    if (!rf.contains("fields")) return NeoN::scalar(1);
    const auto& fields = rf.subDict("fields");
    if (!fields.contains(fieldName)) return NeoN::scalar(1);
    if (fields.isType<int>(fieldName)) return NeoN::scalar(fields.get<int>(fieldName));
    return fields.get<NeoN::scalar>(fieldName);
}

} // namespace

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
// Bring up OpenFOAM (and MPI) before NeoN, matching neoIcoFoam, so the rank is
// known when NeoN configures logging and the rank-0 muting engages immediately.
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        // One-time setup phase (mesh adapter, field creation, scheme/solver mapping, turbulence
        // model). push/popRegion rather than a ScopedRegion because the objects created here outlive
        // the phase -- they are used throughout the time loop below.
        Kokkos::Profiling::pushRegion("neoSimpleFoam.setup");
        auto rt = nf::createAdapterRunTime(runTime);
        auto& mesh = rt.mesh;

        Foam::simpleControl simple(mesh);

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());

        auto turb = nf::TurbulenceModel::create(rt, nu);

        turb->validate(U);

        // TODO: surface interpolation also instantiated in turbulence model -> doubled?!
        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );
        NeoN::scalar cumulativeContErr = 0.0;

        // TEMP diagnostic: report the shared Umpire DEVICE_POOL footprint at each phase so we can
        // tell a true leak (current/live bytes grow across timesteps) from QuickPool fragmentation
        // (actual/reserved bytes balloon while current stays flat). Ginkgo and NeoN draw from this
        // same pool. Remove once the OOM is understood.
        auto logGpuMem = [](const char* tag)
        {
#if NF_WITH_UMPIRE && defined(KOKKOS_ENABLE_CUDA)
            try
            {
                auto pool = NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
                NeoN::Logging::info(
                    "[mem] {}: DEVICE_POOL current={} MB  actual(reserved)={} MB  highWater={} MB",
                    tag,
                    pool.getCurrentSize() >> 20,
                    pool.getActualSize() >> 20,
                    pool.getHighWatermark() >> 20
                );
            }
            catch (...)
            {
                // pool not created yet (no device allocation has happened) — ignore
            }
#else
            static_cast<void>(tag);
#endif
        };

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        Kokkos::Profiling::popRegion(); // end neoSimpleFoam.setup

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            // Per-timestep region. The phase regions below nest under it in the profiler output.
            Kokkos::Profiling::ScopedRegion timeStepRegion("neoSimpleFoam.timeStep");

            rt.t = runTime.time().value();
            NeoN::Logging::info("Time = {}", rt.t);
            logGpuMem("step-start");

            // Steady-state: no rotateOldTimes, no Courant number, no syncRunTimes

            // Momentum predictor (no ddt for steady-state SIMPLE). push/popRegion (not a
            // ScopedRegion) because UEqn outlives this phase -- the pressure corrector below reuses
            // it via computeRAUandHByA(UEqn).
            Kokkos::Profiling::pushRegion("neoSimpleFoam.momentumPredictor");
            // Sub-region for the equation construction: builds the dsl expression and, notably,
            // evaluates turb->gradU() (velocity gradient) + nuEff/nut. Brackets it separately from
            // the assemble/solve so the profile attributes the momentumPredictor cost.
            Kokkos::Profiling::pushRegion("momentum.construct");
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::div(phi, U) - dsl::imp::laplacian(turb->nuEff(), U)
                    + dsl::exp::viscousStress(nu, turb->nut(), turb->gradU()),
                U,
                rt
            );
            Kokkos::Profiling::popRegion(); // momentum.construct

            if (simple.momentumPredictor())
            {
                UEqn.solve(-1.0 * dsl::exp::grad(p));
            }
            else
            {
                UEqn.assemble();
            }
            Kokkos::Profiling::popRegion(); // end neoSimpleFoam.momentumPredictor
            logGpuMem("after-UEqn");

            // SIMPLE / SIMPLEC pressure-velocity coupling (single pass, no inner PISO loop)
            {
                Kokkos::Profiling::ScopedRegion pressureRegion("neoSimpleFoam.pressureCorrector");
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                // SIMPLEC ("consistent" in fvSolution.SIMPLE): use the consistent diagonal
                // rAtU = 1/(1/rAU - H1) which folds the neighbour coupling that plain SIMPLE
                // drops. crAtU aliases crAU for plain SIMPLE so the rest of the block is shared.
                const bool consistent = simple.consistent();
                nnfvcc::VolumeField<NeoN::scalar> crAtU =
                    consistent ? nf::computeRAtU(UEqn, crAU) : crAU;

                // Pressure Laplacian coefficient: rAtU (== rAU for plain SIMPLE) on the faces.
                nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAtU);
                rAU.name = "rAUf";

                // No ddtFluxCorr: SIMPLE is steady-state
                auto phiHbyA = nf::flux(hByA);

                if (consistent)
                {
                    // phiHbyA += interpolate(rAtU - rAU)*snGrad(p)*magSf
                    nf::addConsistentFluxCorrection(phiHbyA, crAU, crAtU, p);
                    // HbyA -= (rAU - rAtU)*grad(p)   (used by the velocity corrector below)
                    nf::subtractConsistentHbyA(hByA, crAU, crAtU, p);
                }

                // TODO additionally missing
                // Foam::adjustPhi(phiHbyA, U, p);
                // Foam::constrainPressure(p, U, phiHbyA, rAU);

                // Pre-solve snapshots: internal vector for applyFieldRelaxation
                // and a full VolumeField for the deferred-correction recovery
                // below (the Laplacian uses pre-solve p in its deferred
                // correction; recovering it post-solve must use the same p).
                auto pPrev = NeoN::dsl::fieldRelaxationSnapshot(p);
                nnfvcc::VolumeField<NeoN::scalar> pSnapshot(p);

                // Non-orthogonal pressure corrector loop
                while (simple.correctNonOrthogonal())
                {
                    Kokkos::Profiling::ScopedRegion pEqnRegion(
                        "neoSimpleFoam.pressureCorrector.pEqn"
                    );
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    // Keep the non-orthogonal faceFluxCorrection so updateFaceVelocity can subtract
                    // it from phi (the deferred correction is otherwise in the RHS only, leaving
                    // phi inconsistent — continuity leaks on non-orthogonal meshes with 'corrected'
                    // snGrad). neoIcoFoam/neoPisoFoam already do this; neoSimpleFoam previously did
                    // not and masked it with an (incorrect) explicit add-back.
                    pEqn.linearSystem().keepFaceFluxCorrection(true);

                    if (ofP.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    pEqn.solve();
                    p.correctBoundaryConditions();

                    if (simple.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);


                        // The non-orthogonal correction is owned by updateFaceVelocity, which
                        // subtracts the Laplacian's stored faceFluxCorrection (ffc) from phi. The
                        // previous explicit add-back here was redundant (double-counting on the
                        // 'corrected' scheme) and, because it hardcoded a 'corrected' snGrad, it
                        // also injected a spurious correction on non-orthogonal meshes even under
                        // an 'uncorrected' laplacian — corrupting continuity. Removed.
                    }
                }
                nf::reportContinuityError(phi, rt, cumulativeContErr);


                // Explicit pressure field under-relaxation — mirrors p.relax()
                // in OpenFOAM's simpleFoam. Blends p with the pre-solve
                // snapshot in place: p = pPrev + alpha*(p - pPrev). alpha=1
                // is a bitwise no-op; reading the factor from
                // relaxationFactors.fields.<name> in fvSolution.
                NeoN::dsl::applyFieldRelaxation(
                    p,
                    pPrev,
                    readFieldRelaxationFactor(rt.fvSolutionDict, "p")
                );
                p.correctBoundaryConditions();

                // Momentum corrector: U = HbyA - rAtU*grad(p) (rAtU == rAU for plain SIMPLE).
                // With the SIMPLEC HbyA correction above this reproduces the plain-SIMPLE
                // reconstruction U = HbyA0 - rAU*grad(p) once p has converged.
                nf::updateVelocity(hByA, crAtU, p, U);
                U.correctBoundaryConditions();
            }
            logGpuMem("after-pEqn");

            {
                Kokkos::Profiling::ScopedRegion turbRegion("neoSimpleFoam.turbulenceCorrect");
                turb->correct(U, phi, rt);
            }
            logGpuMem("after-turb");

            {
                Kokkos::Profiling::ScopedRegion writeRegion("neoSimpleFoam.write");
                runTime.write();
                if (runTime.outputTime())
                {
                    NeoN::Logging::info("Writing p");
                    write(p, mesh);
                    NeoN::Logging::info("Writing U");
                    write(U, mesh);
                    NeoN::Logging::info("Writing turbulence variables");
                    turb->write(mesh);
                }
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
