// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "FoamAdapter/FoamAdapter.hpp"

#include "fvCFD.H"
#include "pisoControl.H"

using Foam::Info;
using Foam::endl;
using Foam::nl;

namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = FoamAdapter;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"

        auto rt = nf::createAdapterRunTime(runTime);
        auto& mesh = rt.mesh;

        Foam::pisoControl piso(mesh);

#include "createFields.H"

        // TODO have central place for the mapper
        auto& solverDict = rt.fvSolutionDict.get<NeoN::Dictionary>("solvers");
        solverDict.get<NeoN::Dictionary>("p") =
            nf::mapFvSolution(solverDict.get<NeoN::Dictionary>("p"));
        solverDict.get<NeoN::Dictionary>("U") =
            nf::mapFvSolution(solverDict.get<NeoN::Dictionary>("U"));

        Info << "creating nf pressure field" << endl;
        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        fvcc::VolumeField<NeoN::scalar>& p =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofp,
                    .name = "p"
                }
            );

        Info << "creating nf velocity field" << endl;
        fvcc::VolumeField<NeoN::Vec3>& U =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::Vec3>>(
                nf::CreateFromFoamField<Foam::volVectorField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofU,
                    .name = "U"
                }
            );

        Info << "creating nf nu field" << endl;
        auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
        fvcc::SurfaceField<NeoN::scalar> nu(rt.exec, "nu", rt.nfMesh, nuBCs);
        NeoN::fill(nu.internalVector(), viscosity.value());
        NeoN::fill(nu.boundaryData().value(), viscosity.value());

        Info << "creating nf phi field" << endl;
        auto phi = nf::constructSurfaceField(rt.exec, rt.nfMesh, ofphi);
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        Info << "\nStarting time loop\n" << endl;
        while (runTime.loop())
        {
            Info << "Time = " << runTime.timeName() << nl << endl;

            auto& oldU = fvcc::oldTime(U);
            oldU.internalVector() = U.internalVector();

            auto coNum = fvcc::computeCoNum(phi, rt.dt);
            if (rt.adjustTimeStep)
            {
                nf::setDeltaT(runTime, rt, coNum);
            }

            // Momentum predictor
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U),
                U,
                rt
            );

            if (piso.momentumPredictor())
            {
                UEqn.solve();
            }
            else
            {
                // NOTE since computing rAU and HbyA requires an assembled system matrix we
                // explicitly trigger assembly here.
                UEqn.assemble();
            }

            // --- PISO loop
            while (piso.correct())
            {
                Info << "PISO loop" << endl;
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                nnfvcc::SurfaceField<NeoN::scalar> rAU =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(crAU);
                rAU.name = "rAUf";

                auto phiHbyA = nf::flux(hByA);
                // TODO: OpenFOAM typically also corrects phiHbyA with
                // + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi);
                // for the first term we can use but fvc::ddtCorr is missing
                // NeoN::Input input = NeoN::TokenList({"linear"});
                // fvcc::SurfaceInterpolation<NeoN::scalar> surfInterpolation(rt.exec, rt.nfMesh,
                // input); auto surfRAU = surfInterpolation.interpolate(rAU);

                // TODO additionally missing
                // Foam::adjustPhi(phiHbyA, U, p);
                // Update the pressure BCs to ensure flux consistency
                // Foam::constrainPressure(p, U, phiHbyA, rAU);

                // Non-orthogonal pressure corrector loop
                while (piso.correctNonOrthogonal())
                {
                    // Pressure corrector
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    if (ofp.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    auto stats = pEqn.solve();
                    p.correctBoundaryConditions();

                    if (piso.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                    }
                }
                // TODO: missing
                // #include "continuityErrs.H"

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
            }

            runTime.write();
            if (runTime.outputTime())
            {
                Info << "writing p field" << endl;
                write(p.internalVector(), mesh, "p");
                Info << "writing U field" << endl;
                write(U.internalVector(), mesh, "U");
            }

            runTime.printExecutionTime(Info);
        }

        Info << "End\n" << endl;
    }
    Kokkos::finalize();

    return 0;
}

// ************************************************************************* //
