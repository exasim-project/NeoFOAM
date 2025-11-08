// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"


#include "fvCFD.H"
#include "pisoControl.H"

using Foam::Info;
using Foam::endl;
using Foam::nl;
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

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

        Info << "creating NeoFOAM fields" << endl;
        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        fvcc::VolumeField<NeoN::scalar>& nfT =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = T,
                    .name = "nfT"
                }
            );
        auto& nfTOld = oldTime(nfT);
        nfTOld.internalVector() = nfT.internalVector();
        nfT.correctBoundaryConditions();

        auto nfKappa = nf::constructSurfaceField(rt.exec, rt.nfMesh, kappa);

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        Info << "\nStarting time loop\n" << endl;
        while (runTime.loop())
        {
            Info << "Time = " << runTime.timeName() << nl << endl;

            Foam::scalar t = runTime.time().value();
            Foam::scalar dt = runTime.deltaT().value();

            dsl::Expression nfTEqn(dsl::imp::ddt(nfT) - dsl::imp::laplacian(nfKappa, nfT));

            dsl::solve(
                nfTEqn,
                nfT,
                t,
                dt,
                rt.fvSchemesDict,
                rt.fvSolutionDict.get<NeoN::Dictionary>("solvers").get<NeoN::Dictionary>("nfT")
            );

            runTime.write();
            if (runTime.outputTime())
            {
                Info << "writing nfT field" << endl;
                write(nfT.internalVector(), rt.mesh, "nfT");
            }

            runTime.printExecutionTime(Info);
        }

        Foam::Info << "End\n" << Foam::endl;
    }
    Kokkos::finalize();

    return 0;
}

// ************************************************************************* //
