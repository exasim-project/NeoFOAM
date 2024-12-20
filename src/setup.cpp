// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/setup.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

scalar computeCoNum(
    const SurfaceField<scalar>& faceFlux,
    const scalar dt
)
{
    const UnstructuredMesh& mesh = faceFlux.mesh();
    const auto exec = faceFlux.exec();
    VolumeField<scalar> phi(
        exec, "phi", mesh, createCalculatedBCs<VolumeBoundary<scalar>>(mesh)
    );
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();

    const auto volPhi = phi.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto surfFaceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();

    scalar maxCoNum = 0.0;
    scalar meanCoNum = 0.0;
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            volPhi[static_cast<size_t>(surfOwner[i])] += flux;
            volPhi[static_cast<size_t>(surfNeighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < volPhi.size(); i++)
        {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar flux = surfFaceFlux[i] * surfFaceFlux[i];
            volPhi[own] += flux;
        }

        // TODO: Correct boundary conditions
	//       In simple scalarAdvection case BC are trivial for now.

	// FIXME: Implement for maxCoNum and meanCoNum calculations.
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
                Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfOwner[i])], flux);
                Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfNeighbour[i])], flux);
            },
            "sumFluxesInternal"
        );

        parallelFor(
            exec,
            {nInternalFaces, faceFlux.size()},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
                Kokkos::atomic_add(&volPhi[own], flux);
            },
            "sumFluxesBoundary"
        );

        // TODO: Correct boundary conditions.
	//       In simple scalarAdvection case BC are trivial for now.

	// FIXME: Reduction is not working properly. Find the bug.
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, double& lmax) {
                double val = (volPhi[celli] / surfV[celli]);
                if( val > lmax ) lmax = val;
                //printf("max %ld %.10e \n", celli, lmax);
            },
            maxCoNum
        );
        //std::cout << "maxCoNum " << maxCoNum << std::endl;

	// TODO: Is calculation of the mean really necessary?
	//       It is only printed and maybe needed for the user?

	// FIXME: Reduction is not working properly. Find the bug.
        scalar sumPhi = 0.0;
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, double& lsum) { lsum += volPhi[celli]; },
            sumPhi
        );

	// FIXME: Reduction is not working properly. Find the bug.
        scalar sumV = 0.0;
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, double& lsum) { lsum += surfV[celli]; },
            sumV
        );

        maxCoNum *= 0.5 * dt;
        meanCoNum = 0.5 * (sumPhi / sumV) * dt;
    }

    std::cout << "Courant Number mean: " << meanCoNum << " max: " << maxCoNum  << " " << dt << std::endl;

    return maxCoNum;
}

};
