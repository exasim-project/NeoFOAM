// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <limits>

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

    scalar maxCoNum = std::numeric_limits<scalar>::lowest();
    scalar meanCoNum = 0.0;
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            volPhi[static_cast<size_t>(surfOwner[i])] += flux;
            volPhi[static_cast<size_t>(surfNeighbour[i])] += flux;
        }

        for (size_t i = nInternalFaces; i < faceFlux.size(); i++)
        {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar flux = surfFaceFlux[i] * surfFaceFlux[i];
            volPhi[own] += flux;
        }

	phi.correctBoundaryConditions();

        scalar totalPhi = 0.0;
        scalar totalVol = 0.0;
        for (size_t i = 0; i < mesh.nCells(); i++)
	{
            double val = volPhi[i] / surfV[i];
            if( val > maxCoNum ) maxCoNum = val;
            totalPhi += volPhi[i];
            totalVol += surfV[i];
	}

        maxCoNum *= 0.5 * dt;
        meanCoNum = 0.5 * (totalPhi / totalVol) * dt;
    }
    else
    {
	// FIXME: If executor GPU the Courant numbers are off with origin in this parallelFor.
	//        If executor CPU the values are correct.
	//        If executor Serial (in if branch above) the values are correct.
	// TODO:  Unit testing...
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
                Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfOwner[i])], flux);
                Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfNeighbour[i])], flux);
                printf("nei own flux %d %d %ld %.10e \n", surfNeighbour[i], surfOwner[i], i, flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, faceFlux.size()},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
                Kokkos::atomic_add(&volPhi[own], flux);
                printf("own flux %ld %ld %.10e \n", own, i, flux);
            }
        );

	phi.correctBoundaryConditions();

	scalar maxValue;
	Kokkos::Max<NeoFOAM::scalar> maxReducer(maxValue);
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, NeoFOAM::scalar& lmax) {
	        NeoFOAM::scalar val = (volPhi[celli] / surfV[celli]);
                if( val > lmax ) lmax = val;
                printf("volPhi %ld %.10e \n", celli, volPhi[celli]);
                //printf("surfV %ld %.10e \n", celli, surfV[celli]);
            },
            maxReducer
        );

	// TODO: Is calculation of the mean really necessary?
	//       It is only printed and maybe needed for the user.
        scalar totalPhi = 0.0;
	Kokkos::Sum<NeoFOAM::scalar> sumPhi(totalPhi);
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, double& lsum) { lsum += volPhi[celli]; },
            sumPhi
        );

        scalar totalVol = 0.0;
	Kokkos::Sum<NeoFOAM::scalar> sumVol(totalVol);
        parallelReduce(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli, double& lsum) { lsum += surfV[celli]; },
            sumVol
        );

        maxCoNum  = maxReducer.reference() * 0.5 * dt;
        meanCoNum = 0.5 * (sumPhi.reference() / sumVol.reference()) * dt;
    }

    std::cout << "Courant Number mean: " << meanCoNum << " max: " << maxCoNum  << std::endl;

    return maxCoNum;
}

};
