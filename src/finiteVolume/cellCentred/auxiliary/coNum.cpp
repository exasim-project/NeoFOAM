// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <limits>

#include "NeoFOAM/core/info.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/auxiliary/coNum.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

scalar computeCoNum(const SurfaceField<scalar>& faceFlux, const scalar dt)
{
    const UnstructuredMesh& mesh = faceFlux.mesh();
    const auto exec = faceFlux.exec();
    VolumeField<scalar> phi(exec, "phi", mesh, createCalculatedBCs<VolumeBoundary<scalar>>(mesh));
    fill(phi.internalField(), 0.0);

    const auto [surfFaceCells, volPhi, surfOwner, surfNeighbour, surfFaceFlux, surfV] = spans(
        mesh.boundaryMesh().faceCells(),
        phi.internalField(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        faceFlux.internalField(),
        mesh.cellVolumes()
    );
    size_t nInternalFaces = mesh.nInternalFaces();

    scalar maxCoNum = std::numeric_limits<scalar>::lowest();
    scalar meanCoNum = 0.0;
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfOwner[i])], flux);
            Kokkos::atomic_add(&volPhi[static_cast<size_t>(surfNeighbour[i])], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, faceFlux.size()},
        KOKKOS_LAMBDA(const size_t i) {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            Kokkos::atomic_add(&volPhi[own], flux);
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
            if (val > lmax) lmax = val;
        },
        maxReducer
    );

    scalar totalPhi = 0.0;
    Kokkos::Sum<NeoFOAM::scalar> sumPhi(totalPhi);
    parallelReduce(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli, scalar& lsum) { lsum += volPhi[celli]; },
        sumPhi
    );

    scalar totalVol = 0.0;
    Kokkos::Sum<NeoFOAM::scalar> sumVol(totalVol);
    parallelReduce(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli, scalar& lsum) { lsum += surfV[celli]; },
        sumVol
    );

    maxCoNum = maxReducer.reference() * 0.5 * dt;
    meanCoNum = 0.5 * (sumPhi.reference() / sumVol.reference()) * dt;
    NF_INFO(
        "Courant Number mean: " + std::to_string(meanCoNum) + " max: " + std::to_string(maxCoNum)
    );

    return maxCoNum;
}

};
