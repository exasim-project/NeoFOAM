// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    Field<scalar>& divPhi
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );
    fill(phif.internalField(), 0.0);
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    surfInterp.interpolate(faceFlux, phi, phif);

    auto surfDivPhi = divPhi.span();

    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto surfFaceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();

    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[static_cast<size_t>(surfOwner[i])] += flux;
            surfDivPhi[static_cast<size_t>(surfNeighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < surfPhif.size(); i++)
        {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[own] += valueOwn;
        }

        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            surfDivPhi[celli] *= 1 / surfV[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[static_cast<size_t>(surfOwner[i])], flux);
                Kokkos::atomic_sub(&surfDivPhi[static_cast<size_t>(surfNeighbour[i])], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, surfPhif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[own], valueOwn);
            }
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { surfDivPhi[celli] *= 1 / surfV[celli]; }
        );
    }
}

void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<scalar>& divPhi
)
{
    Field<scalar>& divPhiField = divPhi.internalField();
    computeDiv(faceFlux, phi, surfInterp, divPhiField);
}

GaussGreenDiv::GaussGreenDiv(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs
)
    : DivOperatorFactory::Register<GaussGreenDiv>(exec, mesh),
      surfaceInterpolation_(exec, mesh, inputs),
      sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};


la::LinearSystem<scalar, localIdx> GaussGreenDiv::createEmptyLinearSystem() const
{
    return sparsityPattern_->linearSystem();
};

void GaussGreenDiv::div(
    VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};

void GaussGreenDiv::div(
    la::LinearSystem<scalar, localIdx>& ls,
    const SurfaceField<scalar>& faceFlux,
    VolumeField<scalar>& phi
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const std::size_t nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto [sFaceFlux, owner, neighbour, diagOffs, ownOffs, neiOffs] = spans(
        faceFlux.internalField(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        sparsityPattern_->diagOffset(),
        sparsityPattern_->ownerOffset(),
        sparsityPattern_->neighbourOffset()
    );
    const auto rowPtrs = ls.matrix().rowPtrs();
    const auto colIdxs = ls.matrix().colIdxs();
    const auto values = ls.matrix().values();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            scalar flux = sFaceFlux[facei];
            // scalar weight = 0.5;
            scalar weight = flux >= 0 ? 1 : 0;
            scalar value = 0;
            std::size_t own = static_cast<std::size_t>(owner[facei]);
            std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

            // add neighbour contribution upper
            std::size_t rowNeiStart = rowPtrs[nei];
            std::size_t rowOwnStart = rowPtrs[own];

            value = -weight * flux;
            // scalar valueNei = (1 - weight) * flux;
            values[rowNeiStart + neiOffs[facei]] += value;
            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], value);

            // upper triangular part

            // add owner contribution lower
            value = flux * (1 - weight);
            values[rowOwnStart + ownOffs[facei]] += value;
            Kokkos::atomic_sub(&values[rowNeiStart + diagOffs[nei]], value);
        }
    );
};

void GaussGreenDiv::div(
    Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};

VolumeField<scalar>
GaussGreenDiv::div(const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi)
{
    std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
    VolumeField<scalar> divPhi(
        exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<scalar>>(mesh_)
    );
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
    return divPhi;
};

std::unique_ptr<DivOperatorFactory> GaussGreenDiv::clone() const
{
    return std::make_unique<GaussGreenDiv>(*this);
}


};
