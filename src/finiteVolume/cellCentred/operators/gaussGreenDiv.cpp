// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ExecutorType>
struct SumInternalFluxesKernel
{
    const ExecutorType& exec;
    const std::pair<size_t, size_t> range;
    const std::span<const scalar> surfFaceFlux;
    const std::span<const scalar> surfPhif;
    mutable std::span<scalar> surfDivPhi;
    const std::span<const int> surfOwner;
    const std::span<const int> surfNeighbour;
    const std::string name;

    KOKKOS_INLINE_FUNCTION
    void call(const size_t i) const
    {
        scalar flux = surfFaceFlux[i] * surfPhif[i];
        Kokkos::atomic_add(&surfDivPhi[static_cast<size_t>(surfOwner[i])], flux);
        Kokkos::atomic_sub(&surfDivPhi[static_cast<size_t>(surfNeighbour[i])], flux);
    }
};

template<>
KOKKOS_INLINE_FUNCTION void SumInternalFluxesKernel<SerialExecutor>::call(const size_t i) const
{
    scalar flux = surfFaceFlux[i] * surfPhif[i];
    surfDivPhi[static_cast<size_t>(surfOwner[i])] += flux;
    surfDivPhi[static_cast<size_t>(surfNeighbour[i])] -= flux;
}


void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    Field<scalar>& divPhi,
    SurfaceField<scalar>& phif
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    surfInterp.interpolate(faceFlux, phi, phif);

    auto surfDivPhi = divPhi.span();

    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto surfFaceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();


    std::visit(
        [&](const auto& e)
        {
            parallelFor2(
                e,
                SumInternalFluxesKernel {
                    e,
                    {0, nInternalFaces},
                    surfFaceFlux,
                    surfPhif,
                    surfDivPhi,
                    surfOwner,
                    surfNeighbour,
                    "sumFluxesInternal"
                }
            );
        },
        exec
    );


    parallelFor(
        exec,
        {nInternalFaces, surfPhif.size()},
        KOKKOS_LAMBDA(const size_t i) {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
            Kokkos::atomic_add(&surfDivPhi[own], valueOwn);
        },
        "sumFluxesBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli) { surfDivPhi[celli] *= 1 / surfV[celli]; },
        "normalizeFluxes"
    );
}

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

    computeDiv(faceFlux, phi, surfInterp, divPhi, phif);
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
      surfaceInterpolation_(exec, mesh, inputs) {};

void GaussGreenDiv::div(
    VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
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
