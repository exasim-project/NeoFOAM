// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

struct SurfaceFluxView
{
    const std::span<const scalar> flux;
    const std::span<const scalar> phif;
    const std::span<const int> owner;
    const std::span<const int> neigh;
};


template<typename ExecutorType>
struct SumInternalFluxesKernel
{
    const ExecutorType& exec;
    const std::pair<size_t, size_t> range;
    const SurfaceFluxView surf;
    const std::string name;

    mutable std::span<scalar> res; //!< span for result

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        scalar flux = surf.flux[i] * surf.phif[i];
        Kokkos::atomic_add(&res[static_cast<size_t>(surf.owner[i])], flux);
        Kokkos::atomic_sub(&res[static_cast<size_t>(surf.neigh[i])], flux);
    }
};

/* specialisation for serial executor without atomics */
template<>
KOKKOS_INLINE_FUNCTION void SumInternalFluxesKernel<SerialExecutor>::operator()(const size_t i
) const
{
    scalar flux = surf.flux[i] * surf.phif[i];
    res[static_cast<size_t>(surf.owner[i])] += flux;
    res[static_cast<size_t>(surf.neigh[i])] -= flux;
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


    const SurfaceFluxView surfView {
        faceFlux.internalField().span(),
        phif.internalField().span(),
        mesh.faceOwner().span(),
        mesh.faceNeighbour().span()
    };

    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();

    std::visit(
        [&](const auto& e)
        {
            AtomicAdd add {e};
            AtomicSub sub {e};
            parallelFor(
                e,
                {0, nInternalFaces},
                KOKKOS_LAMBDA(const size_t i) {
                    scalar flux = surfView.flux[i] * surfView.phif[i];
                    add(surfDivPhi[static_cast<size_t>(surfView.owner[i])], flux);
                    sub(surfDivPhi[static_cast<size_t>(surfView.neigh[i])], flux);
                },
                {0, nInternalFaces},
                KOKKOS_LAMBDA(const size_t i) {
                    auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                    scalar valueOwn = surfView.flux[i] * surfView.phif[i];
                    add(surfDivPhi[own], valueOwn);
                },
                "sumFluxes"
            );
        },
        exec
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
    NF_PING();
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
    NF_PING();
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
    NF_PING();
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};

void GaussGreenDiv::div(
    Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    NF_PING();
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
