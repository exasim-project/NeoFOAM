// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


/* @brief free standing function implementation of the divergence operator
** ie computes \sum_f S_f \cdot \phi_f
**
** @param faceFlux
*/
template<typename ValueType>
void computeDiv(
    const Executor& exec,
    size_t nInternalFaces,
    size_t nBoundaryFaces,
    std::span<const int> neighbour,
    std::span<const int> owner,
    std::span<const int> faceCells,
    std::span<const scalar> faceFlux,
    std::span<const ValueType> phiF,
    std::span<const scalar> V,
    std::span<ValueType> divPhi
)
{
    size_t nCells {V.size()};
    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < faceFlux.size(); i++)
        {
            ValueType flux = faceFlux[i] * phiF[i];
            divPhi[static_cast<size_t>(owner[i])] += flux;
            divPhi[static_cast<size_t>(neighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < nInternalFaces + nBoundaryFaces; i++)
        {
            auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
            ValueType valueOwn = faceFlux[i] * phiF[i];
            divPhi[own] += valueOwn;
        }

        for (size_t celli = 0; celli < nCells; celli++)
        {
            divPhi[celli] *= 1 / V[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                ValueType flux = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&divPhi[static_cast<size_t>(owner[i])], flux);
                Kokkos::atomic_sub(&divPhi[static_cast<size_t>(neighbour[i])], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, nBoundaryFaces},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
                ValueType valueOwn = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&divPhi[own], valueOwn);
            }
        );

        parallelFor(
            exec, {0, nCells}, KOKKOS_LAMBDA(const size_t celli) { divPhi[celli] *= 1 / V[celli]; }
        );
    }
}


template<typename ValueType>
void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation& surfInterp,
    Field<ValueType>& divPhi
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<ValueType> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    fill(phif.internalField(), NeoFOAM::zero<ValueType>::value);
    surfInterp.interpolate(faceFlux, phi, phif);

    size_t nInternalFaces = mesh.nInternalFaces();
    size_t nBoundaryFaces = mesh.nBoundaryFaces();
    computeDiv<ValueType>(
        exec,
        nInternalFaces,
        nBoundaryFaces,
        mesh.faceNeighbour().span(),
        mesh.faceOwner().span(),
        mesh.boundaryMesh().faceCells().span(),
        faceFlux.internalField().span(),
        phif.internalField().span(),
        mesh.cellVolumes().span(),
        divPhi.span()
    );
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
    computeDiv<scalar>(faceFlux, phi, surfaceInterpolation_, divPhi.internalField());
};

void GaussGreenDiv::div(
    Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv<scalar>(faceFlux, phi, surfaceInterpolation_, divPhi);
};

VolumeField<scalar>
GaussGreenDiv::div(const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi)
{
    std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
    VolumeField<scalar> divPhi(
        exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<scalar>>(mesh_)
    );
    computeDiv<scalar>(faceFlux, phi, surfaceInterpolation_, divPhi.internalField());
    return divPhi;
};


void GaussGreenDiv::div(
    VolumeField<Vector>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<Vector>& phi
)
{
    computeDiv<Vector>(faceFlux, phi, surfaceInterpolation_, divPhi.internalField());
};

void GaussGreenDiv::div(
    Field<Vector>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<Vector>& phi
)
{
    computeDiv<Vector>(faceFlux, phi, surfaceInterpolation_, divPhi);
};

VolumeField<Vector>
GaussGreenDiv::div(const SurfaceField<scalar>& faceFlux, VolumeField<Vector>& phi)
{
    std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
    VolumeField<Vector> divPhi(
        exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<Vector>>(mesh_)
    );
    computeDiv<Vector>(faceFlux, phi, surfaceInterpolation_, divPhi.internalField());
    return divPhi;
};

std::unique_ptr<DivOperatorFactory> GaussGreenDiv::clone() const
{
    return std::make_unique<GaussGreenDiv>(*this);
}


};
