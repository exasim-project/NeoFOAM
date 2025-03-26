// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const SurfaceField<scalar>& gamma,
    VolumeField<ValueType>& phi,
    Field<ValueType>& lapPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    SurfaceField<ValueType> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    auto [refGradient, value, valueFraction, refValue] = spans(
        phi.boundaryField().refGrad(),
        phi.boundaryField().value(),
        phi.boundaryField().valueFraction(),
        phi.boundaryField().refValue()
    );

    const auto [result, faceArea, fnGrad, vol] =
        spans(lapPhi, mesh.magFaceAreas(), faceNormalGrad.internalField(), mesh.cellVolumes());


    size_t nInternalFaces = mesh.nInternalFaces();

    // TODO use NeoFOAM::add and sub
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            ValueType flux = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(&result[static_cast<size_t>(owner[i])], flux);
            Kokkos::atomic_sub(&result[static_cast<size_t>(neighbour[i])], flux);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, fnGrad.size()},
        KOKKOS_LAMBDA(const size_t i) {
            size_t own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            ValueType valueOwn = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(&result[own], valueOwn);
        }
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli) { result[celli] *= operatorScaling[celli] / vol[celli]; }
    );
}

#define NF_DECLARE_COMPUTE_EXP_LAP(TYPENAME)                                                       \
    template void computeLaplacianExp<TYPENAME>(                                                   \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const SurfaceField<scalar>&,                                                               \
        VolumeField<TYPENAME>&,                                                                    \
        Field<TYPENAME>&,                                                                          \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_LAP(scalar);
NF_DECLARE_COMPUTE_EXP_LAP(Vector);


template<typename ValueType>
void computeLaplacianImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& gamma,
    VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const SparsityPattern& sparsityPattern,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const std::size_t nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto [owner, neighbour, surfFaceCells, diagOffs, ownOffs, neiOffs] = spans(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        sparsityPattern.diagOffset(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset()
    );

    const auto [sGamma, deltaCoeffs, magFaceArea] = spans(
        gamma.internalField(), faceNormalGradient.deltaCoeffs().internalField(), mesh.magFaceAreas()
    );

    auto [refGradient, value, valueFraction, refValue] = spans(
        phi.boundaryField().refGrad(),
        phi.boundaryField().value(),
        phi.boundaryField().valueFraction(),
        phi.boundaryField().refValue()
    );

    // FIXME: what if order changes
    auto [values, colIdxs, rowPtrs] = ls.matrix().view();

    // const auto rowPtrs = ls.matrix().rowPtrs();
    // const auto colIdxs = ls.matrix().colIdxs();
    // auto values = ls.matrix().values().span();
    auto rhs = ls.rhs().span();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            scalar flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei];

            std::size_t own = static_cast<std::size_t>(owner[facei]);
            std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

            // add neighbour contribution upper
            std::size_t rowNeiStart = rowPtrs[nei];
            std::size_t rowOwnStart = rowPtrs[own];

            // scalar valueNei = (1 - weight) * flux;
            values[rowNeiStart + neiOffs[facei]] += flux * one<ValueType>();
            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], flux * one<ValueType>());

            // upper triangular part

            // add owner contribution lower
            values[rowOwnStart + ownOffs[facei]] += flux * one<ValueType>();
            Kokkos::atomic_sub(&values[rowNeiStart + diagOffs[nei]], flux * one<ValueType>());
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, sGamma.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            std::size_t bcfacei = facei - nInternalFaces;
            scalar flux = sGamma[facei] * magFaceArea[facei];

            std::size_t own = static_cast<std::size_t>(surfFaceCells[bcfacei]);
            std::size_t rowOwnStart = rowPtrs[own];

            values[rowOwnStart + diagOffs[own]] -=
                flux * valueFraction[bcfacei] * deltaCoeffs[facei] * one<ValueType>();
            rhs[own] -=
                (flux
                 * (valueFraction[bcfacei] * deltaCoeffs[facei] * refValue[bcfacei]
                    + (1.0 - valueFraction[bcfacei]) * refGradient[bcfacei]));
        }
    );

    parallelFor(
        exec,
        {0, rhs.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            rhs[celli] *= operatorScaling[celli];
            for (size_t i = rowPtrs[celli]; i < rowPtrs[celli + 1]; i++)
            {
                values[i] *= operatorScaling[celli];
            }
        }
    );
}

#define NF_DECLARE_COMPUTE_IMP_LAP(TYPENAME)                                                       \
    template void computeLaplacianImpl<                                                            \
        TYPENAME>(la::LinearSystem<TYPENAME, localIdx>&, const SurfaceField<scalar>&, VolumeField<TYPENAME>&, const dsl::Coeff, const SparsityPattern&, const FaceNormalGradient<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LAP(scalar);
NF_DECLARE_COMPUTE_IMP_LAP(Vector);


// instantiate the template class
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vector>;

};
