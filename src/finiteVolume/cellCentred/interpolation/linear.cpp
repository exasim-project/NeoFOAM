// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @brief computional kernel to perform a linear interpolation
** from a source volumeField to a surface field. It performs an interpolation
** of the form
**
** d_f = w_f * s_O + ( 1 - w_f ) * s_N
**
**@param src the input field
**@param weights weights for the interpolation
**@param dst the target field
*/
template<typename ValueType>
void computeLinearInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalField().span();
    const auto [srcS, weightS, ownerS, neighS, boundS] = spans(
        src.internalField(),
        weights.internalField(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryField().value()
    );
    size_t nInternalFaces = dst.mesh().nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t own = static_cast<size_t>(ownerS[facei]);
            size_t nei = static_cast<size_t>(neighS[facei]);
            if (facei < nInternalFaces)
            {
                dstS[facei] = weightS[facei] * srcS[own] + (1 - weightS[facei]) * srcS[nei];
            }
            else
            {
                dstS[facei] = weightS[facei] * boundS[facei - nInternalFaces];
            }
        }
    );
}

Linear::Linear(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
    : SurfaceInterpolationFactory::Register<Linear>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

Linear::Linear(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationFactory::Register<Linear>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

void Linear::interpolate(const VolumeField<scalar>& src, SurfaceField<scalar>& dst) const
{
    computeLinearInterpolation(src, geometryScheme_->weights(), dst);
}

void Linear::interpolate(const VolumeField<Vector>& src, SurfaceField<Vector>& dst) const
{
    computeLinearInterpolation(src, geometryScheme_->weights(), dst);
}

void Linear::interpolate(
    [[maybe_unused]] const SurfaceField<scalar>& flux,
    const VolumeField<Vector>& src,
    SurfaceField<Vector>& dst
) const
{
    interpolate(src, dst);
}

void Linear::interpolate(
    [[maybe_unused]] const SurfaceField<scalar>& flux,
    const VolumeField<scalar>& src,
    SurfaceField<scalar>& dst
) const
{
    interpolate(src, dst);
}


std::unique_ptr<SurfaceInterpolationFactory> Linear::clone() const
{
    return std::make_unique<Linear>(*this);
}

} // namespace NeoFOAM
