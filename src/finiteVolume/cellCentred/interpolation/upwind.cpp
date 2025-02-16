// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @brief computional kernel to perform an upwind interpolation
** from a source volumeField to a surface field. It performs an interpolation
** of the form
**
** d_f = w_f * s_O + ( 1 - w_f ) * s_N where w_1 is either 0,1 depending on the
** direction of the flux
**
**@param src the input field
**@param weights weights for the interpolation
**@param dst the target field
*/
template<typename ValueType>
void computeUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalField().span();
    const auto [srcS, weightS, ownerS, neighS, boundS, fluxS] = spans(
        src.internalField(),
        weights.internalField(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryField().value(),
        flux.internalField()
    );
    size_t nInternalFaces = dst.mesh().nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            if (facei < nInternalFaces)
            {
                if (fluxS[facei] >= 0)
                {
                    size_t own = static_cast<size_t>(ownerS[facei]);
                    dstS[facei] = srcS[own];
                }
                else
                {
                    size_t nei = static_cast<size_t>(neighS[facei]);
                    dstS[facei] = srcS[nei];
                }
            }
            else
            {
                dstS[facei] = weightS[facei] * boundS[facei - nInternalFaces];
            }
        }
    );
}

Upwind::Upwind(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
    : SurfaceInterpolationFactory::Register<Upwind>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

void Upwind::interpolate(
    [[maybe_unused]] const VolumeField<scalar>& src, [[maybe_unused]] SurfaceField<scalar>& dst
) const
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const SurfaceField<scalar>& flux, const VolumeField<scalar>& src, SurfaceField<scalar>& dst
) const
{
    computeUpwindInterpolation(src, flux, geometryScheme_->weights(), dst);
}

void Upwind::interpolate(
    [[maybe_unused]] const VolumeField<Vector>& src, [[maybe_unused]] SurfaceField<Vector>& dst
) const
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const SurfaceField<scalar>& flux, const VolumeField<Vector>& src, SurfaceField<Vector>& dst
) const
{
    computeUpwindInterpolation(src, flux, geometryScheme_->weights(), dst);
}

std::unique_ptr<SurfaceInterpolationFactory> Upwind::clone() const
{
    return std::make_unique<Upwind>(*this);
}

} // namespace NeoFOAM
