// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

#include <Kokkos_Core.hpp>

#include <functional>

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


template<typename ValueType>
class Upwind : public SurfaceInterpolationFactory<ValueType>::template Register<Upwind<ValueType>>
{

    using Base = SurfaceInterpolationFactory<ValueType>::template Register<Upwind<ValueType>>;

public:

    Upwind(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "upwind"; }

    static std::string doc() { return "upwind interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const override
    {
        NF_ERROR_EXIT("limited scheme require a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override
    {
        computeUpwindInterpolation(src, flux, geometryScheme_->weights(), dst);
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<Upwind>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
