// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


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
            if (facei < nInternalFaces)
            {
                size_t own = static_cast<size_t>(ownerS[facei]);
                size_t nei = static_cast<size_t>(neighS[facei]);
                dstS[facei] = weightS[facei] * srcS[own] + (1 - weightS[facei]) * srcS[nei];
            }
            else
            {
                dstS[facei] = weightS[facei] * boundS[facei - nInternalFaces];
            }
        }
    );
}

template<typename ValueType>
class Linear : public SurfaceInterpolationFactory<ValueType>::template Register<Linear<ValueType>>
{
    using Base = SurfaceInterpolationFactory<ValueType>::template Register<Linear<ValueType>>;

public:

    Linear(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    Linear(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};


    static std::string name() { return "linear"; }

    static std::string doc() { return "linear interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const override
    {
        computeLinearInterpolation(src, geometryScheme_->weights(), dst);
    }

    void interpolate(
        const SurfaceField<scalar>&, const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst
    ) const override
    {
        interpolate(src, dst);
    }


    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<Linear>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
