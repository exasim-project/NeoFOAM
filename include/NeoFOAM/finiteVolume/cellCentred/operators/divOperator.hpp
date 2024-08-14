// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{

class DivOperatorFactory :
    public NeoFOAM::RuntimeSelectionFactory<
        DivOperatorFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const SurfaceInterpolation&>>
{

public:

    static std::string name() { return "DivOperatorFactory"; }

    DivOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~DivOperatorFactory() {} // Virtual destructor

    virtual void
    div(VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    ) = 0;

    virtual void
    div(Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<DivOperatorFactory> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

class DivOperator
{

public:

    DivOperator(const DivOperator& divOperator)
        : exec_(divOperator.exec_), mesh_(divOperator.mesh_),
          divOperatorStrategy_(divOperator.divOperatorStrategy_->clone()) {};

    DivOperator(DivOperator&& divOperator)
        : exec_(divOperator.exec_), mesh_(divOperator.mesh_),
          divOperatorStrategy_(std::move(divOperator.divOperatorStrategy_)) {};

    DivOperator(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<DivOperatorFactory> divOperatorStrategy
    )
        : exec_(exec), mesh_(mesh), divOperatorStrategy_(std::move(divOperatorStrategy)) {};

    // DivOperator(
    //     const Executor& exec, const UnstructuredMesh& mesh, std::string interpolationScheme
    // )
    //     : exec_(exec), mesh_(mesh),
    //       divOperatorStrategy_(DivOperatorFactory::create(interpolationScheme, exec, mesh)
    //       ) {};

    void
    div(VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    ) const
    {
        divOperatorStrategy_->div(divPhi, faceFlux, phi);
    }

    void
    div(Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) const
    {
        divOperatorStrategy_->div(divPhi, faceFlux, phi);
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<DivOperatorFactory> divOperatorStrategy_;
};


} // namespace NeoFOAM
