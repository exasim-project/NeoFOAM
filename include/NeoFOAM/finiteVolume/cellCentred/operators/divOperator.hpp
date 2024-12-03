// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
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
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<DivOperatorFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, Input inputs)
    {
        std::string key;
        // input is dictionary the key is "interpolation"
        if (std::holds_alternative<NeoFOAM::Dictionary>(inputs))
        {
            key = std::get<NeoFOAM::Dictionary>(inputs).get<std::string>("DivOperator");
        }
        else
        {
            key = std::get<NeoFOAM::TokenList>(inputs).get<std::string>(0);
            std::get<NeoFOAM::TokenList>(inputs).remove(0);
        }
        keyExistsOrError(key);
        auto ptr = table().at(key)(exec, uMesh, inputs);
        return ptr;
    }

    static std::string name() { return "DivOperatorFactory"; }

    DivOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~DivOperatorFactory() {} // Virtual destructor

    virtual void
    div(VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    ) = 0;

    virtual void
    div(Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) = 0;

    virtual VolumeField<scalar>
    div(const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) = 0;

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
          divOperatorStrategy_(
              divOperator.divOperatorStrategy_ ? divOperator.divOperatorStrategy_->clone() : nullptr
          ) {};

    DivOperator(DivOperator&& divOperator)
        : exec_(divOperator.exec_), mesh_(divOperator.mesh_),
          divOperatorStrategy_(std::move(divOperator.divOperatorStrategy_)) {};

    DivOperator(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<DivOperatorFactory> divOperatorStrategy
    )
        : exec_(exec), mesh_(mesh), divOperatorStrategy_(std::move(divOperatorStrategy)) {};

    DivOperator(const Executor& exec, const UnstructuredMesh& mesh, const Input& input)
        : exec_(exec), mesh_(mesh),
          divOperatorStrategy_(DivOperatorFactory::create(exec, mesh, input)) {};

    DivOperator(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh), divOperatorStrategy_() {};

    void build(const Input& input)
    {
        divOperatorStrategy_ = DivOperatorFactory::create(exec_, mesh_, input);
    }

    VolumeField<scalar> div(const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) const
    {
        std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
        VolumeField<scalar> divPhi(
            exec_, name,mesh_, VolumeBoundary<scalar>::calculatedBCs(mesh_)
        );
        NeoFOAM::fill(divPhi.internalField(), 0.0);
        NeoFOAM::fill(divPhi.boundaryField().value(), 0.0);
        divOperatorStrategy_->div(divPhi, faceFlux, phi);
        return divPhi;
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
