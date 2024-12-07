// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @class Factory class to create divergence operators by a given name using
 * using NeoFOAMs runTimeFactory mechanism
 */
class DivOperatorFactory :
    public RuntimeSelectionFactory<
        DivOperatorFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<DivOperatorFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, Input inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("DivOperator")
                            : std::get<TokenList>(inputs).pop_front<std::string>();
        keyExistsOrError(key);
        return table().at(key)(exec, uMesh, inputs);
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

    DivOperator(const DivOperator& op)
        : exec_(op.exec_), mesh_(op.mesh_),
          divOperatorStrategy_(
              op.divOperatorStrategy_ ? op.divOperatorStrategy_->clone() : nullptr
          ) {};

    DivOperator(DivOperator&& op)
        : exec_(op.exec_), mesh_(op.mesh_),
          divOperatorStrategy_(std::move(op.divOperatorStrategy_)) {};

    DivOperator(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<DivOperatorFactory> opStrategy
    )
        : exec_(exec), mesh_(mesh), divOperatorStrategy_(std::move(opStrategy)) {};

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
            exec_, name, mesh_, createCalculatedBCs<VolumeBoundary<scalar>>(mesh_)
        );
        fill(divPhi.internalField(), 0.0);
        fill(divPhi.boundaryField().value(), 0.0);
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
