// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
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
                            : std::get<TokenList>(inputs).popFront<std::string>();
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

class DivOperator : public dsl::OperatorMixin<VolumeField<scalar>>
{

public:

    // copy constructor
    DivOperator(const DivOperator& divOp)
        : dsl::OperatorMixin<VolumeField<scalar>>(divOp.exec_, divOp.field_, divOp.type_),
          faceFlux_(divOp.faceFlux_),
          divOperatorStrategy_(
              divOp.divOperatorStrategy_ ? divOp.divOperatorStrategy_->clone() : nullptr
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), faceFlux_(faceFlux),
          divOperatorStrategy_(DivOperatorFactory::create(exec_, phi.mesh(), input)) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        std::unique_ptr<DivOperatorFactory> divOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), faceFlux_(faceFlux),
          divOperatorStrategy_(std::move(divOperatorStrategy)) {};

    DivOperator(
        dsl::Operator::Type termType, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), faceFlux_(faceFlux),
          divOperatorStrategy_(nullptr) {};


    void explicitOperation(Field<scalar>& source)
    {
        if (divOperatorStrategy_ == nullptr)
        {
            NF_ERROR_EXIT("DivOperatorStrategy not initialized");
        }
        NeoFOAM::Field<NeoFOAM::scalar> tmpsource(source.exec(), source.size(), 0.0);
        divOperatorStrategy_->div(tmpsource, faceFlux_, field_);
        source += tmpsource;
    }

    void div(Field<scalar>& divPhi) { divOperatorStrategy_->div(divPhi, faceFlux_, getField()); }

    void div(VolumeField<scalar>& divPhi)
    {
        divOperatorStrategy_->div(divPhi, faceFlux_, getField());
    }


    void build(const Input& input)
    {
        const UnstructuredMesh& mesh = field_.mesh();
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            auto dict = std::get<NeoFOAM::Dictionary>(input);
            std::string schemeName = "div(" + faceFlux_.name + "," + field_.name + ")";
            auto tokens = dict.subDict("divSchemes").get<NeoFOAM::TokenList>(schemeName);
            divOperatorStrategy_ = DivOperatorFactory::create(exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoFOAM::TokenList>(input);
            divOperatorStrategy_ = DivOperatorFactory::create(exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "DivOperator"; }

private:

    const SurfaceField<NeoFOAM::scalar>& faceFlux_;

    std::unique_ptr<DivOperatorFactory> divOperatorStrategy_;
};


} // namespace NeoFOAM
