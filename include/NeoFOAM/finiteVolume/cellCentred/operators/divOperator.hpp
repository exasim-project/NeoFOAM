// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @class Factory class to create divergence operators by a given name using
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename ValueType>
class DivOperatorFactory :
    public RuntimeSelectionFactory<
        DivOperatorFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<DivOperatorFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("DivOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        DivOperatorFactory<ValueType>::keyExistsOrError(key);
        return DivOperatorFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "DivOperatorFactory"; }

    DivOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~DivOperatorFactory() {} // Virtual destructor

    virtual la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const = 0;

    // NOTE currently simple overloading is used here, because templating the virtual function
    // does not work and we cant template the entire class because the static create function
    // cannot access keyExistsOrError and table anymore.
    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) = 0;

    virtual void
    div(la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) = 0;

    virtual void
    div(Field<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) = 0;

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<DivOperatorFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;
};

template<typename ValueType>
class DivOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    // copy constructor
    DivOperator(const DivOperator& divOp)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            divOp.exec_, divOp.coeffs_, divOp.field_, divOp.type_
        ),
          faceFlux_(divOp.faceFlux_),
          divOperatorStrategy_(
              divOp.divOperatorStrategy_ ? divOp.divOperatorStrategy_->clone() : nullptr
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux),
          divOperatorStrategy_(DivOperatorFactory<ValueType>::create(phi.exec(), phi.mesh(), input)
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        std::unique_ptr<DivOperatorFactory<ValueType>> divOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), divOperatorStrategy_(std::move(divOperatorStrategy)) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), divOperatorStrategy_(nullptr) {};


    void explicitOperation(Field<scalar>& source)
    {
        if (divOperatorStrategy_ == nullptr)
        {
            NF_ERROR_EXIT("DivOperatorStrategy not initialized");
        }
        NeoFOAM::Field<NeoFOAM::scalar> tmpsource(source.exec(), source.size(), 0.0);
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(tmpsource, faceFlux_, this->getField(), operatorScaling);
        source += tmpsource;
    }

    void div(Field<ValueType>& divPhi)
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(divPhi, faceFlux_, this->getField(), operatorScaling);
    }

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const
    {
        if (divOperatorStrategy_ == nullptr)
        {
            NF_ERROR_EXIT("DivOperatorStrategy not initialized");
        }
        return divOperatorStrategy_->createEmptyLinearSystem();
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls)
    {
        if (divOperatorStrategy_ == nullptr)
        {
            NF_ERROR_EXIT("DivOperatorStrategy not initialized");
        }
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(ls, faceFlux_, this->getField(), operatorScaling);
    }


    void div(la::LinearSystem<ValueType, localIdx>& ls)
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(ls, faceFlux_, this->getField(), operatorScaling);
    };

    void div(VolumeField<ValueType>& divPhi)
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(divPhi, faceFlux_, this->getField(), operatorScaling);
    }


    void build(const Input& input)
    {
        const UnstructuredMesh& mesh = this->getField().mesh();
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            auto dict = std::get<NeoFOAM::Dictionary>(input);
            std::string schemeName = "div(" + faceFlux_.name + "," + this->getField().name + ")";
            auto tokens = dict.subDict("divSchemes").get<NeoFOAM::TokenList>(schemeName);
            divOperatorStrategy_ =
                DivOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoFOAM::TokenList>(input);
            divOperatorStrategy_ =
                DivOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "DivOperator"; }

private:

    const SurfaceField<NeoFOAM::scalar>& faceFlux_;

    std::unique_ptr<DivOperatorFactory<ValueType>> divOperatorStrategy_;
};


} // namespace NeoFOAM
