// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @class Factory class to create divergence operators by a given name using
 * using NeoNs runTimeFactory mechanism
 */
template<typename ValueType>
class LaplacianOperatorFactory :
    public RuntimeSelectionFactory<
        LaplacianOperatorFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<LaplacianOperatorFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("LaplacianOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        LaplacianOperatorFactory<ValueType>::keyExistsOrError(key);
        return LaplacianOperatorFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "LaplacianOperatorFactory"; }

    LaplacianOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~LaplacianOperatorFactory() {} // Virtual destructor

    virtual void laplacian(
        VolumeField<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    virtual void laplacian(
        Field<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    virtual void laplacian(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<LaplacianOperatorFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;
};

template<typename ValueType>
class LaplacianOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    // copy constructor
    LaplacianOperator(const LaplacianOperator& lapOp)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            lapOp.exec_, lapOp.coeffs_, lapOp.field_, lapOp.type_
        ),
          gamma_(lapOp.gamma_),
          laplacianOperatorStrategy_(
              lapOp.laplacianOperatorStrategy_ ? lapOp.laplacianOperatorStrategy_->clone() : nullptr
          ) {};

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          gamma_(gamma),
          laplacianOperatorStrategy_(
              LaplacianOperatorFactory<ValueType>::create(this->exec_, phi.mesh(), input)
          ) {};

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        std::unique_ptr<LaplacianOperatorFactory<ValueType>> laplacianOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          gamma_(gamma), laplacianOperatorStrategy_(std::move(laplacianOperatorStrategy)) {};

    LaplacianOperator(
        dsl::Operator::Type termType, const SurfaceField<scalar>& gamma, VolumeField<ValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          gamma_(gamma), laplacianOperatorStrategy_(nullptr) {};


    void explicitOperation(Field<ValueType>& source) const
    {
        NF_ASSERT(laplacianOperatorStrategy_, "LaplacianOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        NeoN::Field<ValueType> tmpsource(source.exec(), source.size(), zero<ValueType>());
        laplacianOperatorStrategy_->laplacian(tmpsource, gamma_, this->field_, operatorScaling);
        source += tmpsource;
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls)
    {
        NF_ASSERT(laplacianOperatorStrategy_, "LaplacianOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        laplacianOperatorStrategy_->laplacian(ls, gamma_, this->field_, operatorScaling);
    }

    // void laplacian(Field<scalar>& lapPhi)
    // {
    //     laplacianOperatorStrategy_->laplacian(lapPhi, gamma_, getField());
    // }

    // void laplacian(la::LinearSystem<scalar, localIdx>& ls)
    // {
    //     laplacianOperatorStrategy_->laplacian(ls, gamma_, getField());
    // };

    void laplacian(VolumeField<scalar>& lapPhi)
    {
        const auto operatorScaling = this->getCoefficient();
        laplacianOperatorStrategy_->laplacian(lapPhi, gamma_, this->getField(), operatorScaling);
    }


    void build(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName = "laplacian(" + gamma_.name + "," + this->field_.name + ")";
            auto tokens = dict.subDict("laplacianSchemes").get<NeoN::TokenList>(schemeName);
            laplacianOperatorStrategy_ =
                LaplacianOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            laplacianOperatorStrategy_ =
                LaplacianOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "LaplacianOperator"; }

private:

    const SurfaceField<NeoN::scalar>& gamma_;

    std::unique_ptr<LaplacianOperatorFactory<ValueType>> laplacianOperatorStrategy_;
};


} // namespace NeoN
