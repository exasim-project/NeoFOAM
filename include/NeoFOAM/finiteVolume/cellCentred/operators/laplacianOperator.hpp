// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/runtimeSelectionFactory.hpp"
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
class LaplacianOperatorFactory :
    public RuntimeSelectionFactory<
        LaplacianOperatorFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<LaplacianOperatorFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, Input inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("LaplacianOperator")
                            : std::get<TokenList>(inputs).popFront<std::string>();
        keyExistsOrError(key);
        return table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "LaplacianOperatorFactory"; }

    LaplacianOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~LaplacianOperatorFactory() {} // Virtual destructor

    virtual la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const = 0;

    virtual void laplacian(
        VolumeField<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
    ) = 0;


    // Pure virtual function for cloning
    virtual std::unique_ptr<LaplacianOperatorFactory> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;
};

class LaplacianOperator : public dsl::OperatorMixin<VolumeField<scalar>>
{

public:

    // copy constructor
    LaplacianOperator(const LaplacianOperator& divOp)
        : dsl::OperatorMixin<VolumeField<scalar>>(divOp.exec_, divOp.field_, divOp.type_),
          gamma_(divOp.gamma_),
          laplacianOperatorStrategy_(
              divOp.laplacianOperatorStrategy_ ? divOp.laplacianOperatorStrategy_->clone() : nullptr
          ) {};

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<scalar>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), gamma_(gamma),
          laplacianOperatorStrategy_(LaplacianOperatorFactory::create(exec_, phi.mesh(), input)) {};

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<scalar>& phi,
        std::unique_ptr<LaplacianOperatorFactory> laplacianOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), gamma_(gamma),
          laplacianOperatorStrategy_(std::move(laplacianOperatorStrategy)) {};

    LaplacianOperator(
        dsl::Operator::Type termType, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), phi, termType), gamma_(gamma),
          laplacianOperatorStrategy_(nullptr) {};


    void explicitOperation(Field<scalar>& source)
    {
        // if (laplacianOperatorStrategy_ == nullptr)
        // {
        //     NF_ERROR_EXIT("LaplacianOperatorStrategy not initialized");
        // }
        // NeoFOAM::Field<NeoFOAM::scalar> tmpsource(source.exec(), source.size(), 0.0);
        // laplacianOperatorStrategy_->div(tmpsource, gamma_, field_);
        // source += tmpsource;
    }

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const
    {
        if (laplacianOperatorStrategy_ == nullptr)
        {
            NF_ERROR_EXIT("LaplacianOperatorStrategy not initialized");
        }
        return laplacianOperatorStrategy_->createEmptyLinearSystem();
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>& ls)
    {
        // if (laplacianOperatorStrategy_ == nullptr)
        // {
        //     NF_ERROR_EXIT("LaplacianOperatorStrategy not initialized");
        // }
        // laplacianOperatorStrategy_->div(ls, gamma_, field_);
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
        laplacianOperatorStrategy_->laplacian(lapPhi, gamma_, getField());
    }


    void build(const Input& input)
    {
        const UnstructuredMesh& mesh = field_.mesh();
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            auto dict = std::get<NeoFOAM::Dictionary>(input);
            std::string schemeName = "div(" + gamma_.name + "," + field_.name + ")";
            auto tokens = dict.subDict("divSchemes").get<NeoFOAM::TokenList>(schemeName);
            laplacianOperatorStrategy_ = LaplacianOperatorFactory::create(exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoFOAM::TokenList>(input);
            laplacianOperatorStrategy_ = LaplacianOperatorFactory::create(exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "LaplacianOperator"; }

private:

    const SurfaceField<NeoFOAM::scalar>& gamma_;

    std::unique_ptr<LaplacianOperatorFactory> laplacianOperatorStrategy_;
};


} // namespace NeoFOAM
