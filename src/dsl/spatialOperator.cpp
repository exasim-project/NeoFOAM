// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#include "NeoFOAM/dsl/spatialOperator.hpp"

namespace NeoFOAM::dsl
{


SpatialOperator::SpatialOperator(const SpatialOperator& eqnOperator)
    : model_ {eqnOperator.model_->clone()}
{}

SpatialOperator::SpatialOperator(SpatialOperator&& eqnOperator)
    : model_ {std::move(eqnOperator.model_)}
{}

void SpatialOperator::explicitOperation(Field<scalar>& source)
{
    model_->explicitOperation(source);
}

void SpatialOperator::implicitOperation(la::LinearSystem<scalar, localIdx>& ls)
{
    model_->implicitOperation(ls);
}

la::LinearSystem<scalar, localIdx> SpatialOperator::createEmptyLinearSystem() const
{
    return model_->createEmptyLinearSystem();
}

SpatialOperator::Type SpatialOperator::getType() const { return model_->getType(); }

std::string SpatialOperator::getName() const { return model_->getName(); }

Coeff& SpatialOperator::getCoefficient() { return model_->getCoefficient(); }

Coeff SpatialOperator::getCoefficient() const { return model_->getCoefficient(); }

void SpatialOperator::build(const Input& input) { model_->build(input); }

const Executor& SpatialOperator::exec() const { return model_->exec(); }

SpatialOperator operator*(scalar scalarCoeff, SpatialOperator rhs)
{
    SpatialOperator result = rhs;
    result.getCoefficient() *= scalarCoeff;
    return result;
}

SpatialOperator operator*(const Field<scalar>& coeffField, SpatialOperator rhs)
{
    SpatialOperator result = rhs;
    result.getCoefficient() *= Coeff(coeffField);
    return result;
}

SpatialOperator operator*(const Coeff& coeff, SpatialOperator rhs)
{
    SpatialOperator result = rhs;
    result.getCoefficient() *= coeff;
    return result;
}


} // namespace NeoFOAM::dsl
