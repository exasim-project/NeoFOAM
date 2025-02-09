// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include "NeoFOAM/dsl/temporalOperator.hpp"

namespace NeoFOAM::dsl
{


TemporalOperator::TemporalOperator(const TemporalOperator& eqnOperator)
    : model_ {eqnOperator.model_->clone()}
{}

TemporalOperator::TemporalOperator(TemporalOperator&& eqnOperator)
    : model_ {std::move(eqnOperator.model_)}
{}

void TemporalOperator::explicitOperation(Field<scalar>& source, scalar t, scalar dt)
{
    model_->explicitOperation(source, t, dt);
}


void TemporalOperator::implicitOperation(
    la::LinearSystem<scalar, localIdx>& ls, scalar t, scalar dt
)
{
    model_->implicitOperation(ls, t, dt);
}

la::LinearSystem<scalar, localIdx> TemporalOperator::createEmptyLinearSystem() const
{
    return model_->createEmptyLinearSystem();
}

SpatialOperator::Type TemporalOperator::getType() const { return model_->getType(); }

std::string TemporalOperator::getName() const { return model_->getName(); }

Coeff& TemporalOperator::getCoefficient() { return model_->getCoefficient(); }

Coeff TemporalOperator::getCoefficient() const { return model_->getCoefficient(); }

void TemporalOperator::build(const Input& input) { model_->build(input); }

const Executor& TemporalOperator::exec() const { return model_->exec(); }

TemporalOperator operator*(scalar scalarCoeff, TemporalOperator rhs)
{
    TemporalOperator result = rhs;
    result.getCoefficient() *= scalarCoeff;
    return result;
}

TemporalOperator operator*(const Field<scalar>& coeffField, TemporalOperator rhs)
{
    TemporalOperator result = rhs;
    result.getCoefficient() *= Coeff(coeffField);
    return result;
}

TemporalOperator operator*(const Coeff& coeff, TemporalOperator rhs)
{
    TemporalOperator result = rhs;
    result.getCoefficient() *= coeff;
    return result;
}


} // namespace NeoFOAM::dsl
