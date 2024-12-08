// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#include "NeoFOAM/dsl/operator.hpp"

namespace NeoFOAM::dsl
{


Operator::Operator(const Operator& eqnOperator) : model_ {eqnOperator.model_->clone()} {}

Operator::Operator(Operator&& eqnOperator) : model_ {std::move(eqnOperator.model_)} {}

void Operator::explicitOperation(Field<scalar>& source) { model_->explicitOperation(source); }

void Operator::temporalOperation(Field<scalar>& field) { model_->temporalOperation(field); }

Operator::Type Operator::getType() const { return model_->getType(); }

Coeff& Operator::getCoefficient() { return model_->getCoefficient(); }

Coeff Operator::getCoefficient() const { return model_->getCoefficient(); }

void Operator::build(const Input& input) { model_->build(input); }

const Executor& Operator::exec() const { return model_->exec(); }

Operator operator*(scalar scalarCoeff, Operator rhs)
{
    Operator result = rhs;
    result.getCoefficient() *= scalarCoeff;
    return result;
}

Operator operator*(const Field<scalar>& coeffField, Operator rhs)
{
    Operator result = rhs;
    result.getCoefficient() *= Coeff(coeffField);
    return result;
}

Operator operator*(const Coeff& coeff, Operator rhs)
{
    Operator result = rhs;
    result.getCoefficient() *= coeff;
    return result;
}


} // namespace NeoFOAM::dsl
