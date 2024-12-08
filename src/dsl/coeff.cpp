// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/dsl/coeff.hpp"

namespace NeoFOAM::dsl
{

Coeff::Coeff() : coeff_(1.0), span_(), hasSpan_(false) {}

Coeff::Coeff(scalar value) : coeff_(value), span_(), hasSpan_(false) {}

Coeff::Coeff(scalar coeff, const Field<scalar>& field)
    : coeff_(coeff), span_(field.span()), hasSpan_(true)
{}

Coeff::Coeff(const Field<scalar>& field) : coeff_(1.0), span_(field.span()), hasSpan_(true) {}

bool Coeff::hasSpan() { return hasSpan_; }

std::span<const scalar> Coeff::span() { return span_; }

Coeff& Coeff::operator*=(scalar rhs)
{
    coeff_ *= rhs;
    return *this;
}

Coeff& Coeff::operator*=(const Coeff& rhs)
{
    if (hasSpan_ && rhs.hasSpan_)
    {
        NF_ERROR_EXIT("Not implemented");
    }

    if (!hasSpan_ && rhs.hasSpan_)
    {
        // Take over the span
        span_ = rhs.span_;
        hasSpan_ = true;
    }

    return this->operator*=(rhs.coeff_);
}

namespace detail
{
void toField(Coeff& coeff, Field<scalar>& rhs)
{
    if (coeff.hasSpan())
    {
        rhs.resize(coeff.span().size());
        fill(rhs, 1.0);
        auto rhsSpan = rhs.span();
        // otherwise we are unable to capture values in the lambda
        parallelFor(
            rhs.exec(), rhs.range(), KOKKOS_LAMBDA(const size_t i) { rhsSpan[i] *= coeff[i]; }
        );
    }
    else
    {
        rhs.resize(1);
        fill(rhs, coeff[0]);
    }
}
} // namespace detail


} // namespace NeoFOAM::dsl
