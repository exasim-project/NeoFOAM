// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

namespace NeoFOAM::dsl
{

/**
 * @class Coeff
 * @brief A class that represents a coefficient for the NeoFOAM DSL.
 *
 * This class stores a single scalar coefficient and optionally span of values.
 * It is used to delay the evaluation of a scalar multiplication with a field to
 * avoid the creation of a temporary field copy.
 * It provides an indexing operator `operator[]` that returns the evaluated value at the specified
 * index.
 */
class Coeff
{

public:

    Coeff() : coeff_(1.0), span_(), hasSpan_(false) {}

    Coeff(scalar value) : coeff_(value), span_(), hasSpan_(false) {}

    Coeff(scalar coeff, const Field<scalar>& field)
        : coeff_(coeff), span_(field.span()), hasSpan_(true)
    {}

    Coeff(const Field<scalar>& field) : coeff_(1.0), span_(field.span()), hasSpan_(true) {}

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const size_t i) const { return (hasSpan_) ? span_[i] * coeff_ : coeff_; }

    Coeff& operator*=(scalar rhs)
    {
        coeff_ *= rhs;
        return *this;
    }
    /* function to force evaluation to a newly created field */
    void toField(Field<scalar>& rhs)
    {
        rhs.resize(span_.size());
        fill(rhs, 1.0);
        auto rhsSpan = rhs.span();
        // otherwise we are unable to capture values in the lambda
        auto& coeff = *this;
        parallelFor(
            rhs.exec(), rhs.range(), KOKKOS_LAMBDA(const size_t i) { rhsSpan[i] *= coeff[i]; }
        );
    }

    Coeff& operator*=(const Coeff& rhs)
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


private:

    scalar coeff_;

    std::span<const scalar> span_;

    bool hasSpan_;
};


inline Coeff operator*(const Coeff& lhs, const Coeff& rhs)
{
    Coeff result = lhs;
    result *= rhs;
    return result;
}

} // namespace NeoFOAM::dsl
