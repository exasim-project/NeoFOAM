// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM::dsl
{

/**
 * @class Coeff
 * @brief A class that represents a coefficient for the NeoFOAM dsl.
 *
 * This class stores a single scalar coefficient and optionally view of values.
 * It is used to delay the evaluation of a scalar multiplication with a field to
 * avoid the creation of a temporary field copy.
 * It provides an indexing operator `operator[]` that returns the evaluated value at the specified
 * index.
 */
class Coeff
{

public:

    Coeff();

    Coeff(scalar value);

    Coeff(scalar coeff, const Field<scalar>& field);

    Coeff(const Field<scalar>& field);

    KOKKOS_INLINE_FUNCTION
    scalar operator[](const size_t i) const { return (hasView_) ? view_[i] * coeff_ : coeff_; }

    bool hasSpan();

    View<const scalar> view();

    Coeff& operator*=(scalar rhs);


    Coeff& operator*=(const Coeff& rhs);


private:

    scalar coeff_;

    View<const scalar> view_;

    bool hasView_;
};


[[nodiscard]] inline Coeff operator*(const Coeff& lhs, const Coeff& rhs)
{
    Coeff result = lhs;
    result *= rhs;
    return result;
}

namespace detail
{
/* @brief function to force evaluation to a field, the field will be resized to hold either a
 * single value or the full field
 *
 * @param field to store the result
 */
void toField(Coeff& coeff, Field<scalar>& rhs);

} // namespace detail

} // namespace NeoFOAM::dsl
