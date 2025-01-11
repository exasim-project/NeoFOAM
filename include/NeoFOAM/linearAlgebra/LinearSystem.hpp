// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"


namespace NeoFOAM::la
{

/**
 * @class LinearSystem
 * @brief A class representing a linear system of equations.
 *
 * The LinearSystem class provides functionality to store and manipulate a linear system of
 * equations. It supports the storage of the coefficient matrix and the right-hand side vector, as
 * well as the solution vector.
 */
template<typename ValueType, typename IndexType>
class LinearSystem
{
public:

    LinearSystem(const CSRMatrix<ValueType, IndexType>& matrix, const Field<ValueType>& rhs)
        : matrix_(matrix), rhs_(rhs) {};

    ~LinearSystem() = default;

    [[nodiscard]] CSRMatrix<ValueType, IndexType>& matrix() { return matrix_; }
    [[nodiscard]] Field<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const CSRMatrix<ValueType, IndexType>& matrix() const { return matrix_; }
    [[nodiscard]] const Field<ValueType>& rhs() const { return rhs_; }

private:

    CSRMatrix<ValueType, IndexType> matrix_;
    Field<ValueType> rhs_;
};

} // namespace NeoFOAM::la
