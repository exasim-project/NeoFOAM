// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

template<typename ValueType, typename IndexType>
class COOMatrix
{

public:

    COOMatrix(
        const Field<ValueType>& values, const Field<IndexType>& rows, const Field<IndexType>& cols
    )
        : values_(values), row_idxs_(rows), col_idxs_(cols) {};

    ~COOMatrix() = default;

    [[nodiscard]] Field<ValueType>& values() { return values_; }
    [[nodiscard]] Field<IndexType>& rows() { return row_idxs_; }
    [[nodiscard]] Field<IndexType>& cols() { return col_idxs_; }

    [[nodiscard]] const Field<ValueType>& values() const { return values_; }
    [[nodiscard]] const Field<IndexType>& rows() const { return row_idxs_; }
    [[nodiscard]] const Field<IndexType>& cols() const { return col_idxs_; }


private:

    Field<ValueType> values_;
    Field<IndexType> row_idxs_;
    Field<IndexType> col_idxs_;
};

} // namespace NeoFOAM
