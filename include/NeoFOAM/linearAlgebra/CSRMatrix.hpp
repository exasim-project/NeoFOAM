// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

template<typename ValueType, typename IndexType>
class CSRMatrix
{

public:

    CSRMatrix(
        const Field<ValueType>& values,
        const Field<IndexType>& colIdxs,
        const Field<IndexType>& rowPtrs
    )
        : values_(values), colIdxs_(colIdxs), rowPtrs_(rowPtrs) {};

    ~CSRMatrix() = default;

    IndexType nRows() const { return rowPtrs_.size() - 1; }

    IndexType nValues() const { return values_.size(); }

    IndexType nColIdxs() const { return colIdxs_.size(); }

    [[nodiscard]] Field<ValueType>& values() { return values_; }
    [[nodiscard]] Field<IndexType>& colIdxs() { return colIdxs_; }
    [[nodiscard]] Field<IndexType>& rowPtrs() { return rowPtrs_; }

    [[nodiscard]] const Field<ValueType>& values() const { return values_; }
    [[nodiscard]] const Field<IndexType>& colIdxs() const { return colIdxs_; }
    [[nodiscard]] const Field<IndexType>& rowPtrs() const { return rowPtrs_; }


private:

    Field<ValueType> values_;
    Field<IndexType> colIdxs_;
    Field<IndexType> rowPtrs_;
};

} // namespace NeoFOAM
