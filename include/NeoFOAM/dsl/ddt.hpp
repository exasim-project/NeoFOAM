// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoFOAM::dsl::temporal
{


// TODO add free factory function
template<typename FieldType>
class Ddt : public OperatorMixin<FieldType>
{

public:

    Ddt(FieldType& field) : OperatorMixin<FieldType>(field.exec(), field, Operator::Type::Temporal)
    {}

    std::string getName() const { return "TimeOperator"; }

    void explicitOperation([[maybe_unused]] Field<scalar>& source, [[maybe_unused]] scalar scale)
    {
        NF_ERROR_EXIT("Not implemented");
    }

    void implicitOperation([[maybe_unused]] Field<scalar>& phi)
    {
        NF_ERROR_EXIT("Not implemented");
    }

private:
};

// see
// https://github.com/exasim-project/NeoFOAM/blob/dsl/operatorIntergration/include/NeoFOAM/finiteVolume/cellCentred/operators/explicitOperators/expOp.hpp

template<typename FieldType>
Ddt<FieldType> ddt(FieldType& in)
{
    return Ddt(in);
};


} // namespace NeoFOAM
