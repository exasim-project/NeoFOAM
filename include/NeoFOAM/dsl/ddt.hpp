// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoFOAM::dsl::temporal
{

template<typename FieldType>
class Ddt : public OperatorMixin<FieldType>
{

public:

    Ddt(FieldType& field, Operator::Type termType) : OperatorMixin<FieldType>(field.exec(), field, termType)
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
};

/* @brief factory function to create a Ddt term as ddt() */
template<typename FieldType>
Ddt<FieldType> ddt(FieldType& in, Operator::Type termType)
{
    return Ddt(in, termType);
};

} // namespace NeoFOAM
