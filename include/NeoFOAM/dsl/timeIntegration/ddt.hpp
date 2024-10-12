// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoFOAM::dsl::temporal
{

class Ddt : public OperatorMixin
{

public:

    Ddt(const Executor& exec, VolumeField& field) : OperatorMixin(exec), field_(field) {}

    std::string getName() const { return "TimeOperator"; }

    Operator::Type getType() const { return Operator::Type::Temporal; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
    {
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1 * scale; }
        );
    }

    VolumeField& volumeField() const { return field_; }

    size_t getSize() const { return field_.internalField().size(); }

    const Executor& exec() const { return exec_; }

    const Executor exec_;

    std::size_t nCells_;

    VolumeField& field_;
};


} // namespace NeoFOAM
