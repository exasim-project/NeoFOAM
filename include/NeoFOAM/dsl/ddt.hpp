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

    void explicitOperation(Field<scalar>& source, scalar scale)
    {
        auto sourceField = source.span();
        parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1 * scale; }
        );
    }

    void temporalOperation(Field<scalar>& phi)
    {
        auto phiSpan = phi.span();
        auto fieldSpan = field_.internalField().span();
        parallelFor(
            phi.exec(), phi.range(), KOKKOS_LAMBDA(const size_t i) { fieldSpan[i] += phiSpan[i]; }
        );
    }

    VolumeField& volumeField() const { return field_; }

    size_t getSize() const { return field_.internalField().size(); }

private:

    VolumeField& field_;
};


} // namespace NeoFOAM
