// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
class FixedValue : public VolumeBoundaryFactory<ValueType>
{

public:

    using FixedValueType = FixedValue<ValueType>;

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : VolumeBoundaryFactory<ValueType>(mesh, patchID),
          fixedValue_(dict.get<ValueType>("uniformValue"))
    {
        VolumeBoundaryFactory<ValueType>::template registerClass<FixedValueType>();
    }

    static std::unique_ptr<VolumeBoundaryFactory<ValueType>>
    create(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
    {
        return std::make_unique<FixedValueType>(mesh, dict, patchID);
    }

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override
    {
        std::visit(
            [&](auto exec)
            { setFixedValue(exec, domainField.boundaryField().refValue().field(), fixedValue_); },
            domainField.exec()
        );
    }

    static std::string name() { return "FixedValue"; }

private:

    template<typename executor>
    void setFixedValue(const executor& exec, std::span<ValueType> valueField, ValueType value)
    {
        if constexpr (std::is_same<std::remove_reference_t<executor>, CPUExecutor>::value)
        {
            for (std::size_t i = this->patchStart(); i < this->patchEnd(); i++)
            {
                valueField[i] = value;
            }
        }
        else
        {
            using runOn = typename executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(this->patchStart(), this->patchEnd()),
                KOKKOS_LAMBDA(std::size_t i) { valueField[i] = value; }
            );
        }
    }

    ValueType fixedValue_;
};

}
