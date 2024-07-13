// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
class FixedValue : public VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), fixedValue_(dict.get<ValueType>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override
    {
        auto boundarySpan = domainField.boundaryField().refValue().span(this->range());
        std::visit(
            [&](auto exec) { setFixedValue(exec, boundarySpan, fixedValue_); }, domainField.exec()
        );
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

    // NOTE: this function can not be private or
    // it will yield the following error:
    // The enclosing parent function for an extended __host__ __device__ lambda cannot have
    // private or protected access within its cla
    template<typename Executor>
    void setFixedValue(const Executor& exec, std::span<ValueType> inField, ValueType targetValue)
    {
        if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
        {
            for (auto& value : inField)
            {
                value = targetValue;
            }
        }
        else
        {
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(0, inField.size()),
                KOKKOS_LAMBDA(std::size_t i) { inField[i] = targetValue; }
            );
        }
    }

private:

    ValueType fixedValue_;
};

}
