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
class FixedGradient : public VolumeBoundaryFactory<ValueType>
{

public:

    using FixedGradientType = FixedGradient<ValueType>;

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : VolumeBoundaryFactory<ValueType>(mesh, patchID),
          fixedGradient_(dict.get<ValueType>("fixedGradient"))
    {
        VolumeBoundaryFactory<ValueType>::template registerClass<FixedGradientType>();
    }

    static std::unique_ptr<VolumeBoundaryFactory<ValueType>>
    create(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
    {
        return std::make_unique<FixedGradientType>(mesh, dict, patchID);
    }

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override
    {
        auto boundarySpan = domainField.boundaryField().refGrad().span(this->range());
        std::visit(
            [&](auto exec) { setFixedGradient(exec, boundarySpan, fixedGradient_); },
            domainField.exec()
        );
    }

    static std::string name() { return "fixedGradient"; }

private:

    template<typename Executor>
    void setFixedGradient(const Executor& exec, std::span<ValueType> inField, ValueType targetValue)
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
	   // TODO implement
           // using runOn = typename Executor::exec;
           // Kokkos::parallel_for(
           //     "parallelForImpl",
           //     Kokkos::RangePolicy<runOn>(0, inField.size()),
           //     KOKKOS_LAMBDA(std::size_t i) { inField[i] = targetValue; }
           // );
        }
    }

    ValueType fixedGradient_;
};

}
