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
        std::visit(
            [&](auto exec) {
                setFixedGradient(
                    exec, domainField.boundaryField().refGrad().span(), fixedGradient_
                );
            },
            domainField.exec()
        );
    }

    static std::string name() { return "fixedGradient"; }

private:

    template<typename executor>
    void setFixedGradient(const executor& exec, std::span<ValueType> gradientField, ValueType value)
    {
        if constexpr (std::is_same<std::remove_reference_t<executor>, CPUExecutor>::value)
        {
            for (std::size_t i = this->patchStart(); i < this->patchEnd(); i++)
            {
                gradientField[i] = value;
            }
        }
        else
        {
            using runOn = typename executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(this->patchStart(), this->patchEnd()),
                KOKKOS_LAMBDA(std::size_t i) { gradientField[i] = value; }
            );
        }
    }

    ValueType fixedGradient_;
};

}
