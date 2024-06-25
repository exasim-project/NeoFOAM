// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryBase.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
class FixedGradient : public VolumeBoundaryFactory<ValueType>, public BoundaryPatchMixin
{

public:

    using FixedGradientType = FixedGradient<ValueType>;

    template<typename executor>
    void setFixedGradient(const executor& exec, std::span<ValueType> field, ValueType value)
    {
        if constexpr (std::is_same<std::remove_reference_t<executor>, CPUExecutor>::value)
        {
            for (std::size_t i = start_; i < end_; i++)
            {
                field[i] = value;
            }
        }
        else
        {
            using runOn = typename executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(start_, end_),
                KOKKOS_LAMBDA(std::size_t i) { field[i] = value; }
            );
        }
    }

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : VolumeBoundaryFactory<ValueType>(), BoundaryPatchMixin(mesh, patchID),
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
                    exec, domainField.boundaryField().refValue().field(), fixedGradient_
                );
            },
            domainField.exec()
        );
    }

    static std::string name() { return "fixedGradient"; }

private:

    ValueType fixedGradient_;
};

}
