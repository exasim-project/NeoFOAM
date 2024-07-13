// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::surfaceBoundary
{

template<typename ValueType>
class FixedGradient :
    public SurfaceBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>;

public:

    using FixedGradientType = FixedGradient<ValueType>;

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), fixedGradient_(dict.get<ValueType>("fixedGradient"))
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override
    {
        auto boundarySpan = domainField.boundaryField().refGrad().span(this->range());
        std::visit(
            [&](auto exec) { setFixedGradient(exec, boundarySpan, fixedGradient_); },
            domainField.exec()
        );
    }

    static std::string name() { return "fixedGradient"; }

    static std::string doc() { return "Set a fixed gradient on the boundary."; }

    static std::string schema() { return "none"; }

    // NOTE: this function can not be private or
    // it will yield the following error:
    // The enclosing parent function for an extended __host__ __device__ lambda cannot have
    // private or protected access within its class
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
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(0, inField.size()),
                KOKKOS_LAMBDA(std::size_t i) { inField[i] = targetValue; }
            );
        }
    }

private:

    ValueType fixedGradient_;
};

}
