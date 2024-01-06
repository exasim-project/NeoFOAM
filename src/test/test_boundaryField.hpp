// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/blas/field.hpp"
#include "NeoFOAM/blas/boundaryFields.hpp"
#include "NeoFOAM/blas/domainField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccVolField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarFixedValueBoundaryField.hpp"


template <typename T, typename primitive>
void print_field(T &a, primitive value)
{
    std::cout << "a has a size of: " << a.size() << std::endl;
    Kokkos::View<primitive *, Kokkos::HostSpace> testview("testview", a.size());
    Kokkos::deep_copy(testview, a.field());
    primitive sum = 0.0;
    for (int i = 0; i < a.size(); i++)
    {
        std::cout << "value: " << testview(i) << " expected: " << value << std::endl;
        sum += testview(i);
    }
    std::cout << "sum: " << sum << std::endl;
}

TEST(BoundaryField, init)
{
    NeoFOAM::boundaryFields<double> BCs(100,10);

    NeoFOAM::domainField<double> a(1000,100,10);
    auto& aIn = a.internalField();

    fill_field(aIn, 2.0);

    copy_and_check_EQ(aIn, 2.0);

    NeoFOAM::fvccVolField<NeoFOAM::scalar> volField(1000,20,2);

    NeoFOAM::boundaryFields<NeoFOAM::scalar>& bField = volField.boundaryField();

    std::unique_ptr<NeoFOAM::fvccBoundaryField<NeoFOAM::scalar>> fixedUniformBC1 = std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(0, 10, 1.0);
    std::unique_ptr<NeoFOAM::fvccBoundaryField<NeoFOAM::scalar>> fixedUniformBC2 = std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(10, 20, 2.0);

    auto& volBCs = volField.boundaryConditions();

    volBCs[0] = std::move(fixedUniformBC1);
    volBCs[1] = std::move(fixedUniformBC2);

    std::cout << "volBCs.size(): " << volBCs.size() << std::endl;

    volField.correctBoundaryConditions();

    auto& bIn = bField.value();
    auto& bRefIn = bField.refValue();
    
    copy_and_check_EQ(bIn, 2.0);
    copy_and_check_EQ(bRefIn, 2.0);

    print_field(bIn, 2.0);
    print_field(bRefIn, 2.0);


}
