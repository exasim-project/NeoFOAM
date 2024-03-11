// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <Kokkos_Core.hpp>

namespace NeoFOAM
{

struct hello_world
{
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const { printf("Hello from i = %i\n", i); }
};

void foo() { Kokkos::parallel_for("HelloWorld", 15, hello_world()); }
} // namespace NeoFOAM
