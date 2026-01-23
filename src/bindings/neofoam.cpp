// SPDX-FileCopyrightText: 2024 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

NB_MODULE(neofoam_bindings, m)
{
    m.doc() = "NeoFOAM Python bindings - OpenFOAM adapter layer";

    m.def(
        "greet",
        []() { return "Hello from NeoFOAM C++ bindings!"; },
        "A simple function to test the bindings"
    );
}
