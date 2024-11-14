// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <cxxabi.h> // for __cxa_demangle
#include <stdlib.h> // for free

#include "NeoFOAM/core/demangle.hpp"


std::string NeoFOAM::demangle(const char* name)
{
    int status;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0)
    {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    else
    {
        return name;
    }
}
