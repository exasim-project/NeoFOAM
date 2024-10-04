// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/demangle.hpp"


std::string NeoFOAM::demangle(const char* name)
{
#ifdef _MSC_VER
    return name; // For MSVC, return the mane directly.
#elif defined(__GNUC__)
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
#endif 
}
