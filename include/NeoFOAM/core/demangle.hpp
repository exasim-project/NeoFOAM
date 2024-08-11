// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <cxxabi.h>
#include <string>

namespace NeoFOAM
{

std::string demangle(const char* mangled_name);

}
