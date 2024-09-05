// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once
// TODO For WIN builds, needs to be ifdef'ed out.
#include <cxxabi.h>
#include <string>

namespace NeoFOAM
{

std::string demangle(const char* mangledName);

}
