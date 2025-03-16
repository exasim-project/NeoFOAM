// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once
// TODO For WIN builds, needs to be ifdef'ed out.
#ifdef __GNUC__
#include <cxxabi.h>
#endif
#include <string>
#include <any>
#include <iostream>
#include <format>

namespace NeoFOAM
{

std::string demangle(const char* mangledName);

template<typename T, typename Container, typename Key>
void logBadAnyCast(const std::bad_any_cast& e, const Key& key, const Container& data)
{
    std::cerr << std::format(
        "Caught a bad_any_cast exception:\n",
        "key: {}\n",
        "requested type: {}\n",
        "actual type: {}\n",
        key,
        demangle(typeid(T).name()),
        demangle(data.at(key).type().name())
    ) << e.what()
              << std::endl;
}

}
