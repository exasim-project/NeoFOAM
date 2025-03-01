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

namespace NeoFOAM
{

std::string demangle(const char* mangledName);

template<typename T, typename Container, typename Key>
void logBadAnyCast(const std::bad_any_cast& e, const Key& key, const Container& data)
{
    std::cerr << "Caught a bad_any_cast exception: \n"
              << std::format("Caught a bad_any_cast exception: key requested: {}, requested type: {}, actual type: {}\n" , 
              key, 
              demangle(typeid(T).name()),
              demangle(data.at(key).type().name()))
}

}
