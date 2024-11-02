// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <any>       // for bad_any_cast
#include <iostream>  // for operator<<, basic_ostream, cerr, endl, ostream
#include <string>    // for operator<<, string
#include <typeinfo>  // for type_info

namespace NeoFOAM
{

std::string demangle(const char* mangledName);

template<typename T, typename Container, typename Key>
void logBadAnyCast(const std::bad_any_cast& e, const Key& key, const Container& data)
{
    std::cerr << "Caught a bad_any_cast exception: \n"
              << "requested type " << demangle(typeid(T).name()) << "\n"
              << "actual type " << demangle(data.at(key).type().name()) << "\n"
              << e.what() << std::endl;
}

}
