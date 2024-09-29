// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <functional>
#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"

#include "NeoFOAM/core/database.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class FieldCollection
{
public:

    FieldCollection(std::string name, NeoFOAM::Database& db);

    static void registerCollection(std::string name, NeoFOAM::Database& db);

private:
    static const std::string typeName;
    std::string name_;
    NeoFOAM::Database& db_;
    NeoFOAM::Collection& collection_;

};


} // namespace NeoFOAM
