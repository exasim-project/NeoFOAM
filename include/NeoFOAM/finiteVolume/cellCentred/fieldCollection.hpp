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
#include "NeoFOAM/core/document.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{



bool validateFieldDoc(const Document& doc);

class FieldDocument
{
public:

    std::string name;
    std::size_t timeIndex;
    std::size_t iterationIndex;
    std::int64_t subCycleIndex;

    
    static NeoFOAM::Document create(FieldDocument fDoc);

    Document doc();
};

// const versions
std::size_t timeIndex(const NeoFOAM::Document& doc);

std::size_t iterationIndex(const NeoFOAM::Document& doc);

std::int64_t subCycleIndex(const NeoFOAM::Document& doc);

// non-const versions
std::size_t& timeIndex(NeoFOAM::Document& doc);

std::size_t& iterationIndex(NeoFOAM::Document& doc);

std::int64_t& subCycleIndex(NeoFOAM::Document& doc);


} // namespace NeoFOAM
