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
#include "NeoFOAM/core/collection.hpp"
#include "NeoFOAM/core/document.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

bool validateFieldDoc(const Document& doc);

// const versions
std::size_t timeIndex(const Document& doc);

std::size_t iterationIndex(const Document& doc);

std::int64_t subCycleIndex(const Document& doc);

// non-const versions
std::size_t& timeIndex(Document& doc);

std::size_t& iterationIndex(Document& doc);

std::int64_t& subCycleIndex(Document& doc);

template<typename ValueType>
VolumeField<ValueType>& volField(Document& doc)
{
    return doc.get<VolumeField<ValueType>>("field");
}

template<typename ValueType>
const VolumeField<ValueType>& volField(const Document& doc)
{
    return doc.get<VolumeField<ValueType>>("field");
}

class FieldDocument
{
public:

    std::string name;
    std::size_t timeIndex;
    std::size_t iterationIndex;
    std::int64_t subCycleIndex;
    std::any field;


    static Document create(FieldDocument fDoc);

    Document doc();
};


class FieldCollection
{
public:

    static const std::string typeName();

    static void create(NeoFOAM::Database& db, std::string name);

    static NeoFOAM::Collection& getCollection(NeoFOAM::Database& db, std::string name);

    static const NeoFOAM::Collection& getCollection(const NeoFOAM::Database& db, std::string name);

};


} // namespace NeoFOAM
