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
#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"


namespace NeoFOAM::finiteVolume::cellCentred
{


class OldTimeDocument
{
public:

    std::string name;
    std::size_t timeIndex;
    std::size_t iterationIndex;
    std::int64_t subCycleIndex;
    std::any field;


    // static Document create(FieldDocument fDoc);

    Document doc();
};


class OldTimeCollection
{
public:

    OldTimeCollection(std::shared_ptr<NeoFOAM::Collection> collection);

    static const std::string typeName();

    static void create(NeoFOAM::Database& db, std::string name);

    static NeoFOAM::Collection& getCollection(NeoFOAM::Database& db, std::string name);

    static const NeoFOAM::Collection& getCollection(const NeoFOAM::Database& db, std::string name);

    static OldTimeCollection get(NeoFOAM::Database& db, std::string name);

private:

    std::shared_ptr<NeoFOAM::Collection> collection_;
};


template<typename geoField>
geoField& oldTime(geoField& field)
{
    FieldCollection fieldCollection(field);
    Document& fieldDoc = fieldCollection.get(field.key);

    std::size_t timeIdx = timeIndex(fieldDoc);

    std::string oldTimeName = field.name + "_0";

    std::vector<key> oldKeys =
        fieldCollection.find([oldTimeName, timeIdx](const Document& doc)
                             { return name(doc) == oldTimeName && timeIndex(doc) == timeIdx - 1; });

    bool found = (oldKeys.size() == 1);
    if (found)
    {
        Document& oldDoc = fieldCollection.get(oldKeys[0]);
        geoField& oldField = oldDoc.get<geoField&>("field");
        return oldField;
    }

    // create oldTime field
    geoField& oldField = fieldCollection.registerField<geoField>(CreateFromExistingField<geoField> {
        .name = oldTimeName,
        .field = field,
        .timeIndex = timeIdx - 1,
        .iterationIndex = iterationIndex(fieldDoc),
        .subCycleIndex = subCycleIndex(fieldDoc)
    });
    return oldField;
}


template<typename geoField>
const geoField& oldTime(const geoField& field)
{
    const Collection& fieldCollection = FieldCollection::getCollection(field);
    const Document& fieldDoc = fieldCollection.get(field.key);

    std::size_t timeIdx = timeIndex(fieldDoc);

    std::string oldTimeName = field.name + "_0";

    std::vector<key> oldKeys =
        fieldCollection.find([oldTimeName, timeIdx](const Document& doc)
                             { return name(doc) == oldTimeName && timeIndex(doc) == timeIdx - 1; });

    bool found = (oldKeys.size() == 1);
    if (found)
    {
        const Document& oldDoc = fieldCollection.get(oldKeys[0]);
        const geoField& oldField = oldDoc.get<const geoField&>("field");
        return oldField;
    }
    else
    {
        NF_THROW("Old field not found");
    }
}

} // namespace NeoFOAM
