// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <functional>
#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"

#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"
#include "NeoFOAM/core/database/document.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/core/database/fieldCollection.hpp"


namespace NeoFOAM::finiteVolume::cellCentred
{


// class OldTimeDocument
// {
// public:

//     std::string name;
//     std::size_t timeIndex;
//     std::size_t iterationIndex;
//     std::int64_t subCycleIndex;
//     std::any field;


//     // static Document create(FieldDocument fDoc);

//     Document doc();
// };


// class OldTimeCollection
// {
// public:

//     OldTimeCollection(std::shared_ptr<NeoFOAM::Collection> collection);

//     static const std::string typeName();

//     static void create(NeoFOAM::Database& db, std::string name);

//     static NeoFOAM::Collection& getCollection(NeoFOAM::Database& db, std::string name);

//     static const NeoFOAM::Collection& getCollection(const NeoFOAM::Database& db, std::string
//     name);

//     static OldTimeCollection get(NeoFOAM::Database& db, std::string name);

// private:

//     std::shared_ptr<NeoFOAM::Collection> collection_;
// };


template<typename geoField>
geoField& oldTime(geoField& field)
{
    FieldCollection& fieldCollection = FieldCollection::instance(field);
    FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);

    std::size_t timeIdx = fieldDoc.timeIndex();

    std::string oldTimeName = field.name + "_0";

    std::vector<std::string> oldKeys = fieldCollection.find(
        [oldTimeName, timeIdx](const Document& doc)
        {
            return doc.get<std::string>("name") == oldTimeName
                && doc.get<std::size_t>("timeIndex") == timeIdx - 1;
        }
    );

    bool found = (oldKeys.size() == 1);
    // print oldKeys
    for (auto key : fieldCollection.sortedKeys())
    {
    }
    for (auto key : oldKeys)
    {
        std::cout << "  -- " << key << std::endl;
    }
    if (found)
    {
        std::cout << "Found old field" << std::endl;
        FieldDocument& oldDoc = fieldCollection.fieldDoc(oldKeys[0]);
        geoField& oldField = oldDoc.field<geoField>();
        return oldField;
    }
    std::cout << "not Found old field" << std::endl;

    // create oldTime field
    geoField& oldField = fieldCollection.registerField<geoField>(CreateFromExistingField<geoField> {
        .name = oldTimeName,
        .field = field,
        .timeIndex = timeIdx - 1,
        .iterationIndex = fieldDoc.iterationIndex(),
        .subCycleIndex = fieldDoc.subCycleIndex()
    });
    return oldField;
}


template<typename geoField>
const geoField& oldTime(const geoField& field)
{
    const FieldCollection& fieldCollection = FieldCollection::instance(field);
    const FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);

    std::size_t timeIdx = fieldDoc.timeIndex();

    std::string oldTimeName = field.name + "_0";

    std::vector<std::string> oldKeys = fieldCollection.find(
        [oldTimeName, timeIdx](const Document& doc)
        {
            return doc.get<std::string>("name") == oldTimeName
                && doc.get<std::size_t>("timeIndex") == timeIdx - 1;
        }
    );

    bool found = (oldKeys.size() == 1);
    if (found)
    {
        const FieldDocument& oldDoc = fieldCollection.fieldDoc(oldKeys[0]);
        const geoField& oldField = oldDoc.field<geoField>();
        return oldField;
    }
    else
    {
        NF_THROW("Old field not found");
    }
}

} // namespace NeoFOAM
