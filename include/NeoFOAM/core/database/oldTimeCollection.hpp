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

class OldTimeDocument
{
public:

    OldTimeDocument(const Document& doc);

    OldTimeDocument(
        std::string nextTime,
        std::string previousTime,
        std::string currentTime,
        int32_t level
    );

    std::string& nextTime();

    const std::string& nextTime() const;

    std::string& previousTime();

    const std::string& previousTime() const;

    std::string& currentTime();

    const std::string& currentTime() const;

    int32_t& level();

    const int32_t& level() const;
  
    Document& doc();

    const Document& doc() const;

    std::string id() const;

    static std::string typeName();

private:

    Document doc_;
};


class OldTimeCollection : public CollectionMixin<OldTimeDocument>
{
public:

    OldTimeCollection(Database& db, std::string name, std::string fieldCollectionName);

    bool contains(const std::string& id) const;

    bool insert(const OldTimeDocument& cc);

    std::string findNextTime(std::string id) const;

    std::string findPreviousTime(std::string id) const;

    OldTimeDocument& oldTimeDoc(const std::string& id);

    const OldTimeDocument& oldTimeDoc(const std::string& id) const;

    template<typename geoField>
    geoField& getOrInsert(std::string IdOfNextField)
    {
        std::string nextId = findNextTime(IdOfNextField);
        FieldCollection& fieldCollection = FieldCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldField is already registered
        {
            OldTimeDocument& otDoc = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(otDoc.previousTime()).field<geoField>();
        }
        FieldDocument& fieldDoc = fieldCollection.fieldDoc(IdOfNextField);

        std::string oldTimeName = fieldDoc.field<geoField>().name + "_0";
        geoField& oldField = fieldCollection.registerField<geoField>(CreateFromExistingField<geoField> {
            .name = oldTimeName,
            .field = fieldDoc.field<geoField>(),
            .timeIndex = fieldDoc.timeIndex() - 1,
            .iterationIndex = fieldDoc.iterationIndex(),
            .subCycleIndex = fieldDoc.subCycleIndex()
        });
        OldTimeDocument oldTimeDoc(
            IdOfNextField,
            oldField.key,
            "",
            -1
        );
        insert(oldTimeDoc);
        return oldField;
    }

    static OldTimeCollection& instance(Database& db, std::string name, std::string fieldCollectionName);

    static const OldTimeCollection& instance(const Database& db, std::string name);

    static OldTimeCollection& instance(FieldCollection& fieldCollection);

    static const OldTimeCollection& instance(const FieldCollection& fieldCollection);

    private:

        std::string fieldCollectionName_;
};

/**
 * @brief Retrieves the old time field of a given field.
 *
 * This function retrieves the old time field of a given field by looking up the field's name and
 * time index in the FieldCollection.
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename geoField>
geoField& oldTime(geoField& field)
{
    FieldCollection& fieldCollection = FieldCollection::instance(field);
    OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);

    FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);

    std::int64_t timeIdx = fieldDoc.timeIndex();

    std::string oldTimeName = field.name + "_0";

    std::vector<std::string> oldKeys = fieldCollection.find(
        [oldTimeName, timeIdx](const Document& doc)
        {
            return doc.get<std::string>("name") == oldTimeName
                && doc.get<std::int64_t>("timeIndex") == timeIdx - 1;
        }
    );

    bool found = (oldKeys.size() == 1);
    if (found)
    {
        FieldDocument& oldDoc = fieldCollection.fieldDoc(oldKeys[0]);
        geoField& oldField = oldDoc.field<geoField>();
        return oldField;
    }

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

/**
 * @brief Retrieves the old time field of a given field (const version).
 *
 * This function retrieves the old time field of a given field by looking up the field's name and
 * time index in the FieldCollection.
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename geoField>
const geoField& oldTime(const geoField& field)
{
    const FieldCollection& fieldCollection = FieldCollection::instance(field);
    const FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);

    std::int64_t timeIdx = fieldDoc.timeIndex();

    std::string oldTimeName = field.name + "_0";

    std::vector<std::string> oldKeys = fieldCollection.find(
        [oldTimeName, timeIdx](const Document& doc)
        {
            return doc.get<std::string>("name") == oldTimeName
                && doc.get<std::int64_t>("timeIndex") == timeIdx - 1;
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
