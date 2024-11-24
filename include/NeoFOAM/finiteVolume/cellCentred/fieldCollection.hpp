// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <limits>
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

using CreateFunction = std::function<Document(NeoFOAM::Database& db)>;


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

    FieldCollection(std::shared_ptr<NeoFOAM::Collection> collection);

    template<class registeredGeoField>
    FieldCollection(registeredGeoField& field)
        : collection_(
            field.registered() ? field.db().getCollectionPtr(field.fieldCollectionName)
                               : throw std::runtime_error("field is not registered")
        )
    {}

    static const std::string typeName();

    static void create(NeoFOAM::Database& db, std::string name);

    static NeoFOAM::Collection& getCollection(NeoFOAM::Database& db, std::string name);

    static const NeoFOAM::Collection& getCollection(const NeoFOAM::Database& db, std::string name);

    template<class registeredGeoField>
    static const NeoFOAM::Collection& getCollection(const registeredGeoField& field)
    {
        const Database& db = field.db();
        return db.getCollection(field.fieldCollectionName);
    }

    static FieldCollection get(NeoFOAM::Database& db, std::string name);

    Document& get(const key& id);

    const Document& get(const key& id) const;

    std::vector<key> find(const std::function<bool(const Document&)>& predicate) const;

    size_t size() const;

    template<class Field>
    Field& registerField(CreateFunction createFunc)
    {
        auto doc = createFunc(collection_->db());
        if (!validateFieldDoc(doc))
        {
            throw std::runtime_error("Document is not valid");
        }

        auto key = collection_->insert(doc);
        auto& fieldDoc = collection_->get(key);
        auto& field = fieldDoc.get<Field&>("field");
        field.key = key;
        field.fieldCollectionName = collection_->name();
        return field;
    }

private:

    std::shared_ptr<NeoFOAM::Collection> collection_;
};


template<typename geoField>
class CreateFromExistingField
{
public:

    std::string name;
    const geoField& field;
    std::size_t timeIndex = std::numeric_limits<std::size_t>::max();
    std::size_t iterationIndex = std::numeric_limits<std::size_t>::max();
    std::int64_t subCycleIndex = std::numeric_limits<std::int64_t>::max();

    Document operator()(Database& db)
    {
        VolumeField<scalar> vf(
            field.exec(),
            name,
            field.mesh(),
            field.internalField(),
            field.boundaryConditions(),
            db,
            "",
            ""
        );
        if (timeIndex == std::numeric_limits<std::size_t>::max())
        {
            const Collection& fieldCollection = FieldCollection::getCollection(field);
            const Document& fieldDoc = fieldCollection.get(field.key);
            timeIndex = NeoFOAM::finiteVolume::cellCentred::timeIndex(fieldDoc);
            iterationIndex = NeoFOAM::finiteVolume::cellCentred::iterationIndex(fieldDoc);
            subCycleIndex = NeoFOAM::finiteVolume::cellCentred::subCycleIndex(fieldDoc);
        }
        return NeoFOAM::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            validateFieldDoc
        );
    }
};


} // namespace NeoFOAM
