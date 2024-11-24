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


class FieldDocument
{
public:

    template<class geoField>
    FieldDocument(
        const geoField& field,
        std::size_t timeIndex,
        std::size_t iterationIndex,
        std::int64_t subCycleIndex
    )
        : doc_(
            Document(
                {{"name", field.name},
                 {"timeIndex", timeIndex},
                 {"iterationIndex", iterationIndex},
                 {"subCycleIndex", subCycleIndex},
                 {"field", field}}
            ),
            validateFieldDoc
        )
    {}

    FieldDocument(const Document& doc);

    Document& doc();

    const Document& doc() const;

    std::string id() const;

    static std::string typeName();

    template<class geoField>
    geoField& field()
    {
        return doc_.get<geoField&>("field");
    }

    template<class geoField>
    const geoField& field() const
    {
        return doc_.get<const geoField&>("field");
    }

    std::string name() const;

    std::string& name();

    std::size_t timeIndex() const;

    std::size_t& timeIndex();

    std::size_t iterationIndex() const;

    std::size_t& iterationIndex();

    std::int64_t subCycleIndex() const;

    std::int64_t& subCycleIndex();

private:

    Document doc_;
};

using CreateFunction = std::function<FieldDocument(NeoFOAM::Database& db)>;

class FieldCollection : public CollectionMixin<FieldDocument>
{
public:

    FieldCollection(NeoFOAM::Database& db, std::string name);

    bool contains(const std::string& id) const;

    std::string insert(const FieldDocument& cc);

    FieldDocument& fieldDoc(const key& id);

    const FieldDocument& fieldDoc(const key& id) const;

    static FieldCollection& instance(NeoFOAM::Database& db, std::string name);

    template<class geoField>
    static FieldCollection& instance(geoField& field)
    {
        return instance(field.db(), field.fieldCollectionName);
    }

    template<class geoField>
    static const FieldCollection& instance(const geoField& field)
    {
        const Database& db = field.db();
        const Collection& collection = db.getCollection(field.fieldCollectionName);
        return collection.as<FieldCollection>();
        // return instance(field.db(), field.fieldCollectionName);
    }

    template<class geoField>
    geoField& registerField(CreateFunction createFunc)
    {
        FieldDocument doc = createFunc(db());
        if (!validateFieldDoc(doc.doc()))
        {
            throw std::runtime_error("Document is not valid");
        }

        std::string key = insert(doc);
        FieldDocument& fd = fieldDoc(key);
        geoField& field = fd.field<geoField>();
        field.key = key;
        field.fieldCollectionName = name();
        return field;
    }
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

    FieldDocument operator()(Database& db)
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
        if (field.registered())
        {
            const FieldCollection& fieldCollection = FieldCollection::instance(field);
            const FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);
            if (timeIndex == std::numeric_limits<std::size_t>::max())
            {
                timeIndex = fieldDoc.timeIndex();
            }
            if (iterationIndex == std::numeric_limits<std::size_t>::max())
            {
                iterationIndex = fieldDoc.iterationIndex();
            }
            if (subCycleIndex == std::numeric_limits<std::int64_t>::max())
            {
                subCycleIndex = fieldDoc.subCycleIndex();
            }
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
