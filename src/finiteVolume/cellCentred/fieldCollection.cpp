// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// const versions
std::size_t timeIndex(const NeoFOAM::Document& doc) { return doc.get<std::size_t>("timeIndex"); }

std::size_t iterationIndex(const NeoFOAM::Document& doc)
{
    return doc.get<std::size_t>("iterationIndex");
}

std::int64_t subCycleIndex(const NeoFOAM::Document& doc)
{
    return doc.get<std::int64_t>("subCycleIndex");
}

// non-const versions
std::size_t& timeIndex(NeoFOAM::Document& doc) { return doc.get<std::size_t>("timeIndex"); }

std::size_t& iterationIndex(NeoFOAM::Document& doc)
{
    return doc.get<std::size_t>("iterationIndex");
}

std::int64_t& subCycleIndex(NeoFOAM::Document& doc)
{
    return doc.get<std::int64_t>("subCycleIndex");
}

// Initialize the static member
bool validateFieldDoc(const Document& doc)
{
    return doc.contains("name") && doc.contains("timeIndex") && doc.contains("iterationIndex")
        && doc.contains("subCycleIndex") && hasId(doc) && doc.contains("field");
}

Document FieldDocument::create(FieldDocument fDoc) { return fDoc.doc(); }

Document FieldDocument::doc()
{
    return Document(
        {{"name", name},
         {"timeIndex", timeIndex},
         {"iterationIndex", iterationIndex},
         {"subCycleIndex", subCycleIndex},
         {"field", field}},
        validateFieldDoc
    );
}

const std::string FieldCollection::typeName() { return "FieldCollection"; }

void FieldCollection::create(NeoFOAM::Database& db, std::string name)
{
    db.createCollection(name, FieldCollection::typeName());
}

NeoFOAM::Collection& FieldCollection::getCollection(NeoFOAM::Database& db, std::string name)
{
    return db.getCollection(name);
}

const NeoFOAM::Collection&
FieldCollection::getCollection(const NeoFOAM::Database& db, std::string name)
{
    return db.getCollection(name);
}

} // namespace NeoFOAM::finiteVolume::cellCentred
