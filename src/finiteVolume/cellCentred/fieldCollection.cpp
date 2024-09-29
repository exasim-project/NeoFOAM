// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

const std::string FieldCollection::typeName = "FieldCollection";

FieldCollection::FieldCollection(std::string name, NeoFOAM::Database& db)
    : name_(name), db_(db), collection_(db.getCollection(name))
{

}

void FieldCollection::registerCollection(std::string name, NeoFOAM::Database& db)
{

    FieldCollection::registerCollection("fieldCollection", db);

    db.createCollection(name, FieldCollection::typeName);
}



} // namespace NeoFOAM
