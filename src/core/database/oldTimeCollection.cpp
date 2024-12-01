// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/oldTimeCollection.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// OldTimeCollection::OldTimeCollection(std::shared_ptr<NeoFOAM::Collection> collection)
//     : collection_(collection)
// {}

// void OldTimeCollection::create(NeoFOAM::Database& db, std::string name)
// {
//     db.createCollection(name, OldTimeCollection::typeName());
// }

// NeoFOAM::Collection& OldTimeCollection::getCollection(NeoFOAM::Database& db, std::string name)
// {
//     return db.getCollection(name);
// }

// const NeoFOAM::Collection&
// OldTimeCollection::getCollection(const NeoFOAM::Database& db, std::string name)
// {
//     return db.getCollection(name);
// }

// OldTimeCollection OldTimeCollection::get(NeoFOAM::Database& db, std::string name)
// {
//     return OldTimeCollection(db.getCollectionPtr(name));
// }

// const std::string OldTimeCollection::typeName() { return "OldTimeCollection"; }

} // namespace NeoFOAM::finiteVolume::cellCentred
