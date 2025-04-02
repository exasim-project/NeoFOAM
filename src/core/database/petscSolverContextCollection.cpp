// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/petscSolverContextCollection.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// Initialize the static member
bool validatContextDoc(const Document& doc)
{
    return doc.contains("eqnName") && doc.contains("context");
}

petscSolverContextDocument::petscSolverContextDocument(const Document& doc)
    : doc_(doc, validatContextDoc)
{}

std::string petscSolverContextDocument::id() const { return doc_.id(); }

std::string petscSolverContextDocument::typeName() { return "petscSolverContextDocument"; }

Document& petscSolverContextDocument::doc() { return doc_; }

const Document& petscSolverContextDocument::doc() const { return doc_; }

std::string petscSolverContextDocument::eqnName() const { return doc_.get<std::string>("eqnName"); }

std::string& petscSolverContextDocument::eqnName() { return doc_.get<std::string>("eqnName"); }


petscSolverContextCollection::petscSolverContextCollection(NeoFOAM::Database& db, std::string name)
    : NeoFOAM::CollectionMixin<petscSolverContextDocument>(db, name)
{}

bool petscSolverContextCollection::contains(const std::string& id) const
{
    return docs_.contains(id);
}

std::string petscSolverContextCollection::insert(const petscSolverContextDocument& cc)
{
    std::string id = cc.id();
    if (contains(id))
    {
        return "";
    }
    docs_.emplace(id, cc);
    return id;
}

petscSolverContextDocument&
petscSolverContextCollection::petscSolverContextDoc(const std::string& id)
{
    return docs_.at(id);
}

const petscSolverContextDocument&
petscSolverContextCollection::petscSolverContextDoc(const std::string& id) const
{
    return docs_.at(id);
}

petscSolverContextCollection&
petscSolverContextCollection::instance(NeoFOAM::Database& db, std::string name)
{
    NeoFOAM::Collection& col = db.insert(name, petscSolverContextCollection(db, name));
    return col.as<petscSolverContextCollection>();
}

const petscSolverContextCollection&
petscSolverContextCollection::instance(const NeoFOAM::Database& db, std::string name)
{
    const NeoFOAM::Collection& col = db.at(name);
    return col.as<petscSolverContextCollection>();
}

} // namespace NeoFOAM::finiteVolume::cellCentred
