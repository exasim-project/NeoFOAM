// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/document.hpp"

namespace NeoFOAM
{

bool hasId(Dictionary doc) { return doc.contains("id"); }

Document::Document() : Dictionary(), id_(generateID()), validator_(hasId)
{
    if (!contains("id"))
    {
        insert("id", id_);
    }
}

Document::Document(const Dictionary& dict, DocumentValidator validator)
    : Dictionary(dict), id_(generateID()), validator_(validator)
{
    if (!contains("id"))
    {
        insert("id", id_);
    }
    validate();
}

bool Document::validate() const
{
    if (!validator_(*this))
    {
        throw std::runtime_error("Document validation failed");
    }
    return true;
}

const std::string& name(const NeoFOAM::Document& doc) { return doc.get<std::string>("name"); }

std::string& name(NeoFOAM::Document& doc) { return doc.get<std::string>("name"); }


} // namespace NeoFOAM
