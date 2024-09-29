#include "NeoFOAM/core/database.hpp"

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

Document::Document(const Dictionary& dict, std::function<bool(Dictionary)> validator)
    : Dictionary(dict), id_(generateID()), validator_(validator)
{
    if (!contains("id"))
    {
        insert("id", id_);
    }
    validate();
}

void Document::validate()
{
    if (!validator_(*this))
    {
        throw std::runtime_error("Document validation failed");
    }
}

Collection::Collection(std::string type, Database& db) : documents_(), type_(type), db_(db) {}

std::string Collection::type() const { return type_; }

std::string Collection::insert(Document doc)
{
    std::string id = doc.get<std::string>("id");
    documents_[id] = std::move(doc);
    return id;
}

Document& Collection::get(const std::string& id)
{
    auto it = documents_.find(id);
    if (it != documents_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Document not found");
}

const Document& Collection::get(const std::string& id) const
{
    auto it = documents_.find(id);
    if (it != documents_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Document not found");
}

void Collection::update(const std::string& id, const Document& doc) { documents_[id] = doc; }

void Collection::update(const Document& doc) { update(doc.id(), doc); }

std::vector<key> Collection::find(const std::function<bool(const Document&)>& predicate) const
{
    std::vector<key> result;
    for (const auto& [key, doc] : documents_)
    {
        if (predicate(doc))
        {
            result.push_back(doc.id());
        }
    }
    return result;
}

size_t Collection::size() const { return documents_.size(); }

void Database::createCollection(std::string name, std::string type)
{
    collections_.emplace(name, Collection(type, *this));
}

Collection& Database::getCollection(const std::string& name)
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Collection not found");
}

const Collection& Database::getCollection(const std::string& name) const
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Collection not found");
}

} // namespace NeoFOAM