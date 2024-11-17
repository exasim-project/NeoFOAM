#include "NeoFOAM/core/collection.hpp"


namespace NeoFOAM
{


Collection::Collection(std::string type, std::string name, Database& db)
    : documents_(), type_(type), name_(name), db_(db)
{}

Database& Collection::db() { return db_; }

const Database& Collection::db() const { return db_; }


std::string Collection::name() const { return name_; }

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

} // namespace NeoFOAM