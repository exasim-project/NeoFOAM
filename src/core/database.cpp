#include "NeoFOAM/core/database.hpp"

namespace NeoFOAM
{

Document::Document()
    : Dictionary(), id_(generateID()), validator_(hasId)
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

std::string Collection::insert(Document doc) {
    std::string id = doc.get<std::string>("id");
    documents_[id] = std::move(doc);
    return id;
}

std::optional<Document> Collection::getDocument(const std::string& id) const {
    auto it = documents_.find(id);
    if (it != documents_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<Document> Collection::find(const std::function<bool(const Document&)>& predicate) const {
    std::vector<Document> result;
    for (const auto& [key,doc] : documents_) {
        if (predicate(doc)) {
            result.push_back(doc);
        }
    }
    return result;
}

size_t Collection::size() const {
    return documents_.size();
}

void Database::createCollection(const std::string& name) {
    collections_.emplace(name, Collection{});
}

std::optional<Collection> Database::getCollection(const std::string& name) const {
    auto it = collections_.find(name);
    if (it != collections_.end()) {
        return it->second;
    }
    return std::nullopt;
}

} // namespace NeoFOAM