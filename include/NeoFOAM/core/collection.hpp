// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include "NeoFOAM/core/document.hpp"

namespace NeoFOAM
{

using key = std::string;


// forward declaration
class Database;

class Collection
{
public:

    template<typename CollectionType>
    Collection(CollectionType collection)
        : impl_(std::make_unique<Model<CollectionType>>(std::move(collection)))
    {}

    Collection(const Collection& other) : impl_(other.impl_->clone()) {}

    Document& get(const key& id);
    const Document& get(const key& id) const;
    std::vector<key> find(const std::function<bool(const Document&)>& predicate) const;
    size_t size() const;
    std::string type() const;
    std::string name() const;
    Database& db();
    const Database& db() const;

    template<typename CollectionType>
    CollectionType& as()
    {
        auto derived = dynamic_cast<Model<CollectionType>*>(impl_.get());
        if (!derived)
        {
            throw std::bad_cast();
        }
        return derived->collection_;
    }

private:

    struct Concept
    {
        virtual ~Concept() = default;
        virtual Document& get(const key& id) = 0;
        virtual const Document& get(const key& id) const = 0;
        virtual std::vector<key> find(const std::function<bool(const Document&)>& predicate
        ) const = 0;
        virtual size_t size() const = 0;
        virtual std::string type() const = 0;
        virtual std::string name() const = 0;
        virtual Database& db() = 0;
        virtual const Database& db() const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template<typename CollectionType>
    struct Model : Concept
    {
        Model(CollectionType collection) : collection_(std::move(collection)) {}

        Document& get(const key& id) override { return collection_.get(id); }

        const Document& get(const key& id) const override { return collection_.get(id); }

        std::vector<key> find(const std::function<bool(const Document&)>& predicate) const override
        {
            return collection_.find(predicate);
        }

        size_t size() const override { return collection_.size(); }

        std::string type() const override { return collection_.type(); }

        std::string name() const override { return collection_.name(); }

        Database& db() override { return collection_.db(); }

        const Database& db() const override { return collection_.db(); }

        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }

        CollectionType collection_;
    };

    std::unique_ptr<Concept> impl_;
};

template<typename DocumentType>
class CollectionMixin
{

public:

    CollectionMixin(NeoFOAM::Database& db, std::string name) : docs_(), db_(db), name_(name) {}

    Document& get(const key& id) { return docs_.at(id).doc(); }

    const Document& get(const key& id) const { return docs_.at(id).doc(); }

    std::vector<std::string> find(const std::function<bool(const Document&)>& predicate) const
    {
        std::vector<std::string> result;
        for (const auto& [key, doc] : docs_)
        {
            if (predicate(doc.doc()))
            {
                result.push_back(doc.id());
            }
        }
        return result;
    }

    std::size_t size() const { return docs_.size(); }

    const NeoFOAM::Database& db() const { return db_; }

    NeoFOAM::Database& db() { return db_; }

    const std::string& name() const { return name_; }

    std::string type() const { return DocumentType::typeName(); }


protected:

    std::unordered_map<key, DocumentType> docs_;
    std::string name_;
    NeoFOAM::Database& db_;
};

} // namespace NeoFOAM
