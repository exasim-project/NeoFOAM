// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <optional>
#include <functional>
#include <atomic>
#include "NeoFOAM/core/dictionary.hpp"

namespace NeoFOAM
{

using key = std::string;
using DocumentValidator = std::function<bool(Dictionary)>;

bool hasId(Dictionary doc);


class Document : public Dictionary
{
public:

    Document();

    Document(const Dictionary& dict, DocumentValidator validator = hasId);

    bool validate() const;

    std::string id() const { return get<std::string>("id"); }

private:

    static std::string generateID()
    {
        static std::atomic<int> counter {0};
        return "doc_" + std::to_string(counter++);
    }
    std::string id_;
    DocumentValidator validator_;
};

const std::string& name(const NeoFOAM::Document& doc);

std::string& name(NeoFOAM::Document& doc);

} // namespace NeoFOAM
