// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <optional>
#include <string>
#include <NeoFOAM/core/database.hpp>
#include "NeoFOAM/core/document.hpp"

// auto

struct CustomTestDocument
{
    NeoFOAM::DocumentValidator customValidator = [](const NeoFOAM::Document& doc)
    { return doc.contains("name") && doc.contains("value") && hasId(doc); };

    static NeoFOAM::Document create(std::string name, double value)
    {
        return NeoFOAM::Document({{"name", name}, {"value", value}});
    }
};

const double& value(const NeoFOAM::Document& doc) { return doc.get<double>("value"); }

double& value(NeoFOAM::Document& doc) { return doc.get<double>("value"); }

class CustomTestCollection
{
public:
};
