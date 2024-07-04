// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <optional>

namespace NeoFOAM
{

struct Description
{
    std::string value;
};

struct MinLength
{
    size_t value;
};

struct MaxLength
{
    size_t value;
};

struct Pattern
{
    std::string value;
};

class String
{
public:

    std::optional<std::string> description;
    std::optional<size_t> minLength;
    std::optional<size_t> maxLength;
    std::optional<std::string> pattern;

    String() = default;

    // Set methods for each property
    void set(const Description& desc) { description = desc.value; }

    void set(const MinLength& min) { minLength = min.value; }

    void set(const MaxLength& max) { maxLength = max.value; }

    void set(const Pattern& pat) { pattern = pat.value; }
};


}

namespace nlohmann
{
void to_json(nlohmann::json& j, const NeoFOAM::String& schema);
}
